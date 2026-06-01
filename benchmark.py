import json
import sys
from pathlib import Path as _Path

# Ensure the repo root is first on sys.path so local packages (config, utils, …)
# are found even when benchmark.py is invoked from another directory.
_REPO_ROOT = str(_Path(__file__).resolve().parent)
if not sys.path or sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore
import torch
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from config import create_parser, Config
from data_loader import UserDataLoader
from utils import catch_exceptions, evaluate, save_evaluation_file, sort_jsonl
from fsrs_rs_python import FSRS, FSRSItem, FSRSReview, DEFAULT_PARAMETERS  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# FSRS-rs helpers (formerly models/fsrs_rs.py)
# ---------------------------------------------------------------------------

MIN_STABILITY = 0.0001
MIN_RETRIEVABILITY = 0.0001
MAX_RETRIEVABILITY = 0.9999

FSRS7_DECAY1_INDEX = 27
FSRS7_DECAY2_INDEX = 28
FSRS7_BASE1_INDEX = 29
FSRS7_BASE2_INDEX = 30
FSRS7_WEIGHT1_INDEX = 31
FSRS7_WEIGHT2_INDEX = 32
FSRS7_S_WEIGHT_POWER1_INDEX = 33
FSRS7_S_WEIGHT_POWER2_INDEX = 34


def _parse_scalar(value: object, type_name: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Expected a {type_name} history value, got missing data")
    result = str(value).strip()
    if not result:
        raise ValueError(f"Expected a {type_name} history value, got empty text")
    return result


def parse_interval(value: object) -> float:
    """Parse a single delta_t field, clamped to be non-negative."""
    return max(0.0, float(_parse_scalar(value, "numeric review")))


def parse_rating(value: object) -> int:
    return int(float(_parse_scalar(value, "rating")))


def parse_history(history: object, parser) -> list:
    """Split a comma-separated history string and parse each non-empty field."""
    if pd.isna(history):
        return []
    fields = (field.strip() for field in str(history).split(","))
    return [parser(field) for field in fields if field]


def build_reviews(row: pd.Series, *, include_current: bool = False) -> List[FSRSReview]:
    t_history = parse_history(row["t_history"], parse_interval)
    r_history = parse_history(row["r_history"], parse_rating)
    if include_current:
        t_history.append(parse_interval(row["delta_t"]))
        r_history.append(parse_rating(row["rating"]))
    if len(t_history) != len(r_history):
        raise ValueError("Review history lengths do not match")
    return [FSRSReview(delta_t=t, rating=r) for t, r in zip(t_history, r_history)]


def convert_to_items(df: pd.DataFrame) -> List[FSRSItem]:
    """Convert a DataFrame to FSRSItems for fsrs-rs, ordered globally by review_th."""
    pairs = []  # (review_th, FSRSItem)
    for _, group in df.sort_values(by=["card_id", "review_th"]).groupby("card_id"):
        for _, row in group.iterrows():
            item = FSRSItem(reviews=build_reviews(row, include_current=True))
            pairs.append((row["review_th"], item))
    pairs.sort(key=lambda pair: pair[0])
    return [item for _, item in pairs]


def default_parameters() -> List[float]:
    return list(DEFAULT_PARAMETERS)


def train(train_set: pd.DataFrame) -> List[float]:
    """Train FSRS-rs on training data and return optimized weights."""
    backend = FSRS(parameters=[])
    items = convert_to_items(train_set)
    return [round(w, 4) for w in backend.benchmark(items)]


def predict(
    testset: pd.DataFrame, weights: List[float]
) -> tuple[List[float], List[float], pd.DataFrame]:
    """Return (predictions, labels, testset_with_predictions)."""

    def fsrs7_forgetting_curve(delta_t: pd.Series, stability: pd.Series) -> pd.Series:
        stability_array = stability.clip(lower=MIN_STABILITY).to_numpy(dtype=float, copy=False)
        delta_t_array = delta_t.clip(lower=0).to_numpy(dtype=float, copy=False)
        t_over_s = delta_t_array / stability_array

        decay1 = -weights[FSRS7_DECAY1_INDEX]
        decay2 = -weights[FSRS7_DECAY2_INDEX]
        base1 = weights[FSRS7_BASE1_INDEX]
        base2 = weights[FSRS7_BASE2_INDEX]

        if decay1 == 0 or decay2 == 0:
            raise ValueError("FSRS-7 decay parameters must be non-zero")

        factor1 = base1 ** (1 / decay1) - 1
        factor2 = base2 ** (1 / decay2) - 1
        r1 = (1 + factor1 * t_over_s) ** decay1
        r2 = (1 + factor2 * t_over_s) ** decay2

        weight1 = weights[FSRS7_WEIGHT1_INDEX] * stability_array ** (
            -weights[FSRS7_S_WEIGHT_POWER1_INDEX]
        )
        weight2 = weights[FSRS7_WEIGHT2_INDEX] * stability_array ** weights[
            FSRS7_S_WEIGHT_POWER2_INDEX
        ]

        return pd.Series(
            np.clip(
                (weight1 * r1 + weight2 * r2) / (weight1 + weight2),
                MIN_RETRIEVABILITY,
                MAX_RETRIEVABILITY,
            ),
            index=stability.index,
        )

    predictor = FSRS(parameters=weights)
    testset_copy = testset.copy()
    history_items = [FSRSItem(reviews=build_reviews(row)) for _, row in testset_copy.iterrows()]
    memory_states = predictor.memory_state_batch(history_items)
    testset_copy["stability"] = [s.stability for s in memory_states]
    testset_copy["difficulty"] = [s.difficulty for s in memory_states]
    testset_copy["p"] = fsrs7_forgetting_curve(testset_copy["delta_t"], testset_copy["stability"])

    p = testset_copy["p"].tolist()
    y = testset_copy["y"].tolist()

    return p, y, testset_copy


parser = create_parser()
args, _ = parser.parse_known_args()
if args.algo == parser.get_default("algo"):
    args.algo = "FSRS-rs"
elif args.algo != "FSRS-rs":
    raise ValueError("fsrs-rs-benchmark only supports --algo FSRS-rs")
config = Config(args)
config.partitions = "none"

torch.manual_seed(config.seed)
tqdm.pandas()


@catch_exceptions
def process(user_id: int, device_id: Optional[int] = None) -> tuple[dict, Optional[dict]]:
    """Train and evaluate FSRS-rs for a single user across time-series splits."""
    del device_id

    dataset = UserDataLoader(config).load_user_data(user_id)

    def get_weights(train_set: pd.DataFrame) -> List[float]:
        if config.default_params:
            return default_parameters()
        try:
            return train(train_set)
        except Exception as exc:
            if str(exc).endswith("inadequate."):
                if config.verbose_inadequate_data:
                    print("Skipping - Inadequate data")
                return default_parameters()
            print(f"User: {user_id}")
            raise

    # One (weights, testset) pair per time-series split.
    w_list: List[List[float]] = []
    testsets: List[pd.DataFrame] = []
    for split_i, (train_index, test_index) in enumerate(
        TimeSeriesSplit(n_splits=config.n_splits).split(dataset)
    ):
        if config.train_equals_test:
            train_set = dataset.copy()
            test_set = dataset[
                dataset["review_th"] >= dataset.iloc[test_index]["review_th"].min()
            ].copy()
        else:
            train_set = dataset.iloc[train_index]
            test_set = dataset.iloc[test_index]
            if config.equalize_test_with_non_secs:
                train_set = dataset[dataset[f"{split_i}_train"]]
                test_set = dataset[dataset[f"{split_i}_test"]]
        if config.no_test_same_day:
            test_set = test_set[test_set["elapsed_days"] > 0].copy()
        if config.no_train_same_day:
            train_set = train_set[train_set["elapsed_days"] > 0].copy()
        if train_set.empty or test_set.empty:
            continue
        testsets.append(test_set)
        w_list.append(get_weights(train_set))
        if config.train_equals_test:
            break

    # Predict each split's testset and accumulate predictions/labels.
    p: List[float] = []
    y: List[float] = []
    predictions = []
    for weights, testset in zip(w_list, testsets):
        p_partition, y_partition, testset_pred = predict(testset, weights)
        p.extend(p_partition)
        y.extend(y_partition)
        predictions.append(testset_pred)
    save_tmp_df = pd.concat(predictions)
    if "tensor" in save_tmp_df:
        del save_tmp_df["tensor"]

    save_evaluation_file(user_id, save_tmp_df, config)
    stats, raw = evaluate(
        y, p, save_tmp_df, config.get_evaluation_file_name(), user_id, config, w_list
    )
    stats["metrics"] = {"LogLoss": stats["metrics"]["LogLoss"]}
    return stats, raw


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    dataset = pq.ParquetDataset(config.data_path / "revlogs")
    Path(f"evaluation/{config.get_evaluation_file_name()}").mkdir(parents=True, exist_ok=True)
    Path("result").mkdir(parents=True, exist_ok=True)
    Path("raw").mkdir(parents=True, exist_ok=True)
    result_file = Path(f"result/{config.get_evaluation_file_name()}.jsonl")
    raw_file = Path(f"raw/{config.get_evaluation_file_name()}.jsonl")
    if result_file.exists():
        processed_user = {row["user"] for row in sort_jsonl(result_file)}
    else:
        processed_user = set()

    if config.save_raw_output and raw_file.exists():
        sort_jsonl(raw_file)

    unprocessed_users = []
    for user_id in dataset.partitioning.dictionaries[0]:
        user_id_value = user_id.as_py()
        if config.max_user_id is not None and user_id_value > config.max_user_id:
            continue
        if user_id_value in processed_user:
            continue
        unprocessed_users.append(user_id_value)
    unprocessed_users.sort()

    with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
        futures = [executor.submit(process, user_id, None) for user_id in unprocessed_users]
        for future in (
            pbar := tqdm(as_completed(futures), total=len(futures), smoothing=0.03)
        ):
            try:
                result, error = future.result()
                if error:
                    tqdm.write(str(error))
                else:
                    stats, raw = result
                    with open(result_file, "a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(stats, ensure_ascii=False) + "\n")
                    if raw:
                        with open(raw_file, "a", encoding="utf-8", newline="\n") as f:
                            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
                    pbar.set_description(f"Processed {stats['user']}")
            except Exception as e:
                tqdm.write(str(e))

    data = sort_jsonl(result_file)
    # Dataset size must stay fixed for the canonical run (constraint 5), so review
    # preprocessing can't silently change. (benchmark.py counts reviews, not items.)
    if config.max_user_id == 50:
        n_users = len(data)
        total_reviews = sum(d["size"] for d in data)
        assert n_users == 50, f"expected 50 users, got {n_users}"
        assert total_reviews == 1_581_505, f"expected 1,581,505 reviews, got {total_reviews:,}"
    if config.save_raw_output:
        sort_jsonl(raw_file)
