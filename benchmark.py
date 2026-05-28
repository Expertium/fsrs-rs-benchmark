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
from typing import Callable, List, Optional, TypeVar

import matplotlib.pyplot as plt
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

ParsedValue = TypeVar("ParsedValue", int, float)


def parse_interval(value: object, *, clamp_nonnegative: bool = False) -> float:
    if pd.isna(value):
        raise ValueError("Expected a numeric review history value, got missing data")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("Expected a numeric review history value, got empty text")
    parsed = float(value)
    return max(0.0, parsed) if clamp_nonnegative else parsed


def parse_rating(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Expected a rating history value, got missing data")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("Expected a rating history value, got empty text")
    return int(float(value))


def parse_history(
    history: object,
    parser: Callable[..., ParsedValue],
    *,
    clamp_nonnegative: bool = False,
    supports_clamp: bool = False,
) -> List[ParsedValue]:
    if pd.isna(history):
        return []
    values = [value.strip() for value in str(history).split(",") if value.strip()]
    if supports_clamp:
        return [parser(value, clamp_nonnegative=clamp_nonnegative) for value in values]
    return [parser(value) for value in values]


def build_reviews(row: pd.Series, *, include_current: bool = False):
    t_history = parse_history(
        row["t_history"], parse_interval, clamp_nonnegative=True, supports_clamp=True
    )
    r_history = parse_history(row["r_history"], parse_rating)
    if include_current:
        t_history = [*t_history, parse_interval(row["delta_t"], clamp_nonnegative=True)]
        r_history = [*r_history, parse_rating(row["rating"])]

    if len(t_history) != len(r_history):
        raise ValueError("Review history lengths do not match")

    return [
        FSRSReview(delta_t=delta_t, rating=rating)
        for delta_t, rating in zip(t_history, r_history)
    ]


def convert_to_items(df: pd.DataFrame) -> List[FSRSItem]:
    """Convert a DataFrame to a list of FSRSItem objects for fsrs-rs."""

    def accumulate(group):
        items = []
        for _, row in group.iterrows():
            items.append(
                (
                    row["review_th"],
                    FSRSItem(reviews=build_reviews(row, include_current=True)),
                )
            )
        return items

    result_list: list[FSRSItem] = sum(
        df.sort_values(by=["card_id", "review_th"])
        .groupby("card_id")[
            ["review_th", "t_history", "r_history", "delta_t", "rating"]
        ]
        .apply(accumulate)
        .tolist(),
        [],
    )
    result_list = list(map(lambda x: x[1], sorted(result_list, key=lambda x: x[0])))

    return result_list


def default_parameters() -> List[float]:
    return list(DEFAULT_PARAMETERS)


def train(train_set: pd.DataFrame) -> List[float]:
    """Train FSRS-rs on training data and return optimized weights."""
    backend = FSRS(parameters=[])
    train_set_items = convert_to_items(train_set)
    return list(map(lambda x: round(x, 4), backend.benchmark(train_set_items)))


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
    testset_copy["stability"] = [state.stability for state in memory_states]
    testset_copy["difficulty"] = [state.difficulty for state in memory_states]
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
    """Process a single user with the FSRS-rs benchmark."""
    del device_id
    plt.close("all")

    data_loader = UserDataLoader(config)
    dataset = data_loader.load_user_data(user_id)

    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)

    for split_i, (train_index, test_index) in enumerate(tscv.split(dataset)):
        if not config.train_equals_test:
            train_set = dataset.iloc[train_index]
            test_set = dataset.iloc[test_index]
            if config.equalize_test_with_non_secs:
                train_set = dataset[dataset[f"{split_i}_train"]]
                test_set = dataset[dataset[f"{split_i}_test"]]
        else:
            train_set = dataset.copy()
            test_set = dataset[
                dataset["review_th"] >= dataset.iloc[test_index]["review_th"].min()
            ].copy()

        if config.no_test_same_day:
            test_set = test_set[test_set["elapsed_days"] > 0].copy()
        if config.no_train_same_day:
            train_set = train_set[train_set["elapsed_days"] > 0].copy()

        if train_set.empty or test_set.empty:
            continue

        testsets.append(test_set)
        try:
            weights = (
                default_parameters()
                if config.default_params
                else train(train_set)
            )
        except Exception as exc:
            if str(exc).endswith("inadequate."):
                if config.verbose_inadequate_data:
                    print("Skipping - Inadequate data")
                weights = default_parameters()
            else:
                print(f"User: {user_id}")
                raise exc
        w_list.append(weights)

        if config.train_equals_test:
            break

    p = []
    y = []
    save_tmp = []

    for weights, testset in zip(w_list, testsets):
        p_partition, y_partition, testset_pred = predict(testset, weights)
        p.extend(p_partition)
        y.extend(y_partition)
        save_tmp.append(testset_pred)

    save_tmp_df = pd.concat(save_tmp)
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
    unprocessed_users = []
    dataset = pq.ParquetDataset(config.data_path / "revlogs")
    Path(f"evaluation/{config.get_evaluation_file_name()}").mkdir(
        parents=True, exist_ok=True
    )
    Path("result").mkdir(parents=True, exist_ok=True)
    Path("raw").mkdir(parents=True, exist_ok=True)
    result_file = Path(f"result/{config.get_evaluation_file_name()}.jsonl")
    raw_file = Path(f"raw/{config.get_evaluation_file_name()}.jsonl")
    if result_file.exists():
        data = sort_jsonl(result_file)
        processed_user = set(map(lambda x: x["user"], data))
    else:
        processed_user = set()

    if config.save_raw_output and raw_file.exists():
        sort_jsonl(raw_file)

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

    sort_jsonl(result_file)
    if config.save_raw_output:
        sort_jsonl(raw_file)
