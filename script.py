import json
import importlib.util
import sys
import types


def _bootstrap_models_package() -> None:
    package_name = "models"
    if package_name in sys.modules:
        return

    package = types.ModuleType(package_name)
    from pathlib import Path as _Path

    package.__path__ = [str(_Path(__file__).resolve().parent / package_name)]
    sys.modules[package_name] = package

    module_name = f"{package_name}.fsrs_rs"
    module_path = _Path(__file__).resolve().parent / package_name / "fsrs_rs.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


_bootstrap_models_package()

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq  # type: ignore
import torch
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from config import create_parser, Config
from data_loader import UserDataLoader
from models.fsrs_rs import FSRSRsBackend
from utils import catch_exceptions, evaluate, save_evaluation_file, sort_jsonl

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

    fsrs_rs = FSRSRsBackend(config)
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
                FSRSRsBackend.default_parameters()
                if config.default_params
                else fsrs_rs.train(train_set)
            )
        except Exception as exc:
            if str(exc).endswith("inadequate."):
                if config.verbose_inadequate_data:
                    print("Skipping - Inadequate data")
                weights = FSRSRsBackend.default_parameters()
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
        p_partition, y_partition, testset_pred = fsrs_rs.predict(testset, weights)
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
