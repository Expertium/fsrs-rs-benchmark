import argparse
import re
import torch
from pathlib import Path
from typing import List, Optional, Literal, get_args

ModelName = Literal[
    # FSRS family
    "FSRSv1",
    "FSRSv2",
    "FSRSv3",
    "FSRSv4",
    "FSRS-4.5",
    "FSRS-5",
    "FSRS-6",
    "FSRS-6-one-step",
    "FSRS-7",
    "FSRS-rs",
    # Neural networks
    "RNN",
    "GRU",
    "LSTM",
    "Transformer",
    "NN-17",
    # Memory models
    "SM2",
    "SM2-trainable",
    "Ebisu-v2",
    "HLR",
    "ACT-R",
    "Anki",
    # DASH variants
    "DASH",
    "DASH[MCM]",
    "DASH[ACT-R]",
    # Other models
    "AVG",
    "RMSE-BINS-EXPLOIT",
    "MOVING-AVG",
    "90%",
    "LogisticRegression",
]


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processes", default=8, type=int, help="set the number of processes"
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma/space-separated CUDA device IDs to use (e.g., '0,1' or 'all')",
    )
    parser.add_argument("--dev", action="store_true", help="for local development")

    parser.add_argument(
        "--max-user-id",
        type=int,
        default=None,
        help="maximum user ID to process (inclusive)",
    )

    parser.add_argument(
        "--partitions",
        default="none",
        choices=["none", "deck", "preset"],
        help="use partitions instead of presets",
    )
    parser.add_argument(
        "--recency", action="store_true", help="enable recency weighting"
    )
    parser.add_argument(
        "--default", action="store_true", help="evaluate default parameters"
    )
    parser.add_argument(
        "--S0", action="store_true", help="FSRS-5/FSRS-6 with only S0 initialization"
    )
    parser.add_argument(
        "--sched_penalties",
        default=False,
        action="store_true",
        help="Enable FSRS-7 scheduling penalties (penalty 1 & 2). L2 penalty is always on. (default: False)",
    )

    # download revlogs from huggingface
    parser.add_argument(
        "--data",
        default="../anki-revlogs-10k",
        help="path to revlogs/*.parquet",
    )

    # short-term memory research
    parser.add_argument(
        "--secs", action="store_true", help="use elapsed_seconds as interval"
    )
    parser.add_argument(
        "--duration",
        action="store_true",
        help="enable duration feature when training LSTM",
    )

    parser.add_argument(
        "--no_test_same_day",
        action="store_true",
        help="exclude reviews with elapsed_days=0 from testset",
    )
    parser.add_argument(
        "--no_train_same_day",
        action="store_true",
        help="exclude reviews with elapsed_days=0 from trainset",
    )

    parser.add_argument(
        "--equalize_test_with_non_secs",
        action="store_true",
        help="Only test with reviews that would be included in non-secs tests",
    )

    # save detailed results
    parser.add_argument("--raw", action="store_true", help="save raw predictions")
    parser.add_argument(
        "--file", action="store_true", help="save evaluation results to file"
    )

    parser.add_argument("--algo", default="FSRSv3", help="algorithm name")
    parser.add_argument(
        "--short", action="store_true", help="include short-term reviews"
    )
    parser.add_argument(
        "--weights", action="store_true", help="save neural network weights"
    )
    parser.add_argument(
        "--train_equals_test",
        action="store_true",
        help="Set train set equal to test set without splitting",
    )
    parser.add_argument(
        "--n_splits", type=int, default=5, help="Number of splits for TimeSeriesSplit"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training neural models",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=64,
        help="Maximum sequence length for batching inputs",
    )
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=1,
        help="Number of threads for PyTorch intra-op parallelism",
    )
    return parser


class Config:
    """Holds all application configurations derived from command-line arguments and defaults."""

    def __init__(self, args: argparse.Namespace):
        # Store raw args for reference if needed, though direct access should be minimized
        self._raw_args: argparse.Namespace = args

        # Basic arguments from parser
        self.dev_mode: bool = args.dev
        self.default_params: bool = args.default
        self.model_name: ModelName = args.algo
        self.max_user_id: Optional[int] = args.max_user_id
        self.use_secs_intervals: bool = args.secs
        self.lstm_use_duration: bool = args.duration
        self.no_test_same_day: bool = args.no_test_same_day
        self.no_train_same_day: bool = args.no_train_same_day
        self.equalize_test_with_non_secs: bool = args.equalize_test_with_non_secs
        self.only_S0: bool = args.S0
        self.sched_penalties: bool = args.sched_penalties  # only for FSRS-7
        self.save_evaluation_file: bool = args.file
        self.save_weights: bool = args.weights
        self.partitions: str = args.partitions
        self.save_raw_output: bool = args.raw
        self.num_processes: int = args.processes
        self.data_path: Path = Path(args.data)
        self.use_recency_weighting: bool = args.recency
        self.train_equals_test: bool = args.train_equals_test

        def _parse_cuda_devices(raw: Optional[str]) -> Optional[List[int]]:
            if raw is None:
                return None
            value = raw.strip()
            if not value:
                return None
            value_lower = value.lower()
            if value_lower in {"all", "*"}:
                if not torch.cuda.is_available():
                    return []
                return list(range(torch.cuda.device_count()))

            def _parse_part(part: str) -> int:
                try:
                    device_id = int(part)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid CUDA device id '{part}'. Use comma/space-separated integers."
                    ) from exc
                if device_id < 0:
                    raise ValueError("CUDA device IDs must be >= 0.")
                return device_id

            return list(map(_parse_part, filter(None, re.split(r"[,\s]+", value))))

        self.cuda_device_ids: Optional[List[int]] = _parse_cuda_devices(args.gpus)

        # Training/data parameters from parser (with defaults)
        self.n_splits: int = args.n_splits
        self.max_seq_len: int = args.max_seq_len
        self.include_short_term: bool = args.short

        # PyTorch threading settings
        self.torch_num_threads: int = args.torch_num_threads
        torch.set_num_threads(self.torch_num_threads)
        # if hasattr(torch, "set_num_interop_threads"):
        #     torch.set_num_interop_threads(args.torch_num_interop_threads)

        def _validate_model():
            if self.model_name not in get_args(ModelName):
                raise ValueError(
                    f"Model name '{self.model_name}' must be one of {get_args(ModelName)}"
                )

        def _select_device():
            if all([torch.cuda.is_available(), self.model_name in [
                "GRU", "LSTM", "RNN", "NN-17", "Transformer",
            ]]):
                return torch.device("cuda")
            if all([torch.backends.mps.is_available(), self.model_name == "LSTM"]):
                return torch.device("mps")
            return torch.device("cpu")

        _validate_model()
        # Device configuration
        self.device: torch.device = _select_device()

        # Verbosity
        self.verbose_inadequate_data: bool = False

        # Derived file names
        _file_name_parts: list[str] = [self.model_name]
        _suffix_conditions = [
            (self.default_params, "-default"),
            (self.only_S0, "-S0"),
            (self.sched_penalties, "-sched_penalties"),
            (self.include_short_term, "-short"),
            (self.use_secs_intervals, "-secs"),
            (all([self.model_name == "LSTM", self.lstm_use_duration]), "-duration"),
            (self.use_recency_weighting, "-recency"),
            (self.no_test_same_day, "-no_test_same_day"),
            (self.no_train_same_day, "-no_train_same_day"),
            (self.equalize_test_with_non_secs, "-equalize_test_with_non_secs"),
            (self.train_equals_test, "-train_equals_test"),
            (self.partitions != "none", f"-{self.partitions}"),
            (self.dev_mode, "-dev"),
        ]
        _file_name_parts += list(map(lambda cs: cs[1], filter(lambda cs: cs[0], _suffix_conditions)))

        self.base_file_name: str = "".join(_file_name_parts)

        # Seed for reproducibility
        self.seed: int = 42

    def get_evaluation_file_name(self) -> str:
        """Return the derived output file stem used for evaluation artifacts."""
        return self.base_file_name

    def __repr__(self) -> str:
        """Provides a string representation of the configuration."""
        def _inner() -> str:
            attrs = {
                k: v
                for k, v in self.__dict__.items()
                if not k.startswith("_")
            }
            return f"Config({attrs})"
        return _inner()
