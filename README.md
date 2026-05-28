# fsrs-rs-benchmark
A version of srs-benchmark ONLY for FSRS-rs

## How to run the benchmark

### Requirements

Dataset: [open-spaced-repetition/anki-revlogs-10k](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k)

Dependencies:

```bash
uv sync
```

> uv is a tool that helps manage Python environments and dependencies. You can install it from https://docs.astral.sh/uv/.

### Commands

Basic run:

```bash
uv run benchmark.py
```

Evaluate default parameters (no per-user training):

```bash
uv run benchmark.py --default
```

Set the number of worker processes:

```bash
uv run benchmark.py --processes 4
```

Limit processing to the first N users:

```bash
uv run benchmark.py --max-user-id 100
```

Save raw per-review predictions:

```bash
uv run benchmark.py --raw
```

### benchmark.py options

Run `uv run benchmark.py --help` for the full list. Common options include:

| Flag | Description | Default |
| --- | --- | --- |
| `--processes` | Number of worker processes. | `8` |
| `--data` | Path to `revlogs/*.parquet`. | `../anki-revlogs-10k` |
| `--max-user-id` | Maximum user ID to process (inclusive). No limit if unset. | unset |
| `--default` | Evaluate default parameters without per-user training. | off |
| `--n_splits` | Number of TimeSeriesSplit folds. | `5` |
| `--train_equals_test` | Train and test on the same data (measures overfitting). | off |
| `--secs` | Use `elapsed_seconds` as the interval instead of days. | off |
| `--no_test_same_day` | Exclude reviews with `elapsed_days=0` from the test set. | off |
| `--no_train_same_day` | Exclude reviews with `elapsed_days=0` from the training set. | off |
| `--equalize_test_with_non_secs` | Test only on reviews that would be included in non-secs tests. | off |
| `--raw` | Save raw per-review predictions to `raw/<name>.jsonl`. | off |
| `--file` | Save per-user evaluation results to `evaluation/<name>/`. | off |
