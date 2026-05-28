import json
import numpy as np
import traceback
import torch
from pathlib import Path
from sklearn.metrics import root_mean_squared_error  # type: ignore
from functools import wraps
from itertools import accumulate
from numbers import Real
from typing import Any, Hashable, Mapping, TypeAlias, cast
from config import Config

ParameterList: TypeAlias = list[float]
TorchStateDict: TypeAlias = Mapping[str, Any]
ModelState: TypeAlias = ParameterList | TorchStateDict
PartitionedModelState: TypeAlias = dict[Hashable, ModelState]
TrainingState: TypeAlias = ModelState | PartitionedModelState


def catch_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs), None
        except Exception:
            # Try to extract user_id from function arguments
            user_id = None
            if args:
                # Assume user_id is the first argument
                user_id = args[0]
            elif "user_id" in kwargs:
                user_id = kwargs["user_id"]

            # Include user_id in the error message if available
            error_msg = traceback.format_exc()
            if user_id is not None:
                error_msg = f"User {user_id}:\n{error_msg}"

            return None, error_msg

    return wrapper


def mean_bias_error(y, p):
    return np.mean(np.array(p) - np.array(y))


def rmse_matrix(df):
    tmp = df.copy()
    tmp["delta_t"] = tmp["elapsed_days"].map(
        lambda x: round(
            2.48 * np.power(3.62, np.floor(np.log(max(x, 1e-6)) / np.log(3.62))), 2
        )
    )
    tmp["i"] = tmp["i"].map(
        lambda x: round(1.99 * np.power(1.89, np.floor(np.log(x) / np.log(1.89))), 0)
    )
    tmp["rmse_bins_lapse"] = tmp["rmse_bins_lapse"].map(
        lambda x: (
            round(1.65 * np.power(1.73, np.floor(np.log(x) / np.log(1.73))), 0)
            if x != 0
            else 0
        )
    )
    if "weights" not in tmp.columns:
        tmp["weights"] = 1
    tmp = (
        tmp.groupby(["delta_t", "i", "rmse_bins_lapse"])
        .agg({"y": "mean", "p": "mean", "weights": "sum"})
        .reset_index()
    )
    return root_mean_squared_error(tmp["y"], tmp["p"], sample_weight=tmp["weights"])


def cum_concat(x):
    """Concatenate a list of lists using accumulate.

    Args:
        x: A list of lists to be concatenated

    Returns:
        A list of accumulated concatenated lists
    """
    return list(accumulate(x))


def save_evaluation_file(user_id, df, config: Config):
    if config.save_evaluation_file:
        df.to_csv(
            f"evaluation/{config.get_evaluation_file_name()}/{user_id}.tsv",
            sep="\t",
            index=False,
        )


def evaluate(y, p, df, file_name, user_id, config: Config, w_list=None):
    """
    Evaluate model predictions and generate statistics.

    Args:
        y: True labels
        p: Predicted probabilities
        df: DataFrame with predictions
        file_name: Name for output files
        user_id: User ID
        config: Configuration object
        w_list: Optional list of model weights

    Returns:
        tuple: (stats dict, raw predictions dict or None)
    """
    from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score
    from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
    import relplot

    p_calibrated = lowess(
        y, p, it=0, delta=0.01 * (max(p) - min(p)), return_sorted=False
    )
    ici = np.mean(np.abs(p_calibrated - p))
    rmse_raw = root_mean_squared_error(y_true=y, y_pred=p)
    logloss = log_loss(y_true=y, y_pred=p, labels=[0, 1])
    rmse_bins = rmse_matrix(df)
    mbe = mean_bias_error(y, p)
    smECE = relplot.smECE(np.array(p), np.array(y))
    y_hat_90 = (np.array(p) >= 0.9).astype(int)
    precision_90 = precision_score(y, y_hat_90, zero_division=0)
    recall_90 = recall_score(y, y_hat_90, zero_division=0)
    try:
        auc = round(roc_auc_score(y_true=y, y_score=p), 6)
    except Exception:
        auc = None
    stats = {
        "metrics": {
            "RMSE": round(rmse_raw, 6),
            "LogLoss": round(logloss, 6),
            "RMSE(bins)": round(rmse_bins, 6),
            "smECE": round(smECE, 6),
            "AUC": auc,
            "precision@90": round(precision_90, 6),
            "recall@90": round(recall_90, 6),
            "ICI": round(ici, 6),
            "MBE": round(mbe, 6),
        },
        "user": int(user_id),
        "size": len(y),
    }
    if w_list:
        parameters = result_parameters(w_list[-1])
        if parameters is not None:
            cast(Any, stats)["parameters"] = parameters
        elif config.save_weights:
            save_model_state(w_list[-1], file_name, user_id)
    if config.save_raw_output:
        raw = {
            "user": int(user_id),
            "p": list(map(lambda x: round(x, 4), p)),
            "y": list(map(int, y)),
        }
    else:
        raw = None
    return stats, raw


def is_parameter_list(state: Any) -> bool:
    return isinstance(state, list) and all(isinstance(x, Real) for x in state)


def rounded_parameter_list(state: ParameterList) -> ParameterList:
    return [round(float(x), 6) for x in state]


def result_parameters(
    state: TrainingState,
) -> ParameterList | dict[str, ParameterList] | None:
    if is_parameter_list(state):
        return rounded_parameter_list(cast(ParameterList, state))

    if isinstance(state, dict) and all(is_parameter_list(w) for w in state.values()):
        partition_state = state
        return {
            str(partition): rounded_parameter_list(cast(ParameterList, w))
            for partition, w in partition_state.items()
        }

    return None


def save_model_state(state: TrainingState, file_name: str, user_id: int) -> None:
    Path(f"weights/{file_name}").mkdir(parents=True, exist_ok=True)
    torch.save(state, f"weights/{file_name}/{user_id}.pth")


def sort_jsonl(file):
    data = list(map(lambda x: json.loads(x), open(file, encoding="utf-8").readlines()))
    data.sort(key=lambda x: x["user"])
    with file.open("w", encoding="utf-8", newline="\n") as jsonl_file:
        for json_data in data:
            jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
    return data
