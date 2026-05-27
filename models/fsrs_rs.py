"""
FSRS-rs model wrapper for integration with the other.py architecture.

This module provides utilities to work with the fsrs-rs (Rust-based FSRS implementation)
within the benchmark framework.
"""

from typing import Callable, List, Optional, TypeVar
import numpy as np
import pandas as pd
from config import Config

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
) -> List[ParsedValue]:
    if pd.isna(history):
        return []
    values = [value.strip() for value in str(history).split(",") if value.strip()]
    if parser is parse_interval:
        return [parser(value, clamp_nonnegative=clamp_nonnegative) for value in values]
    return [parser(value) for value in values]


def build_reviews(row: pd.Series, *, include_current: bool = False):
    try:
        from fsrs_rs_python import FSRSReview
    except ImportError:
        raise ImportError(
            "fsrs-rs-python is not installed. Please install it to use the FSRS-rs backend."
        )

    t_history = parse_history(
        row["t_history"], parse_interval, clamp_nonnegative=True
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


def convert_to_items(df: pd.DataFrame, config: Config):
    """
    Convert a pandas DataFrame to a list of FSRSItem objects for fsrs-rs.

    Args:
        df: DataFrame with columns: card_id, review_th, t_history, r_history, delta_t, rating
        config: Configuration object

    Returns:
        list[FSRSItem]: List of FSRS items for training/evaluation
    """
    try:
        from fsrs_rs_python import FSRSItem, FSRSReview
    except ImportError:
        raise ImportError(
            "fsrs-rs-python is not installed. Please install it to use the FSRS-rs backend."
        )

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


class FSRSRsBackend:
    """Wrapper for FSRS-rs backend."""

    @staticmethod
    def default_parameters() -> List[float]:
        try:
            from fsrs_rs_python import DEFAULT_PARAMETERS
        except ImportError:
            raise ImportError(
                "fsrs-rs-python is not installed. Please install it to use the FSRS-rs backend."
            )

        return list(DEFAULT_PARAMETERS)

    def __init__(self, config: Config):
        """
        Initialize FSRS-rs backend.

        Args:
            config: Configuration object
        """
        try:
            from fsrs_rs_python import FSRS
        except ImportError:
            raise ImportError(
                "fsrs-rs-python is not installed. Please install it to use the FSRS-rs backend."
            )

        self.config = config
        self.backend = FSRS(parameters=[])

    def train(self, train_set: pd.DataFrame) -> List[float]:
        """
        Train FSRS-rs model on training data.

        Args:
            train_set: Training dataset

        Returns:
            List[float]: Trained FSRS parameters (weights)
        """
        train_set_items = convert_to_items(train_set, self.config)
        weights = list(
            map(lambda x: round(x, 4), self.backend.benchmark(train_set_items))
        )
        return weights

    def predict(
        self, testset: pd.DataFrame, weights: List[float]
    ) -> tuple[List[float], List[float], pd.DataFrame]:
        """
        Make predictions using FSRS-rs model.

        Args:
            testset: Test dataset
            weights: FSRS parameters

        Returns:
            tuple: (predictions, labels, testset_with_predictions)
        """
        from fsrs_rs_python import FSRS, FSRSItem

        def fsrs7_forgetting_curve(delta_t: pd.Series, stability: pd.Series) -> pd.Series:
            stability_array = stability.clip(lower=MIN_STABILITY).to_numpy(
                dtype=float, copy=False
            )
            delta_t_array = delta_t.clip(lower=0).to_numpy(dtype=float, copy=False)
            t_over_s = delta_t_array / stability_array

            decay1 = -weights[FSRS7_DECAY1_INDEX]
            decay2 = -weights[FSRS7_DECAY2_INDEX]
            base1 = weights[FSRS7_BASE1_INDEX]
            base2 = weights[FSRS7_BASE2_INDEX]

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
        history_items = [
            FSRSItem(reviews=build_reviews(row))
            for _, row in testset_copy.iterrows()
        ]
        memory_states = predictor.memory_state_batch(history_items)
        testset_copy["stability"] = [state.stability for state in memory_states]
        testset_copy["difficulty"] = [state.difficulty for state in memory_states]
        testset_copy["p"] = fsrs7_forgetting_curve(
            testset_copy["delta_t"], testset_copy["stability"]
        )

        p = testset_copy["p"].tolist()
        y = testset_copy["y"].tolist()

        return p, y, testset_copy
