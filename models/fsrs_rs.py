"""
FSRS-rs model wrapper for integration with the other.py architecture.

This module provides utilities to work with the fsrs-rs (Rust-based FSRS implementation)
within the benchmark framework.
"""

from typing import List, Optional
import pandas as pd
from config import Config


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

    def parse_int(value: object, *, clamp_nonnegative: bool = False) -> int:
        if pd.isna(value):
            raise ValueError("Expected a numeric review history value, got missing data")
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("Expected a numeric review history value, got empty text")
        parsed = int(float(value))
        return max(0, parsed) if clamp_nonnegative else parsed

    def parse_history(history: object, *, clamp_nonnegative: bool = False) -> List[int]:
        if pd.isna(history):
            return []
        return [
            parse_int(value.strip(), clamp_nonnegative=clamp_nonnegative)
            for value in str(history).split(",")
            if value.strip()
        ]

    def accumulate(group):
        items = []
        for _, row in group.iterrows():
            t_history = parse_history(row["t_history"], clamp_nonnegative=True) + [
                parse_int(row["delta_t"], clamp_nonnegative=True)
            ]
            r_history = parse_history(row["r_history"]) + [parse_int(row["rating"])]
            items.append(
                (
                    row["review_th"],
                    FSRSItem(
                        reviews=[
                            FSRSReview(delta_t=x[0], rating=x[1])
                            for x in zip(t_history, r_history)
                        ]
                    ),
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
        from fsrs_optimizer import Collection, power_forgetting_curve  # type: ignore

        my_collection = Collection(weights)
        testset_copy = testset.copy()

        testset_copy["stability"], testset_copy["difficulty"] = (
            my_collection.batch_predict(testset_copy)
        )
        testset_copy["p"] = power_forgetting_curve(
            testset_copy["delta_t"],
            testset_copy["stability"],
            -weights[20],
        )

        p = testset_copy["p"].tolist()
        y = testset_copy["y"].tolist()

        return p, y, testset_copy
