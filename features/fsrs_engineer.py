from itertools import chain
from typing import Any, cast

import pandas as pd
import torch
from .base import BaseFeatureEngineer


class FSRSFeatureEngineer(BaseFeatureEngineer):
    """Builds the (time_history, rating) tensor features FSRS-style models consume."""

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tensor features for FSRS-style models
        These models use (time_history, rating_history) tensors
        """
        t_history_list, r_history_list = self.get_history_lists(df)

        # Create tensor features with shape (sequence_length, 2)
        # Each row contains [time_interval, rating] for that step
        cast(Any, df)["tensor"] = list(map(
            lambda pair: torch.tensor((pair[0][:-1], pair[1][:-1]), dtype=torch.float32).transpose(0, 1),
            chain.from_iterable(map(zip, t_history_list, r_history_list))
        ))

        return df
