import pandas as pd
from .fsrs_engineer import FSRSFeatureEngineer
from config import Config


def create_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Build FSRS feature columns from raw review logs.

    Args:
        df: Input dataframe with review logs
        config: Configuration object

    Returns:
        Processed dataframe with model-specific (tensor) features
    """
    return FSRSFeatureEngineer(config).create_features(df)
