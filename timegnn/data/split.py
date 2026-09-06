from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_case(df: pd.DataFrame, case_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/test using unique case IDs."""
    cases = df[case_col].drop_duplicates()
    train_cases, test_cases = train_test_split(
        cases, test_size=test_size, random_state=random_state, shuffle=True
    )
    train_df = df[df[case_col].isin(train_cases)].copy()
    test_df = df[df[case_col].isin(test_cases)].copy()
    return train_df, test_df
