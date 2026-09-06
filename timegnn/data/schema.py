from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import pandas as pd


@dataclass
class EventSchema:
    """Schema describing event log columns."""
    time_col: str
    case_col: str
    event_col: str
    feature_cols: List[str] = field(default_factory=list)

    def required_columns(self) -> List[str]:
        """Return all required columns for this schema."""
        return [self.time_col, self.case_col, self.event_col] + list(self.feature_cols)

    def validate(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist in the dataframe."""
        missing = [col for col in self.required_columns() if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
