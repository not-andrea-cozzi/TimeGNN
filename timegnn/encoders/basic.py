from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class BasicLabelEncoder:
    """Minimal label encoder for string categorical values."""
    label_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_label: Dict[int, str] = field(default_factory=dict)

    def fit(self, series: pd.Series) -> None:
        """Fit encoder on a pandas Series."""
        labels = pd.Series(series).astype(str).unique().tolist()
        self.label_to_id = {label: idx for idx, label in enumerate(labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

    def transform(self, series: pd.Series) -> pd.Series:
        """Transform values to integer ids."""
        if not self.label_to_id:
            raise RuntimeError("Encoder is not fitted.")
        return pd.Series(series).astype(str).map(self.label_to_id).fillna(-1).astype(int)

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        """Transform integer ids back to original labels."""
        if not self.id_to_label:
            raise RuntimeError("Encoder is not fitted.")
        return pd.Series(series).map(self.id_to_label)
