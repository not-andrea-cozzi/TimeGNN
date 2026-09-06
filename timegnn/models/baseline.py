from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from ..core.base import BaseModel
from ..core.registry import model_registry
from ..data.schema import EventSchema
from ..encoders.basic import BasicLabelEncoder


@model_registry.register("baseline_most_frequent")
@dataclass
class BaselineMostFrequentModel(BaseModel):
    """Baseline model that predicts the most frequent event."""
    most_frequent_id: Optional[int] = None

    def fit(self, df: pd.DataFrame, schema: EventSchema, encoder: BasicLabelEncoder) -> None:
        """Fit by finding the most frequent event label."""
        encoded = encoder.transform(df[schema.event_col])
        self.most_frequent_id = int(encoded.value_counts().idxmax())

    def predict(self, df: pd.DataFrame, schema: EventSchema, encoder: BasicLabelEncoder) -> pd.Series:
        """Predict the most frequent event for each row."""
        if self.most_frequent_id is None:
            raise RuntimeError("Model is not fitted.")
        return pd.Series([self.most_frequent_id] * len(df), index=df.index)

    def save(self, path: str) -> None:
        """Save model state to disk."""
        payload = {"most_frequent_id": self.most_frequent_id}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BaselineMostFrequentModel":
        """Load model state from disk."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(most_frequent_id=payload.get("most_frequent_id"))
