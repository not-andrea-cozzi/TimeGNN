from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from ..core.base import BaseTrainer
from ..data.schema import EventSchema
from ..encoders.basic import BasicLabelEncoder
from ..metrics.basic import accuracy_score
from ..utils.logging import get_logger


@dataclass
class BasicTrainer(BaseTrainer):
    """Minimal trainer for baseline models."""
    logger_name: str = "timegnn.trainer"

    def train(
        self,
        model,
        train_df: pd.DataFrame,
        schema: EventSchema,
        encoder: BasicLabelEncoder,
    ) -> Dict[str, float]:
        """Train a model and return training metrics."""
        logger = get_logger(self.logger_name)
        logger.info("Training baseline model")
        model.fit(train_df, schema, encoder)
        return {}

    def evaluate(
        self,
        model,
        eval_df: pd.DataFrame,
        schema: EventSchema,
        encoder: BasicLabelEncoder,
    ) -> Dict[str, float]:
        """Evaluate a model and return metrics."""
        logger = get_logger(self.logger_name)
        logger.info("Evaluating model")
        preds = model.predict(eval_df, schema, encoder)
        labels = encoder.transform(eval_df[schema.event_col])
        return {"accuracy": accuracy_score(labels, preds)}
