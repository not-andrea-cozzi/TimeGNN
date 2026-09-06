from __future__ import annotations

from typing import Any, Optional

from .config import Config
from .registry import model_registry
from ..utils.logging import get_logger
from ..data import EventSchema, read_events_csv, split_by_case
from ..encoders import BasicLabelEncoder
from ..train import BasicTrainer
from ..models import baseline  # noqa: F401


class Pipeline:
    """High-level pipeline wrapper for training and evaluation."""
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = get_logger("timegnn.pipeline")
        self.model = None
        self.schema = EventSchema(
            time_col=config.time_col,
            case_col=config.case_col,
            event_col=config.event_col,
            feature_cols=config.extras.get("feature_cols", []),
        )
        self.encoder = BasicLabelEncoder()
        self.trainer = BasicTrainer()

    def build(self) -> None:
        """Instantiate the configured model."""
        self.logger.info("Building model '%s'", self.config.model)
        model_cls = model_registry.get(self.config.model)
        self.model = model_cls()

    def fit(self) -> None:
        """Train the model using the configured dataset."""
        if self.model is None:
            self.build()
        self.logger.info("Starting training")
        df = read_events_csv(self.config.data_source, self.schema)
        train_df, _ = split_by_case(
            df,
            case_col=self.schema.case_col,
            test_size=self.config.extras.get("test_size", 0.2),
            random_state=self.config.seed,
        )
        self.encoder.fit(train_df[self.schema.event_col])
        self.trainer.train(self.model, train_df, self.schema, self.encoder)

    def evaluate(self) -> Any:
        """Evaluate the trained model on the test split."""
        if self.model is None:
            raise RuntimeError("Model not built. Call fit() or build() first.")
        self.logger.info("Evaluating model")
        df = read_events_csv(self.config.data_source, self.schema)
        _, test_df = split_by_case(
            df,
            case_col=self.schema.case_col,
            test_size=self.config.extras.get("test_size", 0.2),
            random_state=self.config.seed,
        )
        return self.trainer.evaluate(self.model, test_df, self.schema, self.encoder)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run model prediction with user-supplied inputs."""
        if self.model is None:
            raise RuntimeError("Model not built. Call fit() or build() first.")
        return self.model.predict(*args, **kwargs)

    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise RuntimeError("Model not built. Call fit() or build() first.")
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load a trained model from disk."""
        model_cls = model_registry.get(self.config.model)
        self.model = model_cls.load(path)
