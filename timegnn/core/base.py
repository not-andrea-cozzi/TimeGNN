from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(ABC):
    """Abstract base class for models."""
    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        raise NotImplementedError


class BaseDataset(ABC):
    """Abstract base class for datasets."""
    @abstractmethod
    def to_dataloader(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class BaseEncoder(ABC):
    """Abstract base class for feature encoders."""
    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class BaseTrainer(ABC):
    """Abstract base class for training loops."""
    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        raise NotImplementedError


class BaseEvaluator(ABC):
    """Abstract base class for evaluation logic."""
    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        raise NotImplementedError
