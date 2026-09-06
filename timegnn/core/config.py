from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Config:
    """Base configuration for pipeline usage."""
    model: str
    task: str
    data_source: str
    time_col: str
    case_col: str
    event_col: str
    device: str = "auto"
    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-3
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return config as a serializable dictionary."""
        return {
            "model": self.model,
            "task": self.task,
            "data_source": self.data_source,
            "time_col": self.time_col,
            "case_col": self.case_col,
            "event_col": self.event_col,
            "device": self.device,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "extras": dict(self.extras),
        }
