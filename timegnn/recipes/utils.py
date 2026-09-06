from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Type, TypeVar

import pandas as pd

T = TypeVar("T")


def build_sequence_table(event: pd.DataFrame, case_index: str, seq_cols: List[str]) -> pd.DataFrame:
    """Build a per-case sequence table for sequence-level features."""
    return event[[case_index] + seq_cols].groupby(case_index).first().reset_index()


def resolve_config(config: T | None, config_cls: Type[T], overrides: Dict[str, Any]) -> T:
    """Return a config instance with optional field overrides.

    Args:
        config: Existing config or None to create defaults.
        config_cls: Dataclass type to instantiate when config is None.
        overrides: Keyword overrides mapped to dataclass fields.

    Returns:
        Resolved config instance.
    """
    if config is None:
        config = config_cls()  # type: ignore[call-arg]

    if not overrides:
        return config

    unknown = [key for key in overrides if not hasattr(config, key)]
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    return replace(config, **overrides)
