from __future__ import annotations

from typing import Optional

import pandas as pd

from .schema import EventSchema


def read_events_csv(path: str, schema: EventSchema) -> pd.DataFrame:
    """Read an events CSV and validate against a schema."""
    df = pd.read_csv(path)
    schema.validate(df)
    return df
