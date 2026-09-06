from __future__ import annotations

import numpy as np
import pandas as pd


def accuracy_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute simple accuracy score."""
    if len(y_true) == 0:
        return 0.0
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return float((y_true_arr == y_pred_arr).mean())
