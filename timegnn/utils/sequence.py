from __future__ import annotations

from typing import Iterable, List

import numpy as np


def pad_sequences(
    sequences: Iterable[np.ndarray],
    padding: str = "post",
    value: float = -1,
    dtype: str = "float32",
) -> np.ndarray:
    """Pad a list of variable-length arrays to a common length."""
    seq_list: List[np.ndarray] = [np.asarray(s) for s in sequences]
    if not seq_list:
        return np.array([], dtype=dtype)

    max_len = max(len(s) for s in seq_list)
    if max_len == 0:
        return np.array([[] for _ in seq_list], dtype=dtype)

    sample_shape = seq_list[0].shape[1:]
    output = np.full((len(seq_list), max_len) + sample_shape, value, dtype=dtype)

    for i, seq in enumerate(seq_list):
        if not len(seq):
            continue
        if padding == "post":
            trunc = seq[:max_len]
            output[i, : len(trunc)] = trunc
        else:
            trunc = seq[-max_len:]
            output[i, -len(trunc) :] = trunc

    return output
