from __future__ import annotations

import numpy as np


def board_to_log2(board: np.ndarray) -> np.ndarray:
    encoded = np.asarray(board).copy()
    mask = encoded > 0
    encoded[mask] = np.log2(encoded[mask]).astype(np.int32)
    return encoded
obs_log2 = board_to_log2
