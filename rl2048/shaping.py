from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


CORNER_TO_INDEX = {
    "TR": lambda n: (0, n - 1),
    "TL": lambda n: (0, 0),
    "BR": lambda n: (n - 1, n - 1),
    "BL": lambda n: (n - 1, 0),
}


def _normalize_corner(anchor_corner: str) -> str:
    corner = str(anchor_corner).upper()
    if corner not in CORNER_TO_INDEX:
        raise ValueError(f"Invalid anchor_corner={anchor_corner!r}. Use one of TR/TL/BR/BL.")
    return corner


def corner_index(size: int, anchor_corner: str) -> Tuple[int, int]:
    corner = _normalize_corner(anchor_corner)
    return CORNER_TO_INDEX[corner](int(size))


def board_to_log2_levels(board: np.ndarray) -> np.ndarray:
    encoded = np.asarray(board, dtype=np.int32).copy()
    mask = encoded > 0
    encoded[mask] = np.log2(encoded[mask]).astype(np.int32)
    return encoded


def max_tile_in_anchor_corner(board: np.ndarray, anchor_corner: str) -> int:
    b = np.asarray(board)
    r, c = corner_index(b.shape[0], anchor_corner)
    max_tile = int(b.max())
    return int(max_tile > 0 and int(b[r, c]) == max_tile)


def _anchor_row_fill_score(board: np.ndarray, anchor_corner: str) -> float:
    b = np.asarray(board)
    n = b.shape[0]
    corner = _normalize_corner(anchor_corner)
    row_idx, col_idx = corner_index(n, corner)

    row = b[row_idx]
    filled = float(np.sum(row > 0))

    if corner in ("TR", "BR"):
        distances = np.arange(n - 1, -1, -1, dtype=np.float32)
    else:
        distances = np.arange(0, n, dtype=np.float32)

    gap_penalty = 0.0
    for idx, value in enumerate(row):
        if value != 0:
            continue

        gap_penalty += float((n - distances[idx]) / max(n, 1))


    col = b[:, col_idx]
    col_support = float(np.sum(col > 0)) / max(n, 1)
    return filled - gap_penalty + 0.25 * col_support


def _monotonic_sequence_score(levels: Iterable[int]) -> float:
    values = [int(v) for v in levels if int(v) > 0]
    if len(values) <= 1:
        return 0.0

    penalty = 0.0
    for prev, nxt in zip(values, values[1:]):

        if nxt > prev:
            penalty += float(nxt - prev)
    return -penalty


def _monotone_snake_score(board: np.ndarray, anchor_corner: str) -> float:
    levels = board_to_log2_levels(board)
    n = levels.shape[0]
    corner = _normalize_corner(anchor_corner)
    row_idx, col_idx = corner_index(n, corner)

    row = levels[row_idx, :]
    col = levels[:, col_idx]

    if corner in ("TR", "BR"):
        row = row[::-1]
    if corner in ("BR", "BL"):
        col = col[::-1]

    return _monotonic_sequence_score(row) + _monotonic_sequence_score(col)


def _big_tile_spread(board: np.ndarray, k: int = 6) -> float:
    levels = board_to_log2_levels(board)
    coords = np.argwhere(levels > 0)
    if len(coords) <= 1:
        return 0.0

    items = [(int(levels[r, c]), int(r), int(c)) for r, c in coords]
    items.sort(reverse=True, key=lambda item: item[0])
    items = items[: min(k, len(items))]

    spread = 0.0
    for i in range(len(items)):
        vi, ri, ci = items[i]
        for j in range(i + 1, len(items)):
            vj, rj, cj = items[j]
            dist = abs(ri - rj) + abs(ci - cj)
            spread += (vi + vj) * dist
    return float(spread)


def _smoothness_penalty(board: np.ndarray) -> float:
    levels = board_to_log2_levels(board).astype(np.float32)
    penalty = 0.0
    n = levels.shape[0]

    for r in range(n):
        for c in range(n):
            center = levels[r, c]
            if center <= 0:
                continue
            if c + 1 < n and levels[r, c + 1] > 0:
                penalty += abs(center - levels[r, c + 1])
            if r + 1 < n and levels[r + 1, c] > 0:
                penalty += abs(center - levels[r + 1, c])
    return float(penalty)


def _trap_penalty(board: np.ndarray) -> float:
    levels = board_to_log2_levels(board)
    n = levels.shape[0]
    trapped = 0.0

    for r in range(n):
        for c in range(n):
            level = int(levels[r, c])
            if level <= 0 or level > 5:
                continue

            left = int(levels[r, c - 1]) if c - 1 >= 0 else 0
            right = int(levels[r, c + 1]) if c + 1 < n else 0
            up = int(levels[r - 1, c]) if r - 1 >= 0 else 0
            down = int(levels[r + 1, c]) if r + 1 < n else 0

            horiz_boxed = left >= level + 2 and right >= level + 2
            vert_boxed = up >= level + 2 and down >= level + 2

            large_neighbors = sum(
                neighbor >= level + 2
                for neighbor in (left, right, up, down)
                if neighbor > 0
            )

            if horiz_boxed or vert_boxed or large_neighbors >= 3:
                trapped += 1.0
    return trapped


def compute_shaping(
    board_before: np.ndarray,
    board_after: np.ndarray,
    anchor_corner: str = "TR",
) -> Dict[str, float]:
    before = np.asarray(board_before, dtype=np.int32)
    after = np.asarray(board_after, dtype=np.int32)
    _normalize_corner(anchor_corner)

    corner_bonus_delta = (
        max_tile_in_anchor_corner(after, anchor_corner)
        - max_tile_in_anchor_corner(before, anchor_corner)
    )
    anchor_row_fill_delta = (
        _anchor_row_fill_score(after, anchor_corner)
        - _anchor_row_fill_score(before, anchor_corner)
    )
    monotone_snake_delta = (
        _monotone_snake_score(after, anchor_corner)
        - _monotone_snake_score(before, anchor_corner)
    )
    big_tile_proximity_delta = _big_tile_spread(before) - _big_tile_spread(after)
    smoothness_delta = _smoothness_penalty(before) - _smoothness_penalty(after)
    empty_delta = float(np.sum(after == 0) - np.sum(before == 0))
    trap_penalty_delta = _trap_penalty(before) - _trap_penalty(after)

    return {
        "corner_bonus_delta": float(corner_bonus_delta),
        "anchor_row_fill_delta": float(anchor_row_fill_delta),
        "monotone_snake_delta": float(monotone_snake_delta),
        "big_tile_proximity_delta": float(big_tile_proximity_delta),
        "smoothness_delta": float(smoothness_delta),
        "empty_delta": float(empty_delta),
        "trap_penalty_delta": float(trap_penalty_delta),
    }
