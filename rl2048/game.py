from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .shaping import compute_shaping


Action = int

DEFAULT_SHAPING_WEIGHTS: Dict[str, float] = {
    "merge_reward": 1.0,
    "corner_bonus": 2.0,
    "anchor_row_fill": 0.5,
    "monotone_snake": 0.2,
    "big_tile_proximity": 0.02,
    "smoothness": 0.05,
    "empty": 0.3,
    "trap": 0.3,
    "illegal_penalty": 3.0,
}


class Game2048Env:

    def __init__(
        self,
        size: int = 4,
        seed: Optional[int] = None,
        *,
        use_shaping: bool = False,
        shaping_weights: Optional[Dict[str, float]] = None,
        anchor_corner: str = "TR",
    ):
        self.size = int(size)
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.use_shaping = bool(use_shaping)
        self.anchor_corner = str(anchor_corner).upper()
        if self.anchor_corner not in {"TR", "TL", "BR", "BL"}:
            raise ValueError(
                f"Invalid anchor_corner={anchor_corner!r}. Use one of TR/TL/BR/BL."
            )
        self.shaping_weights = dict(DEFAULT_SHAPING_WEIGHTS)
        if shaping_weights:
            self.shaping_weights.update(shaping_weights)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.board.fill(0)
        self.score = 0
        self._spawn_tile()
        self._spawn_tile()
        return self._get_observation()

    def clone(self, seed: Optional[int] = 0) -> "Game2048Env":
        cloned = Game2048Env(
            size=self.size,
            seed=seed,
            use_shaping=self.use_shaping,
            shaping_weights=self.shaping_weights,
            anchor_corner=self.anchor_corner,
        )
        cloned.board = self.board.copy()
        cloned.score = int(self.score)
        return cloned

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        before = self.board.copy()
        moved, merge_reward = self._move(action)

        if moved:
            self._spawn_tile()

        after = self.board.copy()
        shaping = compute_shaping(
            board_before=before,
            board_after=after,
            anchor_corner=self.anchor_corner,
        )

        self.score += int(merge_reward)
        done = not self._has_moves()
        reward = self.shaping_weights["merge_reward"] * float(merge_reward)

        if self.use_shaping:
            reward += self.shaping_weights["corner_bonus"] * shaping["corner_bonus_delta"]
            reward += self.shaping_weights["anchor_row_fill"] * shaping["anchor_row_fill_delta"]
            reward += self.shaping_weights["monotone_snake"] * shaping["monotone_snake_delta"]
            reward += self.shaping_weights["big_tile_proximity"] * shaping["big_tile_proximity_delta"]
            reward += self.shaping_weights["smoothness"] * shaping["smoothness_delta"]
            reward += self.shaping_weights["empty"] * shaping["empty_delta"]
            reward += self.shaping_weights["trap"] * shaping["trap_penalty_delta"]

        if not moved:
            reward -= float(self.shaping_weights["illegal_penalty"])

        info = {
            "moved": moved,
            "score": self.score,
            "merge_reward": float(merge_reward),
            "use_shaping": self.use_shaping,
            "anchor_corner": self.anchor_corner,
            "corner_bonus_delta": float(shaping["corner_bonus_delta"]),
            "anchor_row_fill_delta": float(shaping["anchor_row_fill_delta"]),
            "monotone_snake_delta": float(shaping["monotone_snake_delta"]),
            "big_tile_proximity_delta": float(shaping["big_tile_proximity_delta"]),
            "smoothness_delta": float(shaping["smoothness_delta"]),
            "empty_delta": float(shaping["empty_delta"]),
            "trap_penalty_delta": float(shaping["trap_penalty_delta"]),
        }
        return self._get_observation(), float(reward), done, info

    def _get_observation(self) -> np.ndarray:
        return self.board.copy()

    def _spawn_tile(self) -> None:
        empty_positions = np.argwhere(self.board == 0)
        if empty_positions.size == 0:
            return

        random_index = self.rng.integers(0, len(empty_positions))
        row, col = empty_positions[random_index]
        self.board[row, col] = 2 if self.rng.random() < 0.9 else 4

    def _has_moves(self) -> bool:
        if np.any(self.board == 0):
            return True

        board = self.board
        if np.any(board[:, :-1] == board[:, 1:]):
            return True
        if np.any(board[:-1, :] == board[1:, :]):
            return True
        return False

    def _move(self, action: Action) -> Tuple[bool, int]:
        working_board = self._orient_board_for_action(self.board.copy(), action)

        moved = False
        total_reward = 0
        for row_index in range(self.size):
            new_row, row_moved, row_reward = self._slide_merge_left(
                working_board[row_index, :]
            )
            working_board[row_index, :] = new_row
            moved = moved or row_moved
            total_reward += row_reward

        new_board = self._undo_action_orientation(working_board, action)
        moved = moved or not np.array_equal(self.board, new_board)
        self.board = new_board
        return moved, total_reward

    @staticmethod
    def _orient_board_for_action(board: np.ndarray, action: Action) -> np.ndarray:
        if action == 2:
            return board
        if action == 3:
            return np.fliplr(board)
        if action == 0:
            return np.rot90(board, 1)
        if action == 1:
            return np.rot90(board, -1)
        raise ValueError("Invalid action")

    @staticmethod
    def _undo_action_orientation(board: np.ndarray, action: Action) -> np.ndarray:
        if action == 2:
            return board
        if action == 3:
            return np.fliplr(board)
        if action == 0:
            return np.rot90(board, -1)
        if action == 1:
            return np.rot90(board, 1)
        raise ValueError("Invalid action")

    @staticmethod
    def _slide_merge_left(row: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        size = row.shape[0]
        non_zero_tiles = row[row != 0]

        merged_row = []
        reward = 0
        index = 0
        while index < len(non_zero_tiles):
            if (
                index + 1 < len(non_zero_tiles)
                and non_zero_tiles[index] == non_zero_tiles[index + 1]
            ):
                merged_value = int(non_zero_tiles[index] * 2)
                merged_row.append(merged_value)
                reward += merged_value
                index += 2
            else:
                merged_row.append(int(non_zero_tiles[index]))
                index += 1

        new_row = np.array(merged_row + [0] * (size - len(merged_row)), dtype=row.dtype)
        return new_row, not np.array_equal(new_row, row), reward
