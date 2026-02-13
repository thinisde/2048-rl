from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .game import Game2048Env
from .shaping import compute_shaping, max_tile_in_anchor_corner
from .transforms import board_to_log2


def _normalize_corner(anchor_corner: str) -> str:
    corner = str(anchor_corner).upper()
    if corner not in {"TR", "TL", "BR", "BL"}:
        raise ValueError(f"Invalid anchor_corner={anchor_corner!r}. Use TR/TL/BR/BL.")
    return corner


def _move_preferences(anchor_corner: str) -> Tuple[Sequence[int], Sequence[int]]:
    corner = _normalize_corner(anchor_corner)

    if corner == "TR":
        return (3, 0), (2, 1)
    if corner == "TL":
        return (2, 0), (3, 1)
    if corner == "BR":
        return (3, 1), (2, 0)
    return (2, 1), (3, 0)


def _clone_env(env: Game2048Env) -> Game2048Env:
    if hasattr(env, "clone"):
        return env.clone(seed=0)
    clone = Game2048Env(size=env.size, seed=0)
    clone.board = env.board.copy()
    clone.score = int(env.score)
    return clone


def _simulate_without_spawn(env: Game2048Env, action: int) -> Tuple[bool, np.ndarray, int]:
    probe = _clone_env(env)
    moved, merge_reward = probe._move(int(action))
    return bool(moved), probe.board.copy(), int(merge_reward)


def _can_return_corner_next_move(
    board_after: np.ndarray,
    env: Game2048Env,
    primary_actions: Sequence[int],
    anchor_corner: str,
) -> bool:
    probe = _clone_env(env)
    probe.board = board_after.copy()
    for action in primary_actions:
        moved, candidate_board, _merge = _simulate_without_spawn(probe, int(action))
        if not moved:
            continue
        if max_tile_in_anchor_corner(candidate_board, anchor_corner):
            return True
    return False


def _teacher_action_score(
    action: int,
    board_before: np.ndarray,
    board_after: np.ndarray,
    merge_reward: int,
    env: Game2048Env,
    primary_actions: Sequence[int],
    secondary_actions: Sequence[int],
    anchor_corner: str,
) -> float:
    shaping = compute_shaping(
        board_before=board_before,
        board_after=board_after,
        anchor_corner=anchor_corner,
    )
    score = 0.0

    if action in primary_actions:
        score += 1.25
    elif action in secondary_actions:
        score -= 0.35

    score += 2.4 * shaping["corner_bonus_delta"]
    score += 0.8 * shaping["anchor_row_fill_delta"]
    score += 0.5 * shaping["monotone_snake_delta"]
    score += 0.03 * shaping["big_tile_proximity_delta"]
    score += 0.2 * shaping["smoothness_delta"]
    score += 0.35 * shaping["empty_delta"]
    score += 0.4 * shaping["trap_penalty_delta"]
    score += 0.04 * float(merge_reward)

    corner_before = bool(max_tile_in_anchor_corner(board_before, anchor_corner))
    corner_after = bool(max_tile_in_anchor_corner(board_after, anchor_corner))
    if corner_before and not corner_after:
        score -= 2.25
        if _can_return_corner_next_move(
            board_after=board_after,
            env=env,
            primary_actions=primary_actions,
            anchor_corner=anchor_corner,
        ):
            score += 1.4

    return float(score)


def teacher_action(env: Game2048Env, anchor_corner: str = "TR") -> int:
    corner = _normalize_corner(anchor_corner)
    primary_actions, secondary_actions = _move_preferences(corner)
    candidate_order: List[int] = list(primary_actions) + list(secondary_actions)

    board_before = env.board.copy()
    legal_scored: List[Tuple[int, float]] = []
    for order_idx, action in enumerate(candidate_order):
        moved, board_after, merge_reward = _simulate_without_spawn(env, int(action))
        if not moved:
            continue
        score = _teacher_action_score(
            action=action,
            board_before=board_before,
            board_after=board_after,
            merge_reward=merge_reward,
            env=env,
            primary_actions=primary_actions,
            secondary_actions=secondary_actions,
            anchor_corner=corner,
        )

        score -= order_idx * 1e-3
        legal_scored.append((int(action), float(score)))

    if not legal_scored:

        return int(primary_actions[0])

    legal_scored.sort(key=lambda item: item[1], reverse=True)
    return int(legal_scored[0][0])


def collect_demonstrations(
    n_episodes: int,
    out_path: str | Path,
    *,
    anchor_corner: str = "TR",
    seed: int = 2026,
    max_steps_per_episode: int = 2048,
) -> Path:
    corner = _normalize_corner(anchor_corner)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    obs_log_list = []
    action_list = []
    reward_list = []
    next_obs_log_list = []
    done_list = []

    for ep in range(int(n_episodes)):
        env = Game2048Env(seed=seed + ep)
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < int(max_steps_per_episode):
            action = int(teacher_action(env, anchor_corner=corner))
            next_obs, reward, done, _info = env.step(action)

            obs_log_list.append(board_to_log2(obs))
            action_list.append(action)
            reward_list.append(float(reward))
            next_obs_log_list.append(board_to_log2(next_obs))
            done_list.append(bool(done))

            obs = next_obs
            steps += 1

    if obs_log_list:
        obs_array = np.stack(obs_log_list, axis=0).astype(np.int16)
        next_obs_array = np.stack(next_obs_log_list, axis=0).astype(np.int16)
        action_array = np.asarray(action_list, dtype=np.int64)
        reward_array = np.asarray(reward_list, dtype=np.float32)
        done_array = np.asarray(done_list, dtype=np.bool_)
    else:
        obs_array = np.zeros((0, 4, 4), dtype=np.int16)
        next_obs_array = np.zeros((0, 4, 4), dtype=np.int16)
        action_array = np.zeros((0,), dtype=np.int64)
        reward_array = np.zeros((0,), dtype=np.float32)
        done_array = np.zeros((0,), dtype=np.bool_)

    np.savez_compressed(
        output,
        obs=obs_array,
        action=action_array,
        reward=reward_array,
        next_obs=next_obs_array,
        done=done_array,
        anchor_corner=np.array(corner),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--out", type=str, default="models/teacher_demos.npz")
    parser.add_argument(
        "--anchor_corner",
        type=str,
        default="TR",
        choices=["TR", "TL", "BR", "BL"],
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_steps", type=int, default=2048)
    args = parser.parse_args()

    out = collect_demonstrations(
        n_episodes=args.episodes,
        out_path=args.out,
        anchor_corner=args.anchor_corner,
        seed=args.seed,
        max_steps_per_episode=args.max_steps,
    )
    print(f"Saved demonstrations to {out}")


if __name__ == "__main__":
    main()
