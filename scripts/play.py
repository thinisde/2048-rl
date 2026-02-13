from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl2048.agent import choose_greedy_legal_action, get_device, load_q_network
from rl2048.game import Game2048Env
from rl2048.transforms import board_to_log2

DEFAULT_MODEL_PATH = Path("models/qnet_2048_dqn.pt")


def print_board(board) -> None:
    print("-" * 25)
    for row in board:
        print(" ".join(f"{v:5d}" for v in row))
    print("-" * 25)


def play_game(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    delay: float = 0.2,
    seed: int = 123,
) -> None:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = get_device()
    print("Using device:", device)

    q_net, _ = load_q_network(model_path=model_path, device=device)

    env = Game2048Env(seed=seed)
    obs = env.reset()
    done = False
    info = {"score": 0}

    print("\nStarting Game\n")

    while not done:
        print_board(env.board)
        print("choosing action...", flush=True)

        obs_log = board_to_log2(obs)
        action, q_values = choose_greedy_legal_action(q_net, obs_log, env, device)
        rounded_q = [round(v, 3) for v in q_values]
        print("stepping env with action:", action, "| q-values:", rounded_q, flush=True)

        obs, reward, done, info = env.step(action)
        print(
            "step done | reward:",
            reward,
            "| done:",
            done,
            "| score:",
            info.get("score", 0),
            flush=True,
        )
        time.sleep(max(0.0, float(delay)))

    print_board(env.board)

    print("\nGame Over")
    print("Final Score:", info.get("score", 0))
    print("Max Tile:", int(env.board.max()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to model state_dict (.pt).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Seconds to wait between moves.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Environment seed.")
    args = parser.parse_args()

    play_game(model_path=args.model, delay=args.delay, seed=args.seed)


if __name__ == "__main__":
    main()
