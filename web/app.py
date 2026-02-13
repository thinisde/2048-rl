from __future__ import annotations

import argparse
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request

from rl2048.agent import (
    ACTION_NAMES,
    choose_greedy_legal_action,
    get_device,
    load_q_network,
    predict_q_values,
)
from rl2048.game import Game2048Env
from rl2048.transforms import board_to_log2


def _sanitize_info(info: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            clean[key] = value
        elif hasattr(value, "tolist"):
            clean[key] = value.tolist()
        else:
            clean[key] = str(value)
    return clean


class PlayController:
    def __init__(self, model_path: Path, seed: int = 123):
        self.lock = Lock()
        self.device = get_device()
        self.q_net, _ = load_q_network(model_path=model_path, device=self.device)
        self.env = Game2048Env(seed=seed)
        self.done = False
        self.last_action: int | None = None
        self.last_reward: float = 0.0
        self.last_info: Dict[str, Any] = {}
        self.last_decision_q_values: list[float] = []
        self.reset(seed=seed)

    def _current_q_values(self) -> list[float]:
        obs_log = board_to_log2(self.env.board)
        q_values = predict_q_values(self.q_net, obs_log, self.device)
        return [float(v) for v in q_values.tolist()]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "board": self.env.board.tolist(),
            "score": int(self.env.score),
            "max_tile": int(self.env.board.max()),
            "done": bool(self.done),
            "last_action": self.last_action,
            "last_action_name": (
                ACTION_NAMES.get(self.last_action)
                if self.last_action is not None
                else None
            ),
            "last_reward": float(self.last_reward),
            "last_info": self.last_info,
            "decision_q_values": self.last_decision_q_values,
            "q_values": self._current_q_values(),
        }

    def reset(self, seed: int | None = None) -> Dict[str, Any]:
        with self.lock:
            self.env.reset(seed=seed)
            self.done = False
            self.last_action = None
            self.last_reward = 0.0
            self.last_info = {}
            self.last_decision_q_values = []
            return self.snapshot()

    def step_model(self) -> Dict[str, Any]:
        with self.lock:
            if self.done:
                return self.snapshot()

            obs_log = board_to_log2(self.env.board)
            action, q_values = choose_greedy_legal_action(
                q_net=self.q_net,
                obs_log=obs_log,
                env=self.env,
                device=self.device,
            )

            _obs, reward, done, info = self.env.step(action)
            self.done = bool(done)
            self.last_action = int(action)
            self.last_reward = float(reward)
            self.last_info = _sanitize_info(info)
            self.last_decision_q_values = [float(v) for v in q_values.tolist()]
            return self.snapshot()

    def step_manual(self, action: int) -> Dict[str, Any]:
        with self.lock:
            if self.done:
                return self.snapshot()

            _obs, reward, done, info = self.env.step(int(action))
            self.done = bool(done)
            self.last_action = int(action)
            self.last_reward = float(reward)
            self.last_info = _sanitize_info(info)
            self.last_decision_q_values = self._current_q_values()
            return self.snapshot()


app = Flask(__name__, template_folder="templates", static_folder="static")
controller: PlayController | None = None


def _get_controller() -> PlayController:
    if controller is None:
        raise RuntimeError("Controller has not been initialized.")
    return controller


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    return jsonify(_get_controller().snapshot())


@app.post("/api/reset")
def api_reset():
    payload = request.get_json(silent=True) or {}
    seed_value = payload.get("seed")
    if seed_value is None:
        seed = None
    else:
        try:
            seed = int(seed_value)
        except (TypeError, ValueError):
            return jsonify({"error": "seed must be an integer"}), 400
    return jsonify(_get_controller().reset(seed=seed))


@app.post("/api/step/model")
def api_step_model():
    return jsonify(_get_controller().step_model())


@app.post("/api/step/manual")
def api_step_manual():
    payload = request.get_json(silent=True) or {}
    if "action" not in payload:
        return jsonify({"error": "Missing 'action' field"}), 400
    try:
        action = int(payload["action"])
    except (TypeError, ValueError):
        return jsonify({"error": "action must be an integer in [0, 3]"}), 400
    if action not in (0, 1, 2, 3):
        return jsonify({"error": "action must be one of 0, 1, 2, 3"}), 400
    return jsonify(_get_controller().step_manual(action))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/qnet_2048_dqn.pt",
        help="Path to model state_dict (.pt).",
    )
    parser.add_argument("--seed", type=int, default=123, help="Environment seed.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run training first or pass --model."
        )

    global controller
    controller = PlayController(model_path=model_path, seed=args.seed)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
