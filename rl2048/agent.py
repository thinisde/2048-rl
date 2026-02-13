from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .game import Game2048Env
from .qnet import QNetwork

ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_q_network(
    model_path: str | Path,
    device: torch.device | None = None,
    *,
    max_level: int = 16,
    emb_dim: int = 32,
    n_actions: int = 4,
) -> Tuple[QNetwork, torch.device]:
    if device is None:
        device = get_device()

    model_path = Path(model_path)
    q_net = QNetwork(max_level=max_level, emb_dim=emb_dim, n_actions=n_actions).to(
        device
    )
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()
    return q_net, device


def predict_q_values(
    q_net: QNetwork,
    obs_log: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs_log, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(obs_tensor).squeeze(0)
    return q_values.detach().cpu().numpy()


def choose_greedy_legal_action(
    q_net: QNetwork,
    obs_log: np.ndarray,
    env: Game2048Env,
    device: torch.device,
) -> Tuple[int, np.ndarray]:
    q_values = predict_q_values(q_net=q_net, obs_log=obs_log, device=device)
    ranked_actions = np.argsort(q_values)[::-1].tolist()

    for action in ranked_actions:
        if hasattr(env, "clone"):
            test_env = env.clone(seed=0)
        else:
            test_env = Game2048Env(size=env.size, seed=0)
            test_env.board = env.board.copy()
            test_env.score = int(env.score)
        moved, _merge_reward = test_env._move(int(action))
        if moved:
            return int(action), q_values

    return int(np.argmax(q_values)), q_values
