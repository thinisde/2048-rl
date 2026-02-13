from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class VecStep:

    obs: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    info: List[Dict[str, Any]]


class VecRunner:

    def __init__(
        self,
        make_env: Callable[[int], Any],
        n_envs: int,
        *,
        base_seed: int = 0,
        auto_reset: bool = True,
        obs_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.n_envs = int(n_envs)
        self.auto_reset = bool(auto_reset)
        self.obs_transform = obs_transform
        self.envs = [make_env(base_seed + idx) for idx in range(self.n_envs)]

    def reset(self, *, seeds: Optional[np.ndarray] = None) -> np.ndarray:
        observations: List[np.ndarray] = []
        for env_idx, env in enumerate(self.envs):
            if seeds is None:
                obs = env.reset()
            else:
                obs = env.reset(seed=int(seeds[env_idx]))
            observations.append(self._transform_observation(obs))
        return np.stack(observations, axis=0)

    def step(self, actions: np.ndarray) -> VecStep:
        action_batch = np.asarray(actions)
        if action_batch.shape != (self.n_envs,):
            raise ValueError(
                f"actions must have shape ({self.n_envs},), got {action_batch.shape}"
            )

        observations: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []

        for env, action in zip(self.envs, action_batch):
            obs, reward, done, info = env.step(int(action))

            if done and self.auto_reset:
                info = dict(info)
                info["terminal_observation"] = obs
                obs = env.reset()

            observations.append(self._transform_observation(obs))
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)

        return VecStep(
            obs=np.stack(observations, axis=0),
            reward=np.asarray(rewards, dtype=np.float32),
            done=np.asarray(dones, dtype=np.bool_),
            info=infos,
        )

    def _transform_observation(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_transform is None:
            return obs
        return self.obs_transform(obs)
