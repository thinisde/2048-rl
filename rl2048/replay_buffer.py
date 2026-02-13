from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, capacity: int, board_shape=(4, 4), device: str = "cpu"):
        self.capacity = int(capacity)
        self.board_shape = tuple(board_shape)
        self.device = torch.device(device)

        self.states = np.zeros((self.capacity, *self.board_shape), dtype=np.int16)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self.board_shape), dtype=np.int16)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)

        self.write_index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(self, states, actions, rewards, next_states, dones) -> None:
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        dones = np.asarray(dones)

        batch_size = int(states.shape[0])
        if states.shape[1:] != self.board_shape:
            raise ValueError(
                f"Expected state shape (*, {self.board_shape}), got {states.shape}"
            )
        if next_states.shape[1:] != self.board_shape:
            raise ValueError(
                f"Expected next_state shape (*, {self.board_shape}), got {next_states.shape}"
            )
        if not (
            actions.shape[0] == rewards.shape[0] == next_states.shape[0] == dones.shape[0] == batch_size
        ):
            raise ValueError("Batch arrays must all have the same first dimension")

        indices = (np.arange(batch_size) + self.write_index) % self.capacity
        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.dones[indices] = dones.astype(np.bool_)

        self.write_index = (self.write_index + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("ReplayBuffer is empty")

        sample_indices = np.random.randint(0, self.size, size=int(batch_size))

        states = torch.tensor(
            self.states[sample_indices],
            dtype=torch.long,
            device=self.device,
        )
        actions = torch.tensor(
            self.actions[sample_indices],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            self.rewards[sample_indices],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            self.next_states[sample_indices],
            dtype=torch.long,
            device=self.device,
        )
        dones = torch.tensor(
            self.dones[sample_indices].astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        return states, actions, rewards, next_states, dones
