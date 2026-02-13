from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, max_level: int = 16, emb_dim: int = 32, n_actions: int = 4):
        super().__init__()
        self.max_level = int(max_level)
        self.n_actions = int(n_actions)


        self.tile_embedding = nn.Embedding(self.max_level + 1, emb_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(emb_dim, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dtype != torch.long:
            inputs = inputs.long()

        safe_inputs = torch.clamp(inputs, 0, self.max_level)
        embedded = self.tile_embedding(safe_inputs)
        embedded = embedded.permute(0, 3, 1, 2).contiguous()
        encoded = self.encoder(embedded)
        return self.q_head(encoded)
QNet = QNetwork
