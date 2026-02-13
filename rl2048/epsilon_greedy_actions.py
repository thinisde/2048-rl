from __future__ import annotations

import torch


def select_epsilon_greedy_actions(
    q_network,
    observation_batch,
    epsilon: float,
    device: torch.device | None = None,
) -> torch.LongTensor:
    if device is None:
        device = next(q_network.parameters()).device

    if not isinstance(observation_batch, torch.Tensor):
        observation_batch = torch.tensor(
            observation_batch,
            dtype=torch.long,
            device=device,
        )
    else:
        observation_batch = observation_batch.to(device=device)
        if observation_batch.dtype != torch.long:
            observation_batch = observation_batch.long()

    batch_size = observation_batch.shape[0]

    q_network.eval()
    with torch.no_grad():
        q_values = q_network(observation_batch)
        greedy_actions = torch.argmax(q_values, dim=1)

    random_actions = torch.randint(low=0, high=4, size=(batch_size,), device=device)
    explore_mask = torch.rand(batch_size, device=device) < float(epsilon)
    return torch.where(explore_mask, random_actions, greedy_actions).long()
epsilon_greedy_actions = select_epsilon_greedy_actions
