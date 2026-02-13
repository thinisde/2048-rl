import torch
import torch.nn.functional as F


def dqn_update(q_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):

    s, a, r, s2, done = replay_buffer.sample(batch_size)

    s = s.to(device)
    a = a.to(device)
    r = r.to(device)
    s2 = s2.to(device)
    done = done.to(device)


    q_values = q_net(s)
    q_sa = q_values.gather(1, a.unsqueeze(1)).squeeze(1)


    with torch.no_grad():
        next_q = target_net(s2)
        max_next_q = next_q.max(dim=1)[0]
        target = r + gamma * (1 - done) * max_next_q


    loss = F.mse_loss(q_sa, target)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
