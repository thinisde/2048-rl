
from __future__ import annotations

from rl2048.agent import choose_greedy_legal_action
from rl2048.epsilon_greedy_actions import select_epsilon_greedy_actions
from rl2048.game import DEFAULT_SHAPING_WEIGHTS, Game2048Env
from rl2048.qnet import QNetwork
from rl2048.replay_buffer import ReplayBuffer
from rl2048.shaping import max_tile_in_anchor_corner
from rl2048.strategy_teacher import teacher_action
from rl2048.transforms import board_to_log2
from rl2048.vec_runner import VecRunner

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TrainConfig:

    n_envs: int = 32
    base_seed: int = 42


    total_env_steps: int = 10_000_000


    replay_capacity: int = 500_000
    min_replay_size: int = 20_000
    batch_size: int = 128


    gamma: float = 0.99
    lr: float = 1e-4
    gradient_clip_norm: float = 10.0


    train_every_env_steps: int = 64
    target_update_every: int = 8_192
    eval_every: int = 1_000_000


    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 3_000_000


    max_level: int = 16
    emb_dim: int = 32
    n_actions: int = 4


    log_every_iters: int = 200


    eval_episodes: int = 3
    eval_max_steps: int = 1024


    use_shaping: bool = False
    anchor_corner: str = "TR"
    shaping_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SHAPING_WEIGHTS)
    )

    warmup_env_steps: int = 100_000
    teacher_mix_start: float = 0.30
    teacher_mix_end: float = 0.10
    teacher_mix_decay_steps: int = 1_000_000
    demo_path: Optional[str] = None


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_env(seed: int, cfg: TrainConfig) -> Game2048Env:
    return Game2048Env(
        seed=seed,
        use_shaping=cfg.use_shaping,
        shaping_weights=cfg.shaping_weights,
        anchor_corner=cfg.anchor_corner,
    )


def epsilon_by_step(step: int, cfg: TrainConfig) -> float:
    if cfg.eps_decay_steps <= 0:
        return float(cfg.eps_end)
    t = min(max(step, 0), cfg.eps_decay_steps)
    frac = t / cfg.eps_decay_steps
    return float(cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start))


def teacher_mix_by_step(step: int, cfg: TrainConfig) -> float:
    if cfg.teacher_mix_decay_steps <= 0:
        return float(cfg.teacher_mix_end)
    t = min(max(step, 0), cfg.teacher_mix_decay_steps)
    frac = t / cfg.teacher_mix_decay_steps
    return float(
        cfg.teacher_mix_start + frac * (cfg.teacher_mix_end - cfg.teacher_mix_start)
    )


def dqn_update_double(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    cfg: TrainConfig,
    device: torch.device,
) -> float:
    s, a, r, s2, done = buffer.sample(cfg.batch_size)

    s = s.to(device=device, dtype=torch.long)
    a = a.to(device=device, dtype=torch.long)
    r = r.to(device=device, dtype=torch.float32)
    s2 = s2.to(device=device, dtype=torch.long)
    done = done.to(device=device, dtype=torch.float32)

    q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = q_net(s2).argmax(dim=1, keepdim=True)
        next_q = target_net(s2).gather(1, next_actions).squeeze(1)
        target = r + cfg.gamma * (1.0 - done) * next_q

    loss = F.smooth_l1_loss(q_sa, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), cfg.gradient_clip_norm)
    optimizer.step()

    return float(loss.item())


def _teacher_actions_for_envs(envs, anchor_corner: str) -> np.ndarray:
    return np.asarray(
        [teacher_action(env, anchor_corner=anchor_corner) for env in envs],
        dtype=np.int64,
    )


def _load_demonstrations_into_buffer(path: str, buffer: ReplayBuffer) -> int:
    demo_path = Path(path)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demonstration file not found: {demo_path}")

    data = np.load(demo_path)
    required_keys = {"obs", "action", "reward", "next_obs", "done"}
    missing = required_keys - set(data.files)
    if missing:
        raise ValueError(f"Demonstration file is missing keys: {sorted(missing)}")

    obs = np.asarray(data["obs"])
    action = np.asarray(data["action"])
    reward = np.asarray(data["reward"])
    next_obs = np.asarray(data["next_obs"])
    done = np.asarray(data["done"])
    n = int(action.shape[0])

    chunk_size = 8192
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        buffer.add_batch(
            obs[start:end],
            action[start:end],
            reward[start:end],
            next_obs[start:end],
            done[start:end],
        )

    return n


def evaluate(
    q_net: QNetwork,
    device: torch.device,
    *,
    n_episodes: int,
    max_steps: int,
    anchor_corner: str,
    use_shaping: bool,
    shaping_weights: dict[str, float],
) -> Tuple[float, float, int, float, float]:
    q_net.eval()

    scores = []
    max_tiles = []
    illegal_moves = 0
    corner_hits = 0
    total_steps = 0

    for ep in range(n_episodes):
        env = Game2048Env(
            seed=1000 + ep,
            use_shaping=use_shaping,
            shaping_weights=shaping_weights,
            anchor_corner=anchor_corner,
        )
        obs = env.reset()
        done = False
        steps = 0
        info = {}

        while (not done) and (steps < max_steps):
            obs_log = board_to_log2(obs)
            action, _q_values = choose_greedy_legal_action(
                q_net=q_net,
                obs_log=obs_log,
                env=env,
                device=device,
            )

            obs, _reward, done, info = env.step(action)
            steps += 1
            total_steps += 1

            if not bool(info.get("moved", True)):
                illegal_moves += 1

            corner_hits += int(max_tile_in_anchor_corner(env.board, anchor_corner))

        score = info.get("score", 0) if isinstance(info, dict) else 0
        scores.append(float(score))
        max_tiles.append(int(env.board.max()))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_tile = sum(max_tiles) / len(max_tiles) if max_tiles else 0.0
    best_tile = max(max_tiles) if max_tiles else 0
    illegal_rate = (illegal_moves / total_steps) if total_steps else 0.0
    corner_adherence = (corner_hits / max(total_steps, 1)) if total_steps else 0.0
    return (
        float(avg_score),
        float(avg_tile),
        int(best_tile),
        float(illegal_rate),
        float(corner_adherence),
    )


def save_checkpoint(
    path: str,
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    env_steps: int,
    iters: int,
    cfg: TrainConfig,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": q_net.state_dict(),
            "target_model": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "env_steps": int(env_steps),
            "iters": int(iters),
            "cfg": cfg.__dict__,
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str,
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location=device)
    q_net.load_state_dict(ckpt["model"])
    if "target_model" in ckpt:
        target_net.load_state_dict(ckpt["target_model"])
    else:
        target_net.load_state_dict(q_net.state_dict())
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    env_steps = int(ckpt.get("env_steps", 0))
    iters = int(ckpt.get("iters", 0))
    return env_steps, iters


def train(resume: Optional[str], checkpoint_path: str, cfg: TrainConfig) -> None:
    device = get_device()
    print("Using device:", device)
    print(
        "Strategy config:",
        {
            "use_shaping": cfg.use_shaping,
            "anchor_corner": cfg.anchor_corner,
            "warmup_env_steps": cfg.warmup_env_steps,
            "teacher_mix_start": cfg.teacher_mix_start,
            "teacher_mix_end": cfg.teacher_mix_end,
            "teacher_mix_decay_steps": cfg.teacher_mix_decay_steps,
            "demo_path": cfg.demo_path,
        },
    )

    vec = VecRunner(
        lambda seed: make_env(seed, cfg),
        n_envs=cfg.n_envs,
        base_seed=cfg.base_seed,
        auto_reset=True,
        obs_transform=board_to_log2,
    )

    q_net = QNetwork(
        max_level=cfg.max_level,
        emb_dim=cfg.emb_dim,
        n_actions=cfg.n_actions,
    ).to(device)
    target_net = QNetwork(
        max_level=cfg.max_level,
        emb_dim=cfg.emb_dim,
        n_actions=cfg.n_actions,
    ).to(device)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    buffer = ReplayBuffer(capacity=cfg.replay_capacity, device=str(device))

    env_steps = 0
    iters = 0

    if resume:
        print("Resuming from:", resume)
        env_steps, iters = load_checkpoint(resume, q_net, target_net, optimizer, device)
        print(f"Loaded checkpoint (env_steps={env_steps}, iters={iters})")
    else:
        target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    demos_loaded = 0
    if cfg.demo_path:
        demos_loaded = _load_demonstrations_into_buffer(cfg.demo_path, buffer)
        print(f"Loaded {demos_loaded} demo transitions from {cfg.demo_path}")


    warmup_env_steps = 0 if demos_loaded > 0 else int(cfg.warmup_env_steps)

    obs = vec.reset()
    next_target_update = env_steps + cfg.target_update_every
    next_eval_at = env_steps + cfg.eval_every

    losses = []
    t0 = time.time()

    while env_steps < cfg.total_env_steps:
        iters += 1

        if env_steps < warmup_env_steps:
            phase = "A"
            eps = 0.0
            teacher_mix = 1.0
            actions_np = _teacher_actions_for_envs(vec.envs, cfg.anchor_corner)
        else:
            eps = epsilon_by_step(env_steps, cfg)
            actions = select_epsilon_greedy_actions(q_net, obs, epsilon=eps)
            actions_np = actions.cpu().numpy().astype(np.int64)

            if env_steps < warmup_env_steps + cfg.teacher_mix_decay_steps:
                phase = "B"
                teacher_mix = teacher_mix_by_step(env_steps - warmup_env_steps, cfg)
                if teacher_mix > 0.0:
                    teacher_np = _teacher_actions_for_envs(vec.envs, cfg.anchor_corner)
                    override_mask = np.random.random(cfg.n_envs) < teacher_mix
                    actions_np[override_mask] = teacher_np[override_mask]
            else:
                phase = "C"
                teacher_mix = 0.0

        step = vec.step(actions_np)
        buffer.add_batch(
            obs,
            actions_np,
            step.reward,
            step.obs,
            step.done,
        )
        obs = step.obs
        env_steps += cfg.n_envs

        if (
            phase != "A"
            and len(buffer) >= cfg.min_replay_size
            and (env_steps % cfg.train_every_env_steps == 0)
        ):
            loss = dqn_update_double(q_net, target_net, optimizer, buffer, cfg, device)
            losses.append(loss)

        if env_steps >= next_target_update:
            target_net.load_state_dict(q_net.state_dict())
            next_target_update += cfg.target_update_every

        if iters % cfg.log_every_iters == 0:
            dt = time.time() - t0
            steps_per_sec = env_steps / max(dt, 1e-9)
            avg_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1)
            print(
                f"phase={phase} iters={iters} env_steps={env_steps} "
                f"eps={eps:.3f} teacher_mix={teacher_mix:.3f} "
                f"buffer={len(buffer)} avg_loss_100={avg_loss:.4f} "
                f"steps/s={steps_per_sec:.0f}"
            )

        if env_steps >= next_eval_at:
            avg_score, avg_tile, best_tile, illegal_rate, corner_rate = evaluate(
                q_net,
                device,
                n_episodes=cfg.eval_episodes,
                max_steps=cfg.eval_max_steps,
                anchor_corner=cfg.anchor_corner,
                use_shaping=cfg.use_shaping,
                shaping_weights=cfg.shaping_weights,
            )
            print(
                f"\nEVAL @ {env_steps} steps:"
                f"\n  Avg Score: {avg_score:.1f}"
                f"\n  Avg Max Tile: {avg_tile:.1f}"
                f"\n  Best Tile: {best_tile}"
                f"\n  Illegal Move Rate: {illegal_rate:.3%}"
                f"\n  Corner Adherence: {corner_rate:.3%}\n"
            )
            next_eval_at += cfg.eval_every

            save_checkpoint(
                checkpoint_path,
                q_net,
                target_net,
                optimizer,
                env_steps,
                iters,
                cfg,
            )
            print("Saved checkpoint:", checkpoint_path)

    save_checkpoint(
        checkpoint_path, q_net, target_net, optimizer, env_steps, iters, cfg
    )
    print("Saved final checkpoint:", checkpoint_path)

    model_path = Path("models/qnet_2048_dqn.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(q_net.state_dict(), model_path)
    print("Saved model weights:", model_path)


def main() -> None:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="models/checkpoint.pt")

    parser.add_argument("--use_shaping", action="store_true")
    parser.add_argument(
        "--anchor_corner",
        type=str,
        default=cfg.anchor_corner,
        choices=["TR", "TL", "BR", "BL"],
    )
    parser.add_argument(
        "--teacher_mix_start",
        type=float,
        default=cfg.teacher_mix_start,
    )
    parser.add_argument(
        "--teacher_mix_end",
        type=float,
        default=cfg.teacher_mix_end,
    )
    parser.add_argument(
        "--teacher_mix_decay_steps",
        type=int,
        default=cfg.teacher_mix_decay_steps,
    )
    parser.add_argument("--warmup_env_steps", type=int, default=cfg.warmup_env_steps)
    parser.add_argument("--demo_path", type=str, default=None)

    args = parser.parse_args()

    cfg.use_shaping = bool(args.use_shaping)
    cfg.anchor_corner = str(args.anchor_corner).upper()
    cfg.teacher_mix_start = float(args.teacher_mix_start)
    cfg.teacher_mix_end = float(args.teacher_mix_end)
    cfg.teacher_mix_decay_steps = int(args.teacher_mix_decay_steps)
    cfg.warmup_env_steps = int(args.warmup_env_steps)
    cfg.demo_path = args.demo_path

    checkpoint_path = Path(args.ckpt)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    train(resume=args.resume, checkpoint_path=str(checkpoint_path), cfg=cfg)


if __name__ == "__main__":
    main()
