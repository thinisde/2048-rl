# 2048 RL Project

This project trains a Deep Q-Network (DQN) agent to play 2048 using a strategy-guided reinforcement learning pipeline.

## Project Structure

- `rl2048/`: core RL code (environment, model, replay buffer, shaping, teacher policy)
- `scripts/`: executable training and play scripts
- `web/`: Flask-based visualizer UI and API
- `models/`: saved checkpoints and model weights
- `train.py`: main training entrypoint
- `play.py`: terminal play entrypoint
- `serve_web.py`: web visualizer entrypoint

## How the Model Works

The model is a Q-network (`rl2048/qnet.py`) that predicts Q-values for the 4 actions:

- `0`: up
- `1`: down
- `2`: left
- `3`: right

Input is the board encoded in log2 space:

- `0 -> 0`
- `2 -> 1`
- `4 -> 2`
- ...

Architecture:

1. Tile embedding layer (`nn.Embedding`)
2. Two convolution layers with ReLU
3. Fully connected Q-head outputting 4 Q-values

At inference time, the policy selects the highest-Q legal move.

## Environment and Rewards

The environment is `Game2048Env` in `rl2048/game.py`.

Base reward:

- merge score increase from the move

Optional strategy shaping (`--use_shaping`):

- corner adherence improvement
- anchor-row fill improvement
- monotone snake improvement
- big-tile proximity improvement
- smoothness improvement
- empty-cell improvement
- trapped-small-tile reduction

Illegal/no-op moves are penalized strongly.

Default shaping weights:

- `merge_reward`: `1.0`
- `corner_bonus`: `2.0`
- `anchor_row_fill`: `0.5`
- `monotone_snake`: `0.2`
- `big_tile_proximity`: `0.02`
- `smoothness`: `0.05`
- `empty`: `0.3`
- `trap`: `0.3`
- `illegal_penalty`: `3.0`

## Strategy Teacher

`rl2048/strategy_teacher.py` provides a deterministic teacher policy with an anchor corner (default `TR`).

Behavior:

- prefers anchor-preserving moves (`RIGHT`/`UP` for `TR`)
- uses fallback directions when needed
- avoids illegal moves
- prefers moves that allow restoring the corner quickly if displaced

It can generate demonstrations (`.npz`) with:

- `obs` (log2 board)
- `action`
- `reward`
- `next_obs`
- `done`

## Training Pipeline

Main script: `train.py` (calls `scripts/train.py`).

Training uses:

- Double DQN
- Huber loss
- target network updates
- replay buffer sampling
- epsilon-greedy exploration

Curriculum phases:

1. Phase A (warm-up): teacher-only actions to seed replay
2. Phase B (mixed): epsilon-greedy policy with teacher override probability decay
3. Phase C (pure RL): normal RL policy without teacher override

## Install

```bash
pip install -r requirements.txt
```

## Generate Teacher Demonstrations

```bash
python strategy_teacher.py --episodes 200 --out models/teacher_demos.npz --anchor_corner TR
```

## Train

Strategy-guided run:

```bash
python train.py \
  --use_shaping \
  --anchor_corner TR \
  --teacher_mix_start 0.3 \
  --teacher_mix_end 0.1 \
  --teacher_mix_decay_steps 1000000 \
  --warmup_env_steps 100000 \
  --demo_path models/teacher_demos.npz \
  --ckpt models/checkpoint.pt
```

Baseline-like run (no shaping, no demos):

```bash
python train.py --ckpt models/checkpoint.pt
```

Training outputs:

- checkpoint: `models/checkpoint.pt`
- model weights: `models/qnet_2048_dqn.pt`

## Play in Terminal

```bash
python play.py --model models/qnet_2048_dqn.pt --delay 0.1 --seed 123
```

## Visualize in Browser

```bash
python serve_web.py --model models/qnet_2048_dqn.pt --host 127.0.0.1 --port 5000
```

Open:

- `http://127.0.0.1:5000`

The UI supports:

- manual moves
- single model step
- autoplay
- live score and max tile
- per-action Q-values

## Evaluate

Evaluation inside training logs includes:

- average score
- average max tile
- best tile
- illegal move rate
- corner adherence rate

You can trigger these metrics by running training with periodic eval enabled (`eval_every` in `scripts/train.py`).

## Reproducibility Notes

- Different devices (`cpu`/`mps`) and random seeds produce different outcomes.
- Teacher demos can improve early stability and reduce illegal-move loops.
