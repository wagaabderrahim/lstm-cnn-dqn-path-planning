"""
Training Script for Baseline and Proposed DQN Architectures.
Supports:
  - baseline:      Global 15x15 CNN-DQN
  - cnn_local:     Local 11x11 CNN-DQN
  - dueling_local: Local 11x11 Dueling CNN-DQN
  - lstm:          Local 11x11 Dueling CNN-LSTM-DQN (Proposed)
"""

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.env import GridWorld
from src.models import CnnDQN, DuelingCnnDQN, DuelingCnnLstmDQN
from src.replay_buffer import ReplayBuffer, SequenceReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN Path Planning Agents.")
    parser.add_argument("--algo", type=str, default="lstm",
                        choices=["baseline", "cnn_local", "dueling_local", "lstm"],
                        help="Architecture to train.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per episode.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (Adam).")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--seq_len", type=int, default=4, help="Sequence length for LSTM.")
    parser.add_argument("--lstm_hidden", type=int, default=128, help="LSTM hidden units.")
    parser.add_argument("--target_update", type=int, default=10, help="Target network update interval (episodes).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--data", type=str, default="data/train_grids_500.npz",
                        help="Optional pre-generated training environments.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training '{args.algo}' on {device} for {args.episodes} episodes...")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    obs_radius = None if args.algo == "baseline" else 5
    env = GridWorld(size=15, obstacle_ratio=0.18, max_steps=args.max_steps, seed=args.seed, obs_radius=obs_radius)

    # Pre-generated training environments if present
    train_grids = None
    if os.path.exists(args.data):
        archive = np.load(args.data)
        train_grids = archive["grids"]
        print(f"Using {len(train_grids)} training environments from '{args.data}'.")

    # Instantiate Networks
    if args.algo == "baseline":
        policy_net = CnnDQN(grid_size=15, num_actions=8).to(device)
        target_net = CnnDQN(grid_size=15, num_actions=8).to(device)
        buffer = ReplayBuffer(capacity=50_000, device=device)
    elif args.algo == "cnn_local":
        policy_net = CnnDQN(grid_size=11, num_actions=8).to(device)
        target_net = CnnDQN(grid_size=11, num_actions=8).to(device)
        buffer = ReplayBuffer(capacity=50_000, device=device)
    elif args.algo == "dueling_local":
        policy_net = DuelingCnnDQN(grid_size=11, num_actions=8).to(device)
        target_net = DuelingCnnDQN(grid_size=11, num_actions=8).to(device)
        buffer = ReplayBuffer(capacity=50_000, device=device)
    elif args.algo == "lstm":
        policy_net = DuelingCnnLstmDQN(grid_size=11, num_actions=8, lstm_hidden=args.lstm_hidden, sequence_length=args.seq_len).to(device)
        target_net = DuelingCnnLstmDQN(grid_size=11, num_actions=8, lstm_hidden=args.lstm_hidden, sequence_length=args.seq_len).to(device)
        buffer = SequenceReplayBuffer(capacity=50_000, sequence_length=args.seq_len, device=device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    metrics_file = save_dir / f"{args.algo}_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "ep_reward", "steps", "reached_goal", "epsilon"])

    for ep in range(1, args.episodes + 1):
        if train_grids is not None and len(train_grids) > 0:
            grid_idx = (ep - 1) % len(train_grids)
            state_np = env.set_grid(train_grids[grid_idx])
        else:
            state_np = env.reset()

        state = torch.from_numpy(state_np).unsqueeze(0).unsqueeze(0).to(device)
        ep_reward = 0.0
        lstm_hidden = None
        state_seq = [state.squeeze(0)] * args.seq_len if args.algo == "lstm" else None

        for step in range(args.max_steps):
            # Action Selection
            if random.random() < epsilon:
                action = random.randrange(8)
            else:
                with torch.no_grad():
                    if args.algo == "lstm":
                        seq_in = torch.stack(state_seq, dim=0).unsqueeze(0)  # (1, T, 1, H, W)
                        q_vals, lstm_hidden = policy_net(seq_in, lstm_hidden)
                        action = q_vals[:, -1, :].argmax(dim=1).item()
                    else:
                        q_vals = policy_net(state)
                        action = q_vals.argmax(dim=1).item()

            next_state_np, reward, done, info = env.step(action)
            next_state = torch.from_numpy(next_state_np).unsqueeze(0).unsqueeze(0).to(device)
            ep_reward += reward

            # Buffer Storage
            if args.algo == "lstm":
                next_state_seq = state_seq[1:] + [next_state.squeeze(0)]
                buffer.push(
                    torch.stack(state_seq),
                    action,
                    reward,
                    torch.stack(next_state_seq),
                    done,
                )
                state_seq = next_state_seq
            else:
                buffer.push(state, action, reward, next_state, done)

            state = next_state

            # Gradient Step
            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                if args.algo == "lstm":
                    # batch.state_sequence: (B, T, 1, H, W)
                    curr_q, _ = policy_net(batch.state_sequence)  # (B, T, num_actions)
                    q_val = curr_q[:, -1, :].gather(1, batch.action.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        next_q, _ = target_net(batch.next_state_sequence)
                        max_next_q = next_q[:, -1, :].max(1)[0]
                        target_q = batch.reward + (args.gamma * max_next_q * (~batch.done))
                else:
                    curr_q = policy_net(batch.state)
                    q_val = curr_q.gather(1, batch.action.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        next_q = target_net(batch.next_state)
                        max_next_q = next_q.max(1)[0]
                        target_q = batch.reward + (args.gamma * max_next_q * (~batch.done))

                loss = loss_fn(q_val, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reached_goal = int(info.get("reached_goal", False))

        # Log episode metrics
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, round(ep_reward, 4), step + 1, reached_goal, round(epsilon, 4)])

        if ep % 50 == 0 or ep == 1:
            print(f"Episode {ep:3d}/{args.episodes} | Steps: {step+1:3d} | Reward: {ep_reward:6.2f} | Goal: {reached_goal} | Eps: {epsilon:.3f}")

        if ep % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # Save final model
    ckpt_path = save_dir / f"{args.algo}_final.pth"
    torch.save(policy_net.state_dict(), ckpt_path)
    print(f"\nTraining completed. Saved model weights to '{ckpt_path}'.")


if __name__ == "__main__":
    main()
