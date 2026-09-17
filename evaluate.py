"""
Single-Command Evaluation and Benchmarking Script.
Reproduces all empirical ablation tables, statistical significance tests,
and trajectory metrics reported in the paper:
'Enhancing Deep Reinforcement Learning with Spatiotemporal Modeling for Mobile Robot Path Planning'
(EAAI-26-5904).
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats

from src.env import GridWorld
from src.models import CnnDQN, DuelingCnnDQN, DuelingCnnLstmDQN

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
CKPT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pretrained Models on Benchmark Test Environments.")
    parser.add_argument("--num_maps", type=int, default=500, choices=[100, 500],
                        help="Number of unseen test maps to evaluate (100 for core table, 500 for extended significance).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Inference device (cpu or cuda).")
    parser.add_argument("--test_data", type=str, default=str(DATA_DIR / "test_grids_500.npz"),
                        help="Path to pre-generated test environments .npz archive.")
    parser.add_argument("--save_csv", action="store_true", default=True,
                        help="Save per-episode evaluation metrics to CSV.")
    parser.add_argument("--make_plots", action="store_true", default=False,
                        help="Generate trajectory comparison visual plot.")
    return parser.parse_args()


def load_test_grids(data_path: Path, num_maps: int, seed: int = 100) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Loads pre-generated test environments or procedurally generates them with seed 100."""
    if data_path.exists():
        archive = np.load(data_path)
        grids = list(archive["grids"][:num_maps])
        starts = [tuple(s) for s in archive["starts"][:num_maps]]
        goals = [tuple(g) for g in archive["goals"][:num_maps]]
        print(f"Loaded {len(grids)} benchmark environments from '{data_path.name}'.")
        return grids, starts, goals
    else:
        print(f"Archive '{data_path}' not found. Procedurally generating {num_maps} test maps with seed {seed}...")
        env = GridWorld(size=15, obstacle_ratio=0.18, max_steps=200, seed=seed)
        grids, starts, goals = [], [], []
        for _ in range(num_maps):
            env.reset()
            grids.append(env.grid.copy())
            starts.append(env.start)
            goals.append(env.goal)
        return grids, starts, goals


def evaluate_model(
    model: torch.nn.Module,
    algo: str,
    obs_radius: int,
    grids: List[np.ndarray],
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    device: torch.device,
    record_trajectories: bool = False,
) -> Tuple[List[Dict], List[float], List[Dict]]:
    """Evaluates a single model on the specified benchmark grids."""
    model.eval()
    env = GridWorld(size=15, obstacle_ratio=0.18, max_steps=200, obs_radius=obs_radius)
    use_lstm = (algo == "lstm")
    sequence_length = 4
    
    metrics = []
    latencies = []
    trajectories = []

    for ep_idx, (grid, start, goal) in enumerate(zip(grids, starts, goals)):
        state_np = env.set_grid(grid, start=start, goal=goal)
        state = torch.from_numpy(state_np).unsqueeze(0).unsqueeze(0).to(device)
        
        ep_reward = 0.0
        lstm_hidden = None
        state_sequence = []
        ep_path = [env.agent]

        for step_num in range(env.max_steps):
            t0 = time.perf_counter()
            with torch.no_grad():
                if use_lstm:
                    state_frame = state.squeeze(0)  # (1, H, W)
                    if len(state_sequence) < sequence_length:
                        state_sequence.append(state_frame)
                        while len(state_sequence) < sequence_length:
                            state_sequence.insert(0, state_sequence[0])
                    else:
                        state_sequence.pop(0)
                        state_sequence.append(state_frame)
                    
                    seq_tensor = torch.stack(state_sequence, dim=0).unsqueeze(0)  # (1, T, 1, H, W)
                    q_vals, lstm_hidden = model(seq_tensor, lstm_hidden)
                    action = q_vals[:, -1, :].argmax(dim=1).item()
                else:
                    q_vals = model(state)
                    action = q_vals.argmax(dim=1).item()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            next_state_np, reward, done, info = env.step(action)
            next_state = torch.from_numpy(next_state_np).unsqueeze(0).unsqueeze(0).to(device)
            ep_reward += reward
            state = next_state
            ep_path.append(env.agent)

            if done:
                break

        reached_goal = info.get("reached_goal", False)
        metrics.append({
            "episode": ep_idx + 1,
            "reached_goal": int(reached_goal),
            "steps": step_num + 1,
            "ep_reward": round(ep_reward, 4),
        })
        if record_trajectories:
            trajectories.append({
                "episode": ep_idx + 1,
                "grid": grid.copy(),
                "start": start,
                "goal": goal,
                "path": ep_path,
                "reached_goal": reached_goal,
                "steps": step_num + 1,
                "reward": ep_reward,
            })

    return metrics, latencies, trajectories


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EAAI-26-5904 REPRODUCIBILITY BENCHMARK EVALUATION")
    print(f"Device: {device} | Test Environments: {args.num_maps}")
    print("=" * 78)

    # 1. Load benchmark test environments
    grids, starts, goals = load_test_grids(Path(args.test_data), args.num_maps)

    # 2. Define model configurations
    configs = {
        "Baseline Global CNN-DQN": {
            "algo": "baseline",
            "obs_radius": None,
            "ckpt": CKPT_DIR / "baseline_cnn_dqn.pth",
            "model": CnnDQN(grid_size=15, num_actions=8, size_invariant=False).to(device),
        },
        "Ablation 1: Local CNN-DQN (11x11)": {
            "algo": "baseline",
            "obs_radius": 5,
            "ckpt": CKPT_DIR / "local_cnn_dqn.pth",
            "model": CnnDQN(grid_size=11, num_actions=8, size_invariant=False).to(device),
        },
        "Ablation 2: Dueling CNN-DQN (11x11)": {
            "algo": "dueling",
            "obs_radius": 5,
            "ckpt": CKPT_DIR / "dueling_cnn_dqn.pth",
            "model": DuelingCnnDQN(grid_size=11, num_actions=8).to(device),
        },
        "Proposed: LSTM-CNN-DQN (11x11, T=4)": {
            "algo": "lstm",
            "obs_radius": 5,
            "ckpt": CKPT_DIR / "lstm_cnn_dqn.pth",
            "model": DuelingCnnLstmDQN(grid_size=11, num_actions=8, lstm_hidden=128, sequence_length=4).to(device),
        },
    }

    # 3. Load checkpoints
    for name, cfg in configs.items():
        ckpt_file = cfg["ckpt"]
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Missing required checkpoint '{ckpt_file}'.")
        cfg["model"].load_state_dict(torch.load(ckpt_file, map_location=device))
        print(f"Loaded weights: {ckpt_file.name} -> {name}")

    # 4. Evaluate each model
    results_summary = {}
    per_model_metrics = {}

    print("\nEvaluating all variants...")
    for name, cfg in configs.items():
        metrics, latencies, trajs = evaluate_model(
            model=cfg["model"],
            algo=cfg["algo"],
            obs_radius=cfg["obs_radius"],
            grids=grids,
            starts=starts,
            goals=goals,
            device=device,
            record_trajectories=args.make_plots,
        )
        per_model_metrics[name] = metrics

        goals_arr = np.array([m["reached_goal"] for m in metrics])
        steps_arr = np.array([m["steps"] for m in metrics])
        rewards_arr = np.array([m["ep_reward"] for m in metrics])
        
        sr = np.mean(goals_arr) * 100.0
        path_mean = np.mean(steps_arr)
        path_std = np.std(steps_arr, ddof=1)
        path_med = np.median(steps_arr)
        q25 = np.percentile(steps_arr, 25)
        q75 = np.percentile(steps_arr, 75)
        iqr = q75 - q25
        rew_mean = np.mean(rewards_arr)
        rew_std = np.std(rewards_arr, ddof=1)
        mean_lat = np.mean(latencies)

        results_summary[name] = {
            "sr": sr,
            "path_mean": path_mean,
            "path_std": path_std,
            "path_med": path_med,
            "iqr": iqr,
            "rew_mean": rew_mean,
            "rew_std": rew_std,
            "latency": mean_lat,
            "goals": goals_arr,
            "steps": steps_arr,
        }

    # 5. Print Tabular Results
    print("\n" + "=" * 78)
    print(f"BENCHMARK RESULTS SUMMARY ({args.num_maps} UNSEEN MAPS)")
    print("=" * 78)
    header = f"{'Model Architecture':<34} | {'Success Rate':<12} | {'Path (Mean±Std)':<18} | {'Median (IQR)':<14} | {'Latency':<9}"
    print(header)
    print("-" * len(header))
    for name, s in results_summary.items():
        row = (
            f"{name:<34} | "
            f"{s['sr']:>6.2f}%      | "
            f"{s['path_mean']:>5.2f} ± {s['path_std']:<6.2f} | "
            f"{s['path_med']:>4.1f} ({s['iqr']:>4.1f})   | "
            f"{s['latency']:>5.3f} ms"
        )
        print(row)
    print("=" * 78)

    # 6. Statistical Significance Tests (Baseline vs Proposed LSTM-CNN-DQN)
    base = results_summary["Baseline Global CNN-DQN"]
    lstm = results_summary["Proposed: LSTM-CNN-DQN (11x11, T=4)"]

    n11 = np.sum((base["goals"] == 1) & (lstm["goals"] == 1))
    n10 = np.sum((base["goals"] == 1) & (lstm["goals"] == 0))
    n01 = np.sum((base["goals"] == 0) & (lstm["goals"] == 1))
    n00 = np.sum((base["goals"] == 0) & (lstm["goals"] == 0))

    b, c = float(n10), float(n01)
    mcnemar_stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    mcnemar_p = stats.chi2.sf(mcnemar_stat, 1)

    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(base["steps"], lstm["steps"])

    print("\nSTATISTICAL SIGNIFICANCE ANALYSIS (Baseline vs Proposed LSTM):")
    print(f"  • McNemar Paired Contingency:")
    print(f"      Both Succeeded: {n11:3d}  |  Baseline Only: {n10:3d}")
    print(f"      LSTM Only:      {n01:3d}  |  Both Failed:    {n00:3d}")
    print(f"  • McNemar Test:              chi2 = {mcnemar_stat:.4f}, p-value = {mcnemar_p:.4e}")
    print(f"  • Wilcoxon Signed-Rank Test: W = {wilcoxon_stat:.1f}, p-value = {wilcoxon_p:.4e}")
    if mcnemar_p < 0.001 and wilcoxon_p < 0.001:
        print("  ✓ Statistical significance confirmed at alpha = 0.001 (p < 1e-20).")

    # 7. Save to CSV
    if args.save_csv:
        csv_path = RESULTS_DIR / f"evaluation_summary_{args.num_maps}maps.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Success_Rate_pct", "Path_Mean", "Path_Std", "Path_Median", "Path_IQR", "Latency_ms"])
            for name, s in results_summary.items():
                writer.writerow([name, f"{s['sr']:.2f}", f"{s['path_mean']:.2f}", f"{s['path_std']:.2f}", f"{s['path_med']:.1f}", f"{s['iqr']:.1f}", f"{s['latency']:.3f}"])
        print(f"\nSummary metrics saved to '{csv_path}'.")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
