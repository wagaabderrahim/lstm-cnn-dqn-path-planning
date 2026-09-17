"""
Reference implementation of the CNN-DQN path-planning pipeline described in
“Convolutional neural network-based deep Q-network (CNN-DQN) path planning
method for mobile robots” (Intelligent Service Robotics, 2025).

This script is self-contained and mirrors the paper’s main ideas:
1) CNN front-end to process a 2D occupancy grid and output Q-values.
2) Exponential-decay greedy exploration (high exploration early, then decays).
3) Endpoint-biased reward to encourage shortest collision-free paths.
4) Simple B-spline post-processing for trajectory smoothing.

The demo trains on a toy gridworld with random rectangular obstacles.
You can adapt the environment size, obstacle density, and training budget.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from tqdm import tqdm

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def a_star_path_exists(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """
    Vérifie si un chemin existe entre start et goal en utilisant A*.
    Retourne True si un chemin existe, False sinon.
    
    Args:
        grid: Grille d'occupation (0=libre, 1=obstacle)
        start: Position de départ (row, col)
        goal: Position d'arrivée (row, col)
    
    Returns:
        True si un chemin existe, False sinon
    """
    if grid[start] == 1.0 or grid[goal] == 1.0:
        return False
    
    h, w = grid.shape
    # Directions 8-connectées
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    
    def heuristic(pos: Tuple[int, int]) -> float:
        """Distance de Manhattan comme heuristique."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    # Structures pour A*
    open_set = [(0, start)]  # (f_score, position)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start)}
    closed_set = set()
    
    while open_set:
        # Prendre le nœud avec le plus petit f_score
        open_set.sort()
        current_f, current = open_set.pop(0)
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        # Si on a atteint le but, un chemin existe
        if current == goal:
            return True
        
        # Explorer les voisins
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            
            # Vérifier les limites
            if neighbor[0] < 0 or neighbor[0] >= h or neighbor[1] < 0 or neighbor[1] >= w:
                continue
            
            # Vérifier si c'est un obstacle
            if grid[neighbor] == 1.0:
                continue
            
            if neighbor in closed_set:
                continue
            
            # Coût de déplacement (1 pour cardinal, sqrt(2) pour diagonal)
            move_cost = 1.0 if abs(dr) + abs(dc) == 1 else 1.414
            tentative_g = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor)
                if neighbor not in [item[1] for item in open_set]:
                    open_set.append((f_score[neighbor], neighbor))
    
    # Aucun chemin trouvé
    return False


class GridWorld:
    """Minimal 2D occupancy-grid environment for path planning.
    
    Supports:
    - Local observation windows (obs_radius) to reduce computational complexity
    """

    def __init__(
        self,
        size: int = 15,
        obstacle_ratio: float = 0.18,
        max_steps: int = 200,
        seed: int | None = None,
        obs_radius: Optional[int] = None,
    ) -> None:
        self.size = size
        self.obstacle_ratio = obstacle_ratio
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.obs_radius = obs_radius  # None = full map, else local window size

        self.grid = np.zeros((size, size), dtype=np.float32)
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.agent = self.start
        self.steps = 0

    def reset(self, max_attempts: int = 100) -> np.ndarray:
        """
        Réinitialise l'environnement en générant une nouvelle grille avec obstacles.
        Vérifie qu'un chemin existe entre start et goal avec A* avant d'accepter la grille.
        
        Args:
            max_attempts: Nombre maximum de tentatives pour générer une grille valide
        
        Returns:
            Observation initiale
        """
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        
        # Générer des grilles jusqu'à trouver une avec un chemin valide
        for attempt in range(max_attempts):
            self.grid.fill(0.0)
            
            # Random rectangular obstacles
            num_obstacles = int(self.size * self.size * self.obstacle_ratio / 4)
            for _ in range(num_obstacles):
                h = self.rng.integers(1, 4)
                w = self.rng.integers(1, 4)
                r = self.rng.integers(0, self.size - h)
                c = self.rng.integers(0, self.size - w)
                self.grid[r : r + h, c : c + w] = 1.0

            # Keep start/goal free
            self.grid[self.start] = 0.0
            self.grid[self.goal] = 0.0
            
            # Vérifier qu'un chemin existe avec A*
            if a_star_path_exists(self.grid, self.start, self.goal):
                # Grille valide trouvée
                self.agent = self.start
                self.steps = 0
                return self._obs()
        
        # Si on n'a pas trouvé de grille valide après max_attempts tentatives,
        # on retourne quand même la dernière grille générée (avec un warning)
        import warnings
        warnings.warn(
            f"Impossible de générer une grille valide après {max_attempts} tentatives. "
            f"Utilisation de la dernière grille générée (chemin peut ne pas exister)."
        )
        self.agent = self.start
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        """Return observation: full map or local window around agent."""
        if self.obs_radius is None:
            # Full map observation (original behavior)
            obs = np.copy(self.grid)
            obs[self.agent] = -1.0  # mark agent
            obs[self.goal] = 2.0    # mark goal
            return obs
        else:
            # Local window observation (reduces computational complexity)
            R = self.obs_radius
            window_size = 2 * R + 1
            obs = np.ones((window_size, window_size), dtype=np.float32) * 1.0  # Default: obstacles
            
            ar, ac = self.agent
            for i in range(window_size):
                for j in range(window_size):
                    # Map window coordinates to global grid
                    gr = ar - R + i
                    gc = ac - R + j
                    if 0 <= gr < self.size and 0 <= gc < self.size:
                        obs[i, j] = self.grid[gr, gc]
                    # else: remains 1.0 (obstacle)
            
            # Mark agent position (center of window)
            obs[R, R] = -1.0
            
            # Mark goal if visible in window
            gr, gc = self.goal
            if abs(gr - ar) <= R and abs(gc - ac) <= R:
                obs[gr - ar + R, gc - ac + R] = 2.0
            
            return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Actions (8 directions):
        0=up, 1=down, 2=left, 3=right,
        4=up-left, 5=up-right, 6=down-left, 7=down-right
        """
        r, c = self.agent
        directions = [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1),   # right
            (-1, -1), # up-left
            (-1, 1),  # up-right
            (1, -1),  # down-left
            (1, 1),   # down-right
        ]
        dr, dc = directions[action]
        r = min(max(r + dr, 0), self.size - 1)
        c = min(max(c + dc, 0), self.size - 1)

        self.steps += 1
        next_pos = (r, c)

        # Collision check
        collision = self.grid[next_pos] == 1.0
        if collision:
            # Stay in place on collision
            next_pos = self.agent

        # Reward shaping: encourage approaching goal, penalize collisions/steps
        old_dist = self._manhattan(self.agent, self.goal)
        new_dist = self._manhattan(next_pos, self.goal)
        progress_reward = (old_dist - new_dist) * 0.5
        step_penalty = -0.05
        collision_penalty = -0.3 if collision else 0.0
        reached_goal = next_pos == self.goal
        goal_bonus = 5.0 if reached_goal else 0.0

        reward = progress_reward + step_penalty + collision_penalty + goal_bonus
        self.agent = next_pos
        
        done = reached_goal or self.steps >= self.max_steps
        info = {"reached_goal": reached_goal}
        return self._obs(), reward, done, info

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --------------------------------------------------------------------------- #
# Replay Buffer
# --------------------------------------------------------------------------- #


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        # States are stored as (1, 1, H, W), need to squeeze batch dim before stacking
        states = [b.state.squeeze(0) if b.state.dim() == 4 else b.state for b in batch]  # (1, H, W) each
        next_states = [b.next_state.squeeze(0) if b.next_state.dim() == 4 else b.next_state for b in batch]
        return Transition(
            state=torch.stack(states),  # (B, 1, H, W)
            action=torch.tensor([b.action for b in batch], device=Device),
            reward=torch.tensor([b.reward for b in batch], device=Device),
            next_state=torch.stack(next_states),  # (B, 1, H, W)
            done=torch.tensor([b.done for b in batch], dtype=torch.bool, device=Device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# --------------------------------------------------------------------------- #
# Prioritized Replay Buffer (for improved DQN variants)
# --------------------------------------------------------------------------- #


class PrioritizedReplayBuffer:
    """Simple proportional prioritized replay buffer.

    Priorities are updated from TD-errors; sampling is proportional to priority^alpha.
    """

    def __init__(self, capacity: int = 50_000, alpha: float = 0.6, beta: float = 0.4) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer: List[Transition] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, *args) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0
        transition = Transition(*args)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Transition, np.ndarray, np.ndarray]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights_t = torch.tensor(weights, dtype=torch.float32, device=Device)

        # States are stored as (1, 1, H, W), need to squeeze batch dim before stacking
        states = [s.state.squeeze(0) if s.state.dim() == 4 else s.state for s in samples]
        next_states = [s.next_state.squeeze(0) if s.next_state.dim() == 4 else s.next_state for s in samples]
        batch = Transition(
            state=torch.stack(states),  # (B, 1, H, W)
            action=torch.tensor([s.action for s in samples], device=Device),
            reward=torch.tensor([s.reward for s in samples], device=Device),
            next_state=torch.stack(next_states),  # (B, 1, H, W)
            done=torch.tensor([s.done for s in samples], dtype=torch.bool, device=Device),
        )
        return batch, indices, weights_t

    def update_priorities(self, indices: np.ndarray, td_errors: torch.Tensor) -> None:
        td = td_errors.detach().abs().cpu().numpy() + 1e-6
        self.priorities[indices] = td

    def __len__(self) -> int:
        return len(self.buffer)


# --------------------------------------------------------------------------- #
# Sequence Replay Buffer (for LSTM-based models)
# --------------------------------------------------------------------------- #


@dataclass
class SequenceTransition:
    """Transition storing a sequence of states for LSTM processing."""
    state_sequence: torch.Tensor  # (T, 1, H, W) or (T, H, W)
    action: int
    reward: float
    next_state_sequence: torch.Tensor
    done: bool


class SequenceReplayBuffer:
    """Replay buffer for storing sequences of states (for LSTM models).
    
    Stores sequences of length T, where T is the sequence_length.
    """

    def __init__(self, capacity: int = 50_000, sequence_length: int = 4) -> None:
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer: Deque[SequenceTransition] = deque(maxlen=capacity)

    def push(self, state_seq: torch.Tensor, action: int, reward: float, next_state_seq: torch.Tensor, done: bool) -> None:
        """Push a sequence transition into the buffer."""
        self.buffer.append(SequenceTransition(state_seq, action, reward, next_state_seq, done))

    def sample(self, batch_size: int) -> SequenceTransition:
        """Sample a batch of sequences."""
        batch = random.sample(self.buffer, batch_size)
        return SequenceTransition(
            state_sequence=torch.stack([b.state_sequence for b in batch]),  # (B, T, 1, H, W)
            action=torch.tensor([b.action for b in batch], device=Device),
            reward=torch.tensor([b.reward for b in batch], device=Device),
            next_state_sequence=torch.stack([b.next_state_sequence for b in batch]),
            done=torch.tensor([b.done for b in batch], dtype=torch.bool, device=Device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# --------------------------------------------------------------------------- #
# CNN-DQN Model
# --------------------------------------------------------------------------- #


class CnnDQN(nn.Module):
    """CNN-DQN avec entrée 2D (carte d'occupation). Si size_invariant=True, utilise
    Global Average Pooling pour accepter n'importe quelle taille de grille (15x15, 100x100, etc.)."""

    def __init__(self, grid_size: int, num_actions: int, size_invariant: bool = False) -> None:
        super().__init__()
        self.size_invariant = size_invariant
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        if size_invariant:
            # Global Average Pooling : sortie (B, 32, 1, 1) → même modèle pour toute taille
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions),
            )
        else:
            feat_dim = grid_size * grid_size * 32
            self.pool = None
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        if self.pool is not None:
            feat = self.pool(feat)
        return self.fc(feat)


class DuelingCnnDQN(nn.Module):
    """Dueling architecture: shared CNN trunk, then value and advantage streams.
    Si size_invariant=True, accepte n'importe quelle taille de grille (Global Average Pooling)."""

    def __init__(self, grid_size: int, num_actions: int, size_invariant: bool = False) -> None:
        super().__init__()
        self.size_invariant = size_invariant
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        if size_invariant:
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = 32
        else:
            self.pool = None
            feat_dim = grid_size * grid_size * 32
        self.value_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        if self.pool is not None:
            feat = self.pool(feat)
        value = self.value_stream(feat)
        adv = self.adv_stream(feat)
        adv_mean = adv.mean(dim=1, keepdim=True)
        return value + adv - adv_mean


class ForwardModelCNN(nn.Module):
    """Lightweight forward model for curiosity: predicts next state from (state, action).

    Input: (B, 2, H, W)  -> state channel + action channel
    Output: (B, 1, H, W) -> predicted next-state occupancy grid
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingCnnLstmDQN(nn.Module):
    """Dueling CNN-LSTM architecture for handling temporal sequences.
    
    Architecture:
    - CNN extracts features from each frame in the sequence
    - LSTM processes the sequence of features
    - Dueling head (value + advantage) outputs Q-values
    """

    def __init__(self, grid_size: int, num_actions: int, lstm_hidden: int = 128, sequence_length: int = 4) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.lstm_hidden = lstm_hidden
        
        # CNN feature extractor (shared across frames)
        self.cnn_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_feat_dim = grid_size * grid_size * 32
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(cnn_feat_dim, lstm_hidden, batch_first=True)
        
        # Dueling head
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T, 1, H, W) or (B, 1, H, W) - sequence of frames or single frame
            hidden: Optional LSTM hidden state
        Returns:
            q_values: (B, num_actions) or (B, T, num_actions)
            hidden: LSTM hidden state
        """
        # Handle single frame case (expand to sequence)
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, 1, H, W)
        
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        
        # Extract CNN features for each frame
        cnn_feat = self.cnn_features(x_flat)  # (B*T, feat_dim)
        cnn_feat = cnn_feat.view(B, T, -1)  # (B, T, feat_dim)
        
        # Process sequence with LSTM
        lstm_out, hidden = self.lstm(cnn_feat, hidden)  # (B, T, lstm_hidden)
        
        # Use last timestep for Q-values (or all timesteps if T > 1)
        if T == 1:
            lstm_last = lstm_out[:, -1, :]  # (B, lstm_hidden)
            value = self.value_stream(lstm_last)  # (B, 1)
            adv = self.adv_stream(lstm_last)  # (B, num_actions)
            adv_mean = adv.mean(dim=1, keepdim=True)
            q = value + adv - adv_mean  # (B, num_actions)
            return q, hidden
        else:
            # Return Q-values for all timesteps
            value = self.value_stream(lstm_out)  # (B, T, 1)
            adv = self.adv_stream(lstm_out)  # (B, T, num_actions)
            adv_mean = adv.mean(dim=2, keepdim=True)
            q = value + adv - adv_mean  # (B, T, num_actions)
            return q, hidden


# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #


def exponential_epsilon(step: int, eps_start: float, eps_end: float, decay: float) -> float:
    """Greedy-exploration schedule based on exponential decay."""
    return eps_end + (eps_start - eps_end) * math.exp(-1.0 * step / decay)


def select_action(
    policy: nn.Module, state: torch.Tensor, step: int, eps_start: float, eps_end: float, decay: float
) -> int:
    eps = exponential_epsilon(step, eps_start, eps_end, decay)
    if random.random() < eps:
        return random.randrange(8)
    with torch.no_grad():
        # state is already (1, 1, H, W) with batch dimension
        q_values = policy(state)
        return int(torch.argmax(q_values, dim=1).item())


def select_action_lstm(
    policy: DuelingCnnLstmDQN,
    state_sequence: torch.Tensor,
    hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
    step: int,
    eps_start: float,
    eps_end: float,
    decay: float,
) -> Tuple[int, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Select action using LSTM policy with sequence input."""
    eps = exponential_epsilon(step, eps_start, eps_end, decay)
    if random.random() < eps:
        return random.randrange(8), hidden
    
    with torch.no_grad():
        # state_sequence: (T, 1, H, W) -> (1, T, 1, H, W)
        q_values, new_hidden = policy(state_sequence.unsqueeze(0), hidden)
        # q_values: (1, num_actions) if T=1, else (1, T, num_actions)
        if q_values.dim() == 3:
            q_values = q_values[:, -1, :]  # Use last timestep
        return int(torch.argmax(q_values, dim=1).item()), new_hidden


def optimize_model(
    policy: nn.Module,
    target: nn.Module,
    buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    gamma: float,
    batch_size: int,
) -> float:
    if len(buffer) < batch_size:
        return 0.0
    batch = buffer.sample(batch_size)

    q_values = policy(batch.state).gather(1, batch.action.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target(batch.next_state).max(1)[0]
        target_q = batch.reward + gamma * next_q * (~batch.done)

    loss = nn.functional.mse_loss(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def optimize_model_prioritized(
    policy: nn.Module,
    target: nn.Module,
    buffer: PrioritizedReplayBuffer,
    optimizer: optim.Optimizer,
    gamma: float,
    batch_size: int,
) -> Tuple[float, Optional[torch.Tensor]]:
    """DQN update with prioritized replay and Double DQN target."""
    if len(buffer) < batch_size:
        return 0.0, None
    batch, indices, weights = buffer.sample(batch_size)

    # Current Q(s,a)
    q_values = policy(batch.state).gather(1, batch.action.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: argmax from policy, value from target
        next_q_policy = policy(batch.next_state)
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
        next_q_target = target(batch.next_state).gather(1, next_actions).squeeze(1)
        target_q = batch.reward + gamma * next_q_target * (~batch.done)

    td_errors = target_q - q_values
    loss = (weights * td_errors.pow(2)).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    buffer.update_priorities(indices, td_errors)
    return float(loss.item()), td_errors


def optimize_model_lstm(
    policy: DuelingCnnLstmDQN,
    target: DuelingCnnLstmDQN,
    buffer: SequenceReplayBuffer,
    optimizer: optim.Optimizer,
    gamma: float,
    batch_size: int,
) -> float:
    """DQN update for LSTM-based models with sequence transitions."""
    if len(buffer) < batch_size:
        return 0.0
    
    batch = buffer.sample(batch_size)
    # batch.state_sequence: (B, T, 1, H, W)
    # batch.next_state_sequence: (B, T, 1, H, W)
    
    # Get Q-values for current state sequences (use last timestep)
    q_values_seq, _ = policy(batch.state_sequence, None)
    q_values = q_values_seq[:, -1, :].gather(1, batch.action.unsqueeze(1)).squeeze(1)  # (B,)
    
    with torch.no_grad():
        # Double DQN: use policy to select action, target to evaluate
        next_q_policy_seq, _ = policy(batch.next_state_sequence, None)
        next_q_policy = next_q_policy_seq[:, -1, :]  # (B, num_actions)
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
        
        next_q_target_seq, _ = target(batch.next_state_sequence, None)
        next_q_target = next_q_target_seq[:, -1, :].gather(1, next_actions).squeeze(1)  # (B,)
        
        target_q = batch.reward + gamma * next_q_target * (~batch.done)
    
    loss = nn.functional.mse_loss(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


# --------------------------------------------------------------------------- #
# Path smoothing with B-spline
# --------------------------------------------------------------------------- #


def smooth_path(path: List[Tuple[int, int]], smoothing: float = 0.0, k: int = 3) -> np.ndarray:
    """Convert discrete grid path to a smooth B-spline curve.

    Robustness tweaks:
    - Drop consecutive duplicates (collisions can cause repeats).
    - Require at least k+1 unique points.
    - Fallback to raw path if splprep fails.
    """
    # Remove consecutive duplicates
    dedup: List[Tuple[int, int]] = []
    for p in path:
        if not dedup or p != dedup[-1]:
            dedup.append(p)

    # Need enough unique points for chosen spline degree
    if len(dedup) < k + 1:
        return np.array(dedup, dtype=np.float32)

    pts = np.array(dedup, dtype=np.float32).T

    # Guard against zero-variance inputs (all points aligned/identical)
    if np.allclose(pts.std(axis=1), 0):
        return np.array(dedup, dtype=np.float32)

    try:
        tck, _ = splprep(pts, s=smoothing, k=min(k, len(dedup) - 1))
        u_fine = np.linspace(0, 1, num=max(50, len(dedup) * 3))
        x_fine, y_fine = splev(u_fine, tck)
        return np.stack([x_fine, y_fine], axis=1)
    except Exception:
        # If fitting fails, return the unsmoothed path
        return np.array(dedup, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #


def plot_episode(
    grid: np.ndarray,
    path: List[Tuple[int, int]],
    smoothed: np.ndarray | None,
    ep: int,
    ep_reward: float,
    reached_goal: bool,
    log_dir: Path,
) -> None:
    """Save a visualization of the grid, discrete path, and smoothed path."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="gray_r", origin="lower")

    if path:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 1], path_arr[:, 0], "-o", color="tab:blue", markersize=3, label="path")

    if smoothed is not None and len(smoothed) > 1:
        ax.plot(smoothed[:, 1], smoothed[:, 0], "-", color="tab:orange", label="B-spline")

    ax.scatter([0], [0], c="green", marker="s", s=60, label="start")
    ax.scatter([grid.shape[1] - 1], [grid.shape[0] - 1], c="red", marker="*", s=100, label="goal")
    ax.set_title(f"Episode {ep+1} | R={ep_reward:.2f} | goal={reached_goal}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_file = log_dir / f"episode_{ep+1:03d}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_env_only(
    grid: np.ndarray,
    ep: int,
    log_dir: Path,
) -> None:
    """Plot uniquement l'environnement (obstacles + start/goal), sans trajectoire."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="gray_r", origin="lower")

    # On suppose start=(0,0), goal=(H-1,W-1) comme dans GridWorld
    h, w = grid.shape
    ax.scatter([0], [0], c="green", marker="s", s=60, label="start")
    ax.scatter([w - 1], [h - 1], c="red", marker="*", s=100, label="goal")

    ax.set_title(f"Environnement d'entraînement {ep+1}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = log_dir / f"train_env_{ep+1:03d}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def save_episode_data(
    log_dir: Path,
    ep: int,
    grid: np.ndarray,
    path: List[Tuple[int, int]],
    ep_reward: float,
    reached_goal: bool,
) -> None:
    """Sauvegarde les données brutes d'un épisode pour analyse ultérieure.

    Contenu (npz):
    - grid: carte d'occupation (H, W)
    - path: trajectoire discrète (N, 2) [row, col]
    - ep_reward: récompense totale de l'épisode
    - reached_goal: booléen (1 ou 0)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    path_arr = np.array(path, dtype=np.int32) if path else np.empty((0, 2), dtype=np.int32)
    out_file = log_dir / f"episode_{ep+1:03d}_data.npz"
    np.savez_compressed(
        out_file,
        grid=grid.astype(np.float32),
        path=path_arr,
        ep_reward=np.array(ep_reward, dtype=np.float32),
        reached_goal=np.array(int(reached_goal), dtype=np.int8),
    )


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #


def train(
    episodes: int = 10,
    grid_size: int = 15,
    obstacle_ratio: float = 0.18,
    max_steps: int = 200,
    eps_start: float = 0.9,
    eps_end: float = 0.05,
    eps_decay: float = 500.0,
    gamma: float = 0.95,
    lr: float = 1e-3,
    batch_size: int = 64,
    target_update: int = 20,
    log_root: str | Path = "runs",
    algo: str = "dueling",  # "baseline", "dueling", "dueling_curiosity", "lstm"
    env_seed: Optional[int] = 0,
    curiosity_beta: float = 0.1,
    obs_radius: Optional[int] = None,  # None = full map, else local window (e.g., 5)
    sequence_length: int = 4,  # For LSTM models
    preview_envs: bool = True,  # Générer et sauvegarder les environnements avant l'entraînement
    size_invariant_baseline: bool = False,  # Si True (baseline): Global Average Pooling → testable sur toute taille de grille
) -> None:
    log_dir = Path(log_root) / time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    metrics_path = log_dir / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("episode,ep_reward,steps,reached_goal,buffer_size\n")

    # Determine observation size (local window or full map)
    obs_size = obs_radius * 2 + 1 if obs_radius is not None else grid_size

    # Optionnel : pré-générer et sauvegarder les environnements d'entraînement
    if preview_envs:
        preview_dir = log_dir / "train_envs"
        tmp_env = GridWorld(
            size=grid_size,
            obstacle_ratio=obstacle_ratio,
            max_steps=max_steps,
            seed=env_seed,
            obs_radius=obs_radius,
        )
        for ep in range(episodes):
            tmp_env.reset()
            plot_env_only(tmp_env.grid, ep, preview_dir)

    # Environnement réel utilisé pour l'entraînement
    env = GridWorld(
        size=grid_size,
        obstacle_ratio=obstacle_ratio,
        max_steps=max_steps,
        seed=env_seed,
        obs_radius=obs_radius,
    )

    # Select architecture and replay buffer according to algo
    algo = algo.lower()
    use_lstm = False
    if algo == "baseline":
        policy: nn.Module = CnnDQN(obs_size, num_actions=8, size_invariant=size_invariant_baseline).to(Device)
        target: nn.Module = CnnDQN(obs_size, num_actions=8, size_invariant=size_invariant_baseline).to(Device)
        buffer: object = ReplayBuffer(capacity=100_000)
        use_prioritized = False
        use_curiosity = False
    elif algo in {"dueling", "dueling_curiosity"}:
        policy = DuelingCnnDQN(obs_size, num_actions=8).to(Device)
        target = DuelingCnnDQN(obs_size, num_actions=8).to(Device)
        buffer = PrioritizedReplayBuffer(capacity=100_000, alpha=0.6, beta=0.4)
        use_prioritized = True
        use_curiosity = algo == "dueling_curiosity"
    elif algo == "lstm":
        policy = DuelingCnnLstmDQN(obs_size, num_actions=8, sequence_length=sequence_length).to(Device)
        target = DuelingCnnLstmDQN(obs_size, num_actions=8, sequence_length=sequence_length).to(Device)
        buffer = SequenceReplayBuffer(capacity=100_000, sequence_length=sequence_length)
        use_prioritized = False
        use_curiosity = False
        use_lstm = True
    else:
        raise ValueError(f"Unknown algo '{algo}', expected 'baseline', 'dueling', 'dueling_curiosity', or 'lstm'.")

    target.load_state_dict(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Curiosity forward model (only for dueling_curiosity)
    forward_model: Optional[ForwardModelCNN] = None
    fm_optimizer: Optional[optim.Optimizer] = None
    if use_curiosity:
        forward_model = ForwardModelCNN().to(Device)
        fm_optimizer = optim.Adam(forward_model.parameters(), lr=1e-3)

    global_step = 0
    reward_hist: List[float] = []
    steps_hist: List[int] = []
    goal_hist: List[bool] = []
    start_time = time.time()

    # Barre de progression sur les épisodes d'entraînement
    for ep in tqdm(range(episodes), desc=f"Training ({algo}, grid={grid_size})"):
        state_np = env.reset()
        # state_np is (H, W), need (B, C, H, W) = (1, 1, H, W)
        state = torch.from_numpy(state_np).unsqueeze(0).unsqueeze(0).to(Device)
        ep_reward = 0.0
        path: List[Tuple[int, int]] = [env.agent]
        
        # For LSTM: maintain a sequence buffer
        state_sequence: List[torch.Tensor] = []
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        for t in range(max_steps):
            if use_lstm:
                # Build state sequence
                # state is (1, 1, H, W), we want to store (1, H, W) in sequence
                state_frame = state.squeeze(0)  # Remove batch dim -> (1, H, W)
                if len(state_sequence) < sequence_length:
                    state_sequence.append(state_frame)
                    # Pad with first state if needed
                    while len(state_sequence) < sequence_length:
                        state_sequence.insert(0, state_sequence[0])
                else:
                    state_sequence.pop(0)
                    state_sequence.append(state_frame)
                
                seq_tensor = torch.stack(state_sequence, dim=0)  # (T, 1, H, W)
                action, lstm_hidden = select_action_lstm(
                    policy, seq_tensor, lstm_hidden, global_step, eps_start, eps_end, eps_decay
                )
            else:
                action = select_action(policy, state, global_step, eps_start, eps_end, eps_decay)
            
            next_state_np, env_reward, done, info = env.step(action)
            # next_state_np is (H, W), need (B, C, H, W) = (1, 1, H, W)
            next_state = torch.from_numpy(next_state_np).unsqueeze(0).unsqueeze(0).to(Device)

            # Curiosity-driven intrinsic reward (optional)
            intrinsic_reward = 0.0
            if use_curiosity and forward_model is not None and fm_optimizer is not None:
                # Build 2-channel input: state + action-channel
                # state is (1, 1, H, W), we need (1, 2, H, W) for forward model
                # action is normalisé dans [0,1] avec 8 actions possibles
                action_channel = torch.full_like(state, float(action) / 7.0)
                fm_input = torch.cat([state, action_channel], dim=1)  # (1, 2, H, W)
                pred_next = forward_model(fm_input)
                curiosity_loss = (pred_next - next_state).pow(2).mean()
                fm_optimizer.zero_grad()
                curiosity_loss.backward()
                fm_optimizer.step()
                intrinsic_reward = curiosity_beta * float(curiosity_loss.detach().item())

            reward = env_reward + intrinsic_reward

            if use_lstm:
                # Build next state sequence
                next_state_seq = state_sequence.copy()
                next_state_frame = next_state.squeeze(0)  # (1, H, W)
                if len(next_state_seq) >= sequence_length:
                    next_state_seq.pop(0)
                next_state_seq.append(next_state_frame)
                next_seq_tensor = torch.stack(next_state_seq, dim=0)  # (T, 1, H, W)
                buffer.push(seq_tensor, action, reward, next_seq_tensor, done)
            else:
                buffer.push(state, action, reward, next_state, done)

            if use_lstm:
                loss = optimize_model_lstm(policy, target, buffer, optimizer, gamma, batch_size)
            elif use_prioritized:
                loss, _ = optimize_model_prioritized(policy, target, buffer, optimizer, gamma, batch_size)
            else:
                loss = optimize_model(policy, target, buffer, optimizer, gamma, batch_size)

            global_step += 1
            state = next_state
            ep_reward += reward
            path.append(env.agent)

            if global_step % target_update == 0:
                target.load_state_dict(policy.state_dict())

            if done:
                break

        reward_hist.append(ep_reward)
        reached_goal = info.get("reached_goal", False)
        steps_hist.append(len(path))
        goal_hist.append(reached_goal)

        # Smooth and plot
        smoothed = smooth_path(path) if reached_goal else None
        plot_episode(env.grid, path, smoothed, ep, ep_reward, reached_goal, log_dir)

        # Sauvegarde détaillée de l'épisode (grille + trajectoire)
        save_episode_data(log_dir, ep, env.grid, path, ep_reward, reached_goal)

        # Persist episode metrics
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"{ep+1},{ep_reward:.4f},{len(path)},{int(reached_goal)},{len(buffer)}\n")

        if (ep + 1) % 20 == 0:
            avg_r = np.mean(reward_hist[-20:])
            print(f"[Episode {ep+1:4d}] avg_reward={avg_r:6.2f} buffer={len(buffer):6d}")

        if reached_goal:
            print(f"  Reached goal in {len(path)} steps; smoothed_len={len(smoothed) if smoothed is not None else 0}.")

    # ------------------------------------------------------------------ #
    # Global curves (similar in esprit to the figures du papier)
    # ------------------------------------------------------------------ #
    episodes_axis = np.arange(1, len(reward_hist) + 1)

    # Moving average helper
    def moving_avg(x: np.ndarray, k: int = 5) -> np.ndarray:
        if len(x) < k:
            return x
        kernel = np.ones(k) / k
        return np.convolve(x, kernel, mode="valid")

    # Sauvegarder le modèle entraîné
    checkpoint_path = log_dir / "checkpoint.pth"
    torch.save(policy.state_dict(), checkpoint_path)
    print(f"✓ Modèle sauvegardé: {checkpoint_path}")
    
    # Reward per episode + moyenne glissante
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(episodes_axis, reward_hist, label="reward par épisode", alpha=0.6)
    ma_r = moving_avg(np.array(reward_hist, dtype=float), k=min(5, len(reward_hist)))
    ax.plot(np.arange(1, len(ma_r) + 1), ma_r, label="moyenne glissante", linewidth=2)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense totale")
    ax.set_title("Récompense par épisode")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(log_dir / "reward_curve.png", dpi=150)
    plt.close(fig)

    # Longueur de chemin par épisode
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(episodes_axis, steps_hist, label="longueur de chemin", color="tab:green")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Nombre de pas")
    ax.set_title("Longueur de chemin par épisode")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(log_dir / "path_length_curve.png", dpi=150)
    plt.close(fig)

    # Taux de réussite cumulatif (optionnel)
    successes = np.cumsum(np.array(goal_hist, dtype=int))
    success_rate = successes / episodes_axis
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(episodes_axis, success_rate, label="taux de réussite cumulatif")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Taux de réussite")
    ax.set_ylim(0, 1.05)
    ax.set_title("Taux de réussite au fil des épisodes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(log_dir / "success_rate_curve.png", dpi=150)
    plt.close(fig)

    dur = time.time() - start_time
    print(f"Training done in {dur/60:.1f} min. Mean reward (last 50): {np.mean(reward_hist[-50:]):.2f}")
    
    # Sauvegarder le modèle entraîné
    checkpoint_path = log_dir / "checkpoint.pth"
    torch.save(policy.state_dict(), checkpoint_path)
    print(f"✓ Modèle sauvegardé: {checkpoint_path}")
    
    return log_dir


def test_model(
    policy: nn.Module,
    env: GridWorld,
    max_steps: int = 200,
    visualize: bool = True,
    save_path: Optional[Path] = None,
) -> Tuple[List[Tuple[int, int]], bool, float]:
    """Teste un modèle sur un environnement et retourne la trajectoire."""
    state_np = env.reset()
    # state_np is (H, W), need (B, C, H, W) = (1, 1, H, W)
    state = torch.from_numpy(state_np).unsqueeze(0).unsqueeze(0).to(Device)
    path: List[Tuple[int, int]] = [env.agent]
    total_reward = 0.0
    
    # Pour LSTM: maintenir une séquence
    state_sequence: List[torch.Tensor] = []
    lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    use_lstm = isinstance(policy, DuelingCnnLstmDQN)
    sequence_length = policy.sequence_length if use_lstm else 4
    
    for t in range(max_steps):
        if use_lstm:
            # Construire la séquence
            state_frame = state.squeeze(0)  # (1, H, W)
            if len(state_sequence) < sequence_length:
                state_sequence.append(state_frame)
                while len(state_sequence) < sequence_length:
                    state_sequence.insert(0, state_sequence[0])
            else:
                state_sequence.pop(0)
                state_sequence.append(state_frame)
            
            seq_tensor = torch.stack(state_sequence, dim=0)  # (T, 1, H, W)
            with torch.no_grad():
                q_values, lstm_hidden = policy(seq_tensor.unsqueeze(0), lstm_hidden)
                if q_values.dim() == 3:
                    q_values = q_values[:, -1, :]
                action = int(torch.argmax(q_values, dim=1).item())
        else:
            with torch.no_grad():
                q_values = policy(state)
                action = int(torch.argmax(q_values, dim=1).item())
        
        next_state_np, reward, done, info = env.step(action)
        # next_state_np is (H, W), need (B, C, H, W) = (1, 1, H, W)
        next_state = torch.from_numpy(next_state_np).unsqueeze(0).unsqueeze(0).to(Device)
        
        total_reward += reward
        path.append(env.agent)
        state = next_state
        
        if done:
            break
    
    reached_goal = info.get("reached_goal", False)
    
    # Visualisation si demandée
    if visualize and save_path is not None:
        smoothed = smooth_path(path) if reached_goal else None
        plot_episode(env.grid, path, smoothed, 0, total_reward, reached_goal, save_path.parent)
    
    return path, reached_goal, total_reward


if __name__ == "__main__":
    train()

