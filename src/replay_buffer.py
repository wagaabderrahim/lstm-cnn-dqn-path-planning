"""
Experience Replay Buffers for standard DQN and Sequence-based LSTM-DQN.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
import random
import numpy as np
import torch


@dataclass
class Transition:
    """Standard Transition tuple for frame-by-frame DQN."""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    """Standard FIFO Replay Buffer for DQN and Dueling DQN."""

    def __init__(self, capacity: int = 50_000, device: Optional[torch.device] = None) -> None:
        self.capacity = capacity
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        states = [b.state.squeeze(0) if b.state.dim() == 4 else b.state for b in batch]
        next_states = [b.next_state.squeeze(0) if b.next_state.dim() == 4 else b.next_state for b in batch]
        return Transition(
            state=torch.stack(states).to(self.device),
            action=torch.tensor([b.action for b in batch], device=self.device, dtype=torch.long),
            reward=torch.tensor([b.reward for b in batch], device=self.device, dtype=torch.float32),
            next_state=torch.stack(next_states).to(self.device),
            done=torch.tensor([b.done for b in batch], device=self.device, dtype=torch.bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class SequenceTransition:
    """Transition storing a sequence of state frames for recurrent (LSTM) processing."""
    state_sequence: torch.Tensor       # Shape: (T, 1, H, W)
    action: int
    reward: float
    next_state_sequence: torch.Tensor  # Shape: (T, 1, H, W)
    done: bool


class SequenceReplayBuffer:
    """Replay buffer storing rolling temporal sequences of length T for LSTM models."""

    def __init__(
        self,
        capacity: int = 50_000,
        sequence_length: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.buffer: Deque[SequenceTransition] = deque(maxlen=capacity)

    def push(
        self,
        state_seq: torch.Tensor,
        action: int,
        reward: float,
        next_state_seq: torch.Tensor,
        done: bool,
    ) -> None:
        self.buffer.append(
            SequenceTransition(
                state_sequence=state_seq.cpu(),
                action=action,
                reward=reward,
                next_state_sequence=next_state_seq.cpu(),
                done=done,
            )
        )

    def sample(self, batch_size: int) -> SequenceTransition:
        batch = random.sample(self.buffer, batch_size)
        return SequenceTransition(
            state_sequence=torch.stack([b.state_sequence for b in batch]).to(self.device),
            action=torch.tensor([b.action for b in batch], device=self.device, dtype=torch.long),
            reward=torch.tensor([b.reward for b in batch], device=self.device, dtype=torch.float32),
            next_state_sequence=torch.stack([b.next_state_sequence for b in batch]).to(self.device),
            done=torch.tensor([b.done for b in batch], device=self.device, dtype=torch.bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)
