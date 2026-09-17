"""
Neural Network Architectures for Mobile Robot Path Planning.
Includes:
- CnnDQN: Standard CNN with global or local occupancy grid input.
- DuelingCnnDQN: Dueling architecture with separate Value and Advantage streams.
- DuelingCnnLstmDQN: Proposed spatiotemporal architecture combining CNN feature extraction,
  LSTM temporal modeling, and Dueling Q-heads.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class CnnDQN(nn.Module):
    """Convolutional DQN for 2D occupancy grid path planning.
    
    Args:
        grid_size: Input spatial dimension (e.g. 15 for global, 11 for local).
        num_actions: Number of discrete navigation actions (default: 8).
        size_invariant: If True, uses Global Average Pooling to accept arbitrary input dimensions.
    """

    def __init__(self, grid_size: int = 15, num_actions: int = 8, size_invariant: bool = False) -> None:
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
    """Dueling CNN-DQN architecture decoupling state value V(s) and action advantage A(s, a).
    
    Args:
        grid_size: Input spatial dimension (e.g. 11 for local window).
        num_actions: Number of discrete navigation actions (default: 8).
        size_invariant: If True, uses Global Average Pooling.
    """

    def __init__(self, grid_size: int = 11, num_actions: int = 8, size_invariant: bool = False) -> None:
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


class DuelingCnnLstmDQN(nn.Module):
    """Spatiotemporal Dueling CNN-LSTM-DQN architecture.
    
    Passes each frame through a shared CNN trunk, models temporal state transitions
    via an LSTM layer, and resolves Q-values through dueling value and advantage streams.

    Args:
        grid_size: Input spatial dimension (e.g. 11 for local window).
        num_actions: Number of discrete navigation actions (default: 8).
        lstm_hidden: Number of hidden units in LSTM (default: 128).
        sequence_length: Temporal sequence window length (default: 4).
    """

    def __init__(
        self,
        grid_size: int = 11,
        num_actions: int = 8,
        lstm_hidden: int = 128,
        sequence_length: int = 4,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.lstm_hidden = lstm_hidden
        
        # Spatial feature extractor (shared across temporal sequence)
        self.cnn_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_feat_dim = grid_size * grid_size * 32
        
        # Temporal recurrence
        self.lstm = nn.LSTM(cnn_feat_dim, lstm_hidden, batch_first=True)
        
        # Dueling decision streams
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

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, T, 1, H, W) or (B, 1, H, W) for single frame.
            hidden: Optional initial LSTM hidden state (h_0, c_0).
        Returns:
            q_values: Q-values tensor of shape (B, num_actions) or (B, T, num_actions).
            hidden: Updated LSTM hidden state (h_t, c_t).
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, 1, H, W)
        
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        
        cnn_feat = self.cnn_features(x_flat)
        cnn_feat = cnn_feat.view(B, T, -1)
        
        lstm_out, hidden = self.lstm(cnn_feat, hidden)
        
        if T == 1:
            lstm_last = lstm_out[:, -1, :]
            value = self.value_stream(lstm_last)
            adv = self.adv_stream(lstm_last)
            adv_mean = adv.mean(dim=1, keepdim=True)
            q = value + adv - adv_mean
            return q, hidden
        else:
            value = self.value_stream(lstm_out)
            adv = self.adv_stream(lstm_out)
            adv_mean = adv.mean(dim=2, keepdim=True)
            q = value + adv - adv_mean
            return q, hidden
