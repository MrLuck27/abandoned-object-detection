"""Optional Deep Temporal Network (LSTM / Temporal MLP) for sequence reasoning."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class TemporalLSTMClassifier(nn.Module):
    """LSTM-based sequence classifier for unattended object verification.

    Takes a temporal sequence of features (e.g., sliding window of 30 frames)
    and predicts binary probability of abandonment: [0: Not Abandoned, 1: Abandoned].
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Logits of shape (batch_size, 2)
        """
        lstm_out, _ = self.lstm(x)
        # Use representation from last temporal step
        last_step = lstm_out[:, -1, :]
        logits = self.fc(last_step)
        return logits


class TemporalMLPClassifier(nn.Module):
    """Lightweight 1D Temporal MLP baseline operating on pooled window features."""

    def __init__(self, input_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, 2)
        """
        return self.net(x)


def extract_temporal_feature_vector(
    bag_center: np.ndarray,
    bag_velocity: np.ndarray,
    owner_center: Optional[np.ndarray],
    owner_distance: float,
    owner_relative_speed: float,
    bag_displacement: float,
    owner_speed: float,
    separation_duration: float,
    bag_confidence: float,
    frame_width: float = 1920.0,
    frame_height: float = 1080.0,
) -> np.ndarray:
    """Extract a 12-dimensional normalized feature vector at a single time step."""
    bx_norm = float(bag_center[0] / max(frame_width, 1.0))
    by_norm = float(bag_center[1] / max(frame_height, 1.0))
    bvx_norm = float(bag_velocity[0] / 100.0)
    bvy_norm = float(bag_velocity[1] / 100.0)

    if owner_center is not None:
        ox_norm = float(owner_center[0] / max(frame_width, 1.0))
        oy_norm = float(owner_center[1] / max(frame_height, 1.0))
    else:
        ox_norm = -1.0
        oy_norm = -1.0

    odist_norm = float(min(owner_distance / 1000.0, 2.0))
    ospeed_norm = float(np.clip(owner_relative_speed / 50.0, -2.0, 2.0))
    bdisp_norm = float(min(bag_displacement / 100.0, 2.0))
    own_speed_norm = float(min(owner_speed / 100.0, 2.0))
    sep_dur_norm = float(min(separation_duration / 60.0, 2.0))
    conf = float(bag_confidence)

    features = np.array(
        [
            bx_norm,
            by_norm,
            bvx_norm,
            bvy_norm,
            ox_norm,
            oy_norm,
            odist_norm,
            ospeed_norm,
            bdisp_norm,
            own_speed_norm,
            sep_dur_norm,
            conf,
        ],
        dtype=np.float32,
    )
    return features
