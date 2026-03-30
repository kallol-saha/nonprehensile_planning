"""Deterministic regression baseline for trajectory prediction.

Simple architecture: vision encoder → MLP → trajectory (T, 4).
Trained with MSE loss. Serves as a lower bound for diffusion models.
"""

import torch
import torch.nn as nn

from visplan.training.vision_encoder import VisionEncoder


class RegressionBaseline(nn.Module):
    """Deterministic trajectory predictor.

    Usage:
        model = RegressionBaseline()
        loss = model.compute_loss(batch)
        traj = model.sample(batch)
    """

    def __init__(self, traj_dim: int = 4, traj_len: int = 32,
                 embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.traj_dim = traj_dim
        self.traj_len = traj_len

        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, traj_len * traj_dim),
        )

    def forward(self, batch):
        cond = self.vision_encoder(
            batch["start_image"], batch["goal_image"], batch["piece_mask"])
        out = self.head(cond)  # (B, T * D)
        return out.view(-1, self.traj_len, self.traj_dim)

    def compute_loss(self, batch):
        pred = self.forward(batch)
        target = batch["trajectory"]
        return nn.functional.mse_loss(pred, target)

    @torch.no_grad()
    def sample(self, batch, **kwargs):
        return self.forward(batch)
