"""PointNet-style encoder for convex polygon vertices.

Takes padded (V_max, 2) vertices + validity mask and produces a
fixed-dimensional embedding that captures the polygon's geometry.
"""

import torch
import torch.nn as nn


class PolygonEncoder(nn.Module):
    """PointNet encoder: shared MLP → masked max-pool → projection.

    Args:
        embed_dim:   output embedding dimension.
        hidden_dim:  width of shared MLP layers.
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, vertices, mask):
        """
        Args:
            vertices: (B, V_max, 2) padded polygon vertices.
            mask:     (B, V_max) float — 1 for real vertices, 0 for padding.

        Returns:
            (B, embed_dim) polygon embedding.
        """
        h = self.shared_mlp(vertices)              # (B, V_max, hidden_dim)
        # Mask out padding before max-pool
        h = h.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        h = h.max(dim=1).values                    # (B, hidden_dim)
        return self.proj(h)                         # (B, embed_dim)
