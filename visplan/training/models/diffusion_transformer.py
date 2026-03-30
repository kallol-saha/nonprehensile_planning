"""Vision-conditioned Diffusion Transformer (DiT) for trajectory prediction.

Architecture:
  - Spatial vision encoder → image tokens for cross-attention
  - Transformer denoiser with:
    - Learned positional encoding for trajectory timesteps
    - AdaLN-Zero conditioning on diffusion timestep (Peebles & Xie 2023)
    - Cross-attention to spatial image features
"""

import math

import torch
import torch.nn as nn

from visplan.training.vision_encoder import SpatialVisionEncoder
from visplan.training.diffusion import GaussianDiffusion


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AdaLNZero(nn.Module):
    """Adaptive Layer Norm with zero-initialised gating (DiT-style)."""

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 3),  # γ, β, gate
        )
        # Zero-initialise the gate so the block starts as identity
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, cond):
        """
        Args:
            x:    (B, L, D)
            cond: (B, cond_dim)
        Returns:
            normed_x, gate — both (B, L, D)
        """
        gamma, beta, gate = self.proj(cond).unsqueeze(1).chunk(3, dim=-1)
        return (1 + gamma) * self.norm(x) + beta, gate


class TransformerBlock(nn.Module):
    """Transformer block with AdaLN-Zero + cross-attention to image tokens."""

    def __init__(self, hidden_dim: int, cond_dim: int, num_heads: int = 8,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.adaln_self = AdaLNZero(hidden_dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                               batch_first=True)

        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                                batch_first=True)
        self.kv_proj = nn.Linear(hidden_dim, hidden_dim)  # project image tokens

        self.adaln_ff = AdaLNZero(hidden_dim, cond_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

    def forward(self, x, cond, img_tokens):
        """
        Args:
            x:          (B, L, D)   trajectory tokens
            cond:       (B, cond_dim)  diffusion time + global scene
            img_tokens: (B, S, D)   spatial image features
        """
        # Self-attention with AdaLN-Zero
        h, gate = self.adaln_self(x, cond)
        h = self.self_attn(h, h, h, need_weights=False)[0]
        x = x + gate * h

        # Cross-attention to image tokens
        q = self.cross_norm(x)
        kv = self.kv_proj(img_tokens)
        h = self.cross_attn(q, kv, kv, need_weights=False)[0]
        x = x + h

        # Feed-forward with AdaLN-Zero
        h, gate = self.adaln_ff(x, cond)
        h = self.ff(h)
        x = x + gate * h

        return x


# --------------------------------------------------------------------------- #
#  Trajectory Diffusion Transformer
# --------------------------------------------------------------------------- #

class TrajectoryDiT(nn.Module):
    """Transformer denoiser for SE(2) trajectories.

    Input:  noisy trajectory (B, T, traj_dim) + diffusion timestep + image features
    Output: predicted noise   (B, T, traj_dim)
    """

    def __init__(self, traj_dim: int = 4, traj_len: int = 32,
                 hidden_dim: int = 256, num_layers: int = 6,
                 num_heads: int = 8, scene_dim: int = 256,
                 time_dim: int = 128):
        super().__init__()
        self.traj_dim = traj_dim
        cond_dim = scene_dim + time_dim

        # Diffusion timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Trajectory token embedding
        self.input_proj = nn.Linear(traj_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, traj_len, hidden_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, cond_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, traj_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t, t, condition):
        """
        Args:
            x_t:       (B, T, traj_dim) noisy trajectory
            t:         (B,) diffusion timestep
            condition: tuple (global_emb, spatial_tokens)
                       global_emb:     (B, scene_dim)
                       spatial_tokens: (B, S, hidden_dim)

        Returns:
            (B, T, traj_dim) predicted noise
        """
        global_emb, img_tokens = condition

        # Condition = concat(scene_global, time)
        t_emb = self.time_mlp(t)                          # (B, time_dim)
        cond = torch.cat([global_emb, t_emb], dim=-1)     # (B, cond_dim)

        # Trajectory tokens
        x = self.input_proj(x_t) + self.pos_embed[:, :x_t.shape[1]]

        for block in self.blocks:
            x = block(x, cond, img_tokens)

        x = self.final_norm(x)
        return self.output_proj(x)


# --------------------------------------------------------------------------- #
#  Full model
# --------------------------------------------------------------------------- #

class DiffusionTransformer(nn.Module):
    """Complete vision-conditioned DiT for trajectory prediction.

    Usage:
        model = DiffusionTransformer()
        loss = model.compute_loss(batch)
        traj = model.sample(batch, device="cuda")
    """

    def __init__(self, traj_dim: int = 4, traj_len: int = 32,
                 embed_dim: int = 256, num_diffusion_steps: int = 100,
                 beta_schedule: str = "cosine", num_layers: int = 6,
                 num_heads: int = 8):
        super().__init__()
        self.traj_dim = traj_dim
        self.traj_len = traj_len

        self.vision_encoder = SpatialVisionEncoder(embed_dim=embed_dim)
        self.noise_pred_net = TrajectoryDiT(
            traj_dim=traj_dim, traj_len=traj_len,
            hidden_dim=embed_dim, num_layers=num_layers,
            num_heads=num_heads, scene_dim=embed_dim)
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_diffusion_steps, beta_schedule=beta_schedule)

    def _encode_scene(self, batch):
        glob, spatial = self.vision_encoder(
            batch["start_image"], batch["goal_image"], batch["piece_mask"])
        return (glob, spatial)

    def compute_loss(self, batch):
        cond = self._encode_scene(batch)
        traj = batch["trajectory"]
        return self.diffusion.training_loss(self.noise_pred_net, traj, cond)

    @torch.no_grad()
    def sample(self, batch, device="cuda", use_ddim=False, ddim_steps=20):
        cond = self._encode_scene(batch)
        B = cond[0].shape[0]
        shape = (B, self.traj_len, self.traj_dim)
        if use_ddim:
            return self.diffusion.ddim_sample(
                self.noise_pred_net, shape, cond, device=device,
                num_steps=ddim_steps)
        return self.diffusion.ddpm_sample(
            self.noise_pred_net, shape, cond, device=device)
