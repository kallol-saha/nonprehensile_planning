"""Vision-conditioned Diffusion Policy with 1D temporal U-Net.

Architecture follows Chi et al. (2023) "Diffusion Policy":
  - Visual encoder → global scene embedding
  - 1D U-Net denoises SE(2) trajectory conditioned on scene + diffusion timestep
  - FiLM conditioning: scene embedding modulates each residual block
"""

import math

import torch
import torch.nn as nn

from visplan.training.vision_encoder import VisionEncoder
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


class FiLMBlock(nn.Module):
    """Residual 1D conv block with FiLM conditioning."""

    def __init__(self, channels, cond_dim, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
        )
        self.film = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        """
        Args:
            x:    (B, C, T)
            cond: (B, cond_dim)
        """
        h = self.net(x)
        gamma, beta = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        h = gamma * h + beta
        return x + h


class Downsample1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# --------------------------------------------------------------------------- #
#  1D U-Net noise predictor
# --------------------------------------------------------------------------- #

class TemporalUNet(nn.Module):
    """1D U-Net for trajectory denoising.

    Input:  noisy trajectory (B, T, traj_dim) + diffusion timestep + scene embedding
    Output: predicted noise   (B, T, traj_dim)
    """

    def __init__(self, traj_dim: int = 4, cond_dim: int = 256,
                 base_channels: int = 128, channel_mults=(1, 2, 4),
                 num_res_blocks: int = 2, time_dim: int = 128):
        super().__init__()
        self.traj_dim = traj_dim
        total_cond_dim = cond_dim + time_dim

        # Diffusion timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Input projection
        self.input_proj = nn.Conv1d(traj_dim, base_channels, 1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [base_channels]
        ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(FiLMBlock(ch, total_cond_dim))
                if ch != out_ch:
                    blocks.append(nn.Conv1d(ch, out_ch, 1))
                    ch = out_ch
            self.down_blocks.append(blocks)
            channels.append(ch)
            self.downsamples.append(Downsample1d(ch))

        # Bottleneck
        self.mid_block1 = FiLMBlock(ch, total_cond_dim)
        self.mid_block2 = FiLMBlock(ch, total_cond_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.upsamples.append(Upsample1d(ch))
            skip_ch = channels.pop()
            blocks = nn.ModuleList()
            blocks.append(nn.Conv1d(ch + skip_ch, out_ch, 1))
            ch = out_ch
            for _ in range(num_res_blocks):
                blocks.append(FiLMBlock(ch, total_cond_dim))
            self.up_blocks.append(blocks)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv1d(ch, traj_dim, 1),
        )

    def forward(self, x_t, t, cond):
        """
        Args:
            x_t:  (B, T_traj, traj_dim) noisy trajectory
            t:    (B,) diffusion timestep
            cond: (B, cond_dim) scene embedding

        Returns:
            (B, T_traj, traj_dim) predicted noise
        """
        # Prepare conditioning
        t_emb = self.time_mlp(t)               # (B, time_dim)
        cond = torch.cat([cond, t_emb], dim=-1)  # (B, cond_dim + time_dim)

        # (B, T_traj, D) → (B, D, T_traj) for 1D convs
        x = x_t.permute(0, 2, 1)
        x = self.input_proj(x)

        # Encoder
        skips = [x]
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for layer in blocks:
                if isinstance(layer, FiLMBlock):
                    x = layer(x, cond)
                else:
                    x = layer(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        # Decoder
        for blocks, up in zip(self.up_blocks, self.upsamples):
            x = up(x)
            skip = skips.pop()
            # Match temporal dimension after up/downsample rounding
            if x.shape[-1] != skip.shape[-1]:
                x = x[..., :skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            for layer in blocks:
                if isinstance(layer, FiLMBlock):
                    x = layer(x, cond)
                else:
                    x = layer(x)

        x = self.output_proj(x)
        return x.permute(0, 2, 1)  # (B, T_traj, traj_dim)


# --------------------------------------------------------------------------- #
#  Full model: vision encoder + diffusion + U-Net
# --------------------------------------------------------------------------- #

class DiffusionUNet(nn.Module):
    """Complete vision-conditioned diffusion policy with 1D U-Net backbone.

    Usage:
        model = DiffusionUNet()
        loss = model.compute_loss(batch)           # training
        traj = model.sample(batch, device="cuda")  # inference
    """

    def __init__(self, traj_dim: int = 4, traj_len: int = 32,
                 embed_dim: int = 256, num_diffusion_steps: int = 100,
                 beta_schedule: str = "cosine", **unet_kwargs):
        super().__init__()
        self.traj_dim = traj_dim
        self.traj_len = traj_len

        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        self.noise_pred_net = TemporalUNet(
            traj_dim=traj_dim, cond_dim=embed_dim, **unet_kwargs)
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_diffusion_steps, beta_schedule=beta_schedule)

    def _encode_scene(self, batch):
        return self.vision_encoder(
            batch["start_image"], batch["goal_image"], batch["piece_mask"])

    def compute_loss(self, batch):
        cond = self._encode_scene(batch)
        traj = batch["trajectory"]  # (B, T, 4)
        return self.diffusion.training_loss(self.noise_pred_net, traj, cond)

    @torch.no_grad()
    def sample(self, batch, device="cuda", use_ddim=False, ddim_steps=20):
        cond = self._encode_scene(batch)
        B = cond.shape[0]
        shape = (B, self.traj_len, self.traj_dim)
        if use_ddim:
            return self.diffusion.ddim_sample(
                self.noise_pred_net, shape, cond, device=device,
                num_steps=ddim_steps)
        return self.diffusion.ddpm_sample(
            self.noise_pred_net, shape, cond, device=device)
