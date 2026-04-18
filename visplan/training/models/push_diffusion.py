"""Geometry-conditioned diffusion model for push trajectories.

Conditions on polygon shape (PointNet) + start/goal SE(2) poses.
Reuses the existing TemporalUNet backbone and GaussianDiffusion scheduler.

Supports classifier-free guidance on the goal pose: during training the
goal embedding is randomly dropped (replaced with zeros) so the model
can generate trajectories with or without a specified goal.
"""

import torch
import torch.nn as nn

from visplan.training.polygon_encoder import PolygonEncoder
from visplan.training.models.diffusion_unet import TemporalUNet
from visplan.training.diffusion import GaussianDiffusion


class ConditionEncoder(nn.Module):
    """Fuse polygon geometry + start pose + goal pose into a single embedding.

    Args:
        embed_dim:  dimension of the final conditioning vector (fed to UNet).
        pose_dim:   dimension of encoded poses (4 = x̃, ỹ, cos θ, sin θ).
        hidden_dim: PointNet hidden width.
    """

    def __init__(self, embed_dim: int = 256, pose_dim: int = 4,
                 hidden_dim: int = 128):
        super().__init__()
        self.polygon_encoder = PolygonEncoder(
            embed_dim=embed_dim, hidden_dim=hidden_dim)

        self.start_mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.goal_mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Fuse three embeddings → single cond vector
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, polygon_vertices, polygon_mask, start_pose, goal_pose):
        """
        Args:
            polygon_vertices: (B, V_max, 2)
            polygon_mask:     (B, V_max)
            start_pose:       (B, 4)
            goal_pose:        (B, 4)

        Returns:
            (B, embed_dim) conditioning embedding.
        """
        poly_emb = self.polygon_encoder(polygon_vertices, polygon_mask)
        start_emb = self.start_mlp(start_pose)
        goal_emb = self.goal_mlp(goal_pose)
        return self.fuse(torch.cat([poly_emb, start_emb, goal_emb], dim=-1))


class PushDiffusionUNet(nn.Module):
    """Geometry-conditioned diffusion policy for push trajectories.

    Usage:
        model = PushDiffusionUNet()
        loss = model.compute_loss(batch)
        traj = model.sample(batch, device="cuda")
    """

    def __init__(
        self,
        traj_dim: int = 4,
        traj_len: int = 151,
        embed_dim: int = 256,
        num_diffusion_steps: int = 100,
        beta_schedule: str = "cosine",
        goal_dropout: float = 0.1,
        **unet_kwargs,
    ):
        super().__init__()
        self.traj_dim = traj_dim
        self.traj_len = traj_len
        self.goal_dropout = goal_dropout

        self.cond_encoder = ConditionEncoder(embed_dim=embed_dim)
        self.noise_pred_net = TemporalUNet(
            traj_dim=traj_dim, cond_dim=embed_dim, **unet_kwargs)
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_diffusion_steps, beta_schedule=beta_schedule)

    def _encode_condition(self, batch, drop_goal: bool = False):
        """Build conditioning vector, optionally dropping goal for CFG."""
        goal = batch["goal_pose"]
        if drop_goal:
            goal = torch.zeros_like(goal)
        return self.cond_encoder(
            batch["polygon_vertices"], batch["polygon_mask"],
            batch["start_pose"], goal,
        )

    def compute_loss(self, batch):
        """Compute masked ε-prediction loss with goal dropout.

        Args:
            batch: dict from PushTrajectoryDataset.

        Returns:
            scalar loss.
        """
        # Classifier-free guidance: randomly drop goal
        drop_goal = self.training and (torch.rand(1).item() < self.goal_dropout)
        cond = self._encode_condition(batch, drop_goal=drop_goal)

        traj = batch["trajectory"]       # (B, T_max, 4)
        mask = batch["trajectory_mask"]  # (B, T_max)

        # Forward diffusion
        B = traj.shape[0]
        t = torch.randint(0, self.diffusion.T, (B,), device=traj.device)
        x_t, noise = self.diffusion.q_sample(traj, t)
        pred_noise = self.noise_pred_net(x_t, t, cond)

        # Masked MSE — only compute loss on valid timesteps
        err = (pred_noise - noise) ** 2                  # (B, T_max, 4)
        err = err.mean(dim=-1)                           # (B, T_max)
        err = (err * mask).sum() / mask.sum().clamp(min=1)
        return err

    @torch.no_grad()
    def sample(self, batch, device="cuda", use_ddim=False, ddim_steps=20,
               cfg_scale: float = 0.0):
        """Generate trajectories conditioned on polygon + start + goal.

        Args:
            batch:     dict with polygon_vertices, polygon_mask, start_pose,
                       goal_pose (each with batch dimension).
            cfg_scale: classifier-free guidance weight.  0 = no guidance,
                       >0 blends unconditional and conditional predictions.
            use_ddim:  use DDIM sampling (faster).
            ddim_steps: DDIM denoising steps.

        Returns:
            (B, T_max, 4) denoised trajectories.
        """
        cond = self._encode_condition(batch, drop_goal=False)
        B = cond.shape[0]
        shape = (B, self.traj_len, self.traj_dim)

        if cfg_scale > 0:
            cond_uncond = self._encode_condition(batch, drop_goal=True)
            return self._cfg_sample(
                shape, cond, cond_uncond, cfg_scale, device,
                use_ddim=use_ddim, ddim_steps=ddim_steps)

        if use_ddim:
            return self.diffusion.ddim_sample(
                self.noise_pred_net, shape, cond,
                device=device, num_steps=ddim_steps)
        return self.diffusion.ddpm_sample(
            self.noise_pred_net, shape, cond, device=device)

    def _cfg_sample(self, shape, cond, cond_uncond, cfg_scale, device,
                    use_ddim=False, ddim_steps=20):
        """DDPM/DDIM sampling with classifier-free guidance."""
        diff = self.diffusion
        x = torch.randn(shape, device=device)

        if use_ddim:
            step_size = diff.T // ddim_steps
            timesteps = list(range(0, diff.T, step_size))[::-1]
        else:
            timesteps = list(range(diff.T - 1, -1, -1))

        for i, t_cur in enumerate(timesteps):
            t = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)

            # Conditional + unconditional predictions
            noise_cond = self.noise_pred_net(x, t, cond)
            noise_uncond = self.noise_pred_net(x, t, cond_uncond)
            pred_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            if use_ddim:
                alpha_bar_t = diff.alpha_bar[t_cur]
                if i + 1 < len(timesteps):
                    alpha_bar_prev = diff.alpha_bar[timesteps[i + 1]]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device)
                x0_pred = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
                dir_xt = (1 - alpha_bar_prev).sqrt() * pred_noise
                x = alpha_bar_prev.sqrt() * x0_pred + dir_xt
            else:
                alpha = diff.alphas[t_cur]
                alpha_bar = diff.alpha_bar[t_cur]
                beta = diff.betas[t_cur]
                mean = (1.0 / alpha.sqrt()) * (
                    x - (beta / (1.0 - alpha_bar).sqrt()) * pred_noise)
                if t_cur > 0:
                    x = mean + beta.sqrt() * torch.randn_like(x)
                else:
                    x = mean

        return x
