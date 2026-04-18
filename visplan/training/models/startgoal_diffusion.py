"""Trajectory diffusion with start/goal supplied ONLY via endpoint clamping.

No conditioning network of any kind. The denoiser takes (x_t, t) and
the FiLM modulation is driven entirely by the diffusion-timestep
embedding. Start and goal enter the model only through inpainting of
x_t at index 0 and end_idx:

    x_t[:, 0]           = √ᾱ_t · start + √(1-ᾱ_t) · ε_fresh_s
    x_t[b, end_idx[b]]  = √ᾱ_t · goal  + √(1-ᾱ_t) · ε_fresh_g

Loss is computed only on the interior "free" timesteps.
"""

import torch
import torch.nn as nn

from visplan.training.models.diffusion_unet import TemporalUNet
from visplan.training.diffusion import GaussianDiffusion


class StartGoalDiffusionUNet(nn.Module):
    """Unconditional trajectory diffusion + endpoint inpainting."""

    def __init__(
        self,
        traj_dim: int = 4,
        traj_len: int = 151,
        num_diffusion_steps: int = 100,
        beta_schedule: str = "cosine",
        **unet_kwargs,
    ):
        super().__init__()
        self.traj_dim = traj_dim
        self.traj_len = traj_len

        # cond_dim=0 → TemporalUNet FiLM is driven by t_emb alone.
        self.noise_pred_net = TemporalUNet(
            traj_dim=traj_dim, cond_dim=0, **unet_kwargs)
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_diffusion_steps, beta_schedule=beta_schedule)

    def _clamp_endpoints(self, x_t, start, goal, t, end_idx):
        B = x_t.shape[0]
        sqrt_ab = self.diffusion.sqrt_alpha_bar[t].view(B, 1)
        sqrt_1_ab = self.diffusion.sqrt_one_minus_alpha_bar[t].view(B, 1)

        start_noised = sqrt_ab * start + sqrt_1_ab * torch.randn_like(start)
        goal_noised = sqrt_ab * goal + sqrt_1_ab * torch.randn_like(goal)

        x_t = x_t.clone()
        x_t[:, 0] = start_noised
        batch_idx = torch.arange(B, device=x_t.device)
        x_t[batch_idx, end_idx] = goal_noised
        return x_t

    def compute_loss(self, batch):
        traj = batch["trajectory"]        # (B, T_max, 4)
        mask = batch["trajectory_mask"]   # (B, T_max)
        B = traj.shape[0]
        end_idx = mask.sum(dim=-1).long() - 1

        t = torch.randint(0, self.diffusion.T, (B,), device=traj.device)
        x_t, noise = self.diffusion.q_sample(traj, t)
        x_t = self._clamp_endpoints(
            x_t, batch["start_pose"], batch["goal_pose"], t, end_idx)

        pred_noise = self.noise_pred_net(x_t, t)   # no cond

        err = ((pred_noise - noise) ** 2).mean(dim=-1)   # (B, T_max)
        loss_mask = mask.clone()
        loss_mask[:, 0] = 0
        batch_idx = torch.arange(B, device=traj.device)
        loss_mask[batch_idx, end_idx] = 0
        return (err * loss_mask).sum() / loss_mask.sum().clamp(min=1)

    @torch.no_grad()
    def sample(self, batch, device="cuda", use_ddim=False, ddim_steps=20):
        """Inpainting-sample a trajectory clamped to start + goal.

        Required: start_pose, goal_pose. Optional: trajectory_mask (for
        per-sample traj_len); defaults to self.traj_len.
        """
        start = batch["start_pose"]
        goal = batch["goal_pose"]
        B = start.shape[0]

        if "trajectory_mask" in batch:
            end_idx = batch["trajectory_mask"].sum(dim=-1).long() - 1
        else:
            end_idx = torch.full((B,), self.traj_len - 1,
                                 device=device, dtype=torch.long)

        shape = (B, self.traj_len, self.traj_dim)
        return self._inpaint_sample(
            shape, start, goal, end_idx,
            device=device, use_ddim=use_ddim, ddim_steps=ddim_steps)

    def _inpaint_sample(self, shape, start, goal, end_idx, device,
                        use_ddim=False, ddim_steps=20):
        diff = self.diffusion
        B = shape[0]
        x = torch.randn(shape, device=device)
        batch_idx = torch.arange(B, device=device)

        if use_ddim:
            step_size = diff.T // ddim_steps
            timesteps = list(range(0, diff.T, step_size))[::-1]
        else:
            timesteps = list(range(diff.T - 1, -1, -1))

        for i, t_cur in enumerate(timesteps):
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)

            x = self._clamp_endpoints(x, start, goal, t, end_idx)
            pred_noise = self.noise_pred_net(x, t)   # no cond

            if use_ddim:
                alpha_bar_t = diff.alpha_bar[t_cur]
                alpha_bar_prev = (diff.alpha_bar[timesteps[i + 1]]
                                  if i + 1 < len(timesteps)
                                  else torch.tensor(1.0, device=device))
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

        # Final clean clamp — exact start/goal
        x[:, 0] = start
        x[batch_idx, end_idx] = goal
        return x
