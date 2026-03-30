"""DDPM noise schedule and sampling utilities shared by diffusion models."""

import torch
import torch.nn as nn
import numpy as np


class GaussianDiffusion(nn.Module):
    """Standard DDPM with cosine schedule.

    Handles:
      - Forward process: q(x_t | x_0) = N(sqrt(ᾱ_t) x_0, (1-ᾱ_t) I)
      - Training loss: predict noise ε from (x_t, t, condition)
      - Sampling: DDPM or DDIM reverse process
    """

    def __init__(self, num_timesteps: int = 100, beta_schedule: str = "cosine"):
        super().__init__()
        self.T = num_timesteps

        if beta_schedule == "cosine":
            betas = self._cosine_betas(num_timesteps)
        elif beta_schedule == "linear":
            betas = np.linspace(1e-4, 0.02, num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas)

        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer("alpha_bar", torch.tensor(alpha_bar, dtype=torch.float32))
        self.register_buffer("sqrt_alpha_bar",
                             torch.tensor(np.sqrt(alpha_bar), dtype=torch.float32))
        self.register_buffer("sqrt_one_minus_alpha_bar",
                             torch.tensor(np.sqrt(1.0 - alpha_bar), dtype=torch.float32))

    @staticmethod
    def _cosine_betas(T, s=0.008):
        t = np.arange(T + 1) / T
        f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return np.clip(betas, 0, 0.999)

    def q_sample(self, x0, t, noise=None):
        """Forward process: sample x_t given x_0 and timestep t.

        Args:
            x0: (B, T_traj, D) clean trajectories
            t:  (B,) integer timesteps in [0, self.T)
            noise: optional pre-sampled noise

        Returns:
            x_t: (B, T_traj, D) noised trajectories
            noise: the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_bar[t]          # (B,)
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t]  # (B,)

        # Expand for broadcasting: (B,) → (B, 1, 1)
        while sqrt_ab.dim() < x0.dim():
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_1_ab = sqrt_1_ab.unsqueeze(-1)

        x_t = sqrt_ab * x0 + sqrt_1_ab * noise
        return x_t, noise

    def training_loss(self, model, x0, condition):
        """Compute ε-prediction MSE loss.

        Args:
            model:     callable(x_t, t, condition) → predicted noise
            x0:        (B, T_traj, D) clean trajectories
            condition: scene embedding from vision encoder

        Returns:
            scalar loss
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        pred_noise = model(x_t, t, condition)
        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddpm_sample(self, model, shape, condition, device="cuda"):
        """Full DDPM reverse sampling loop.

        Args:
            model:     callable(x_t, t, condition) → predicted noise
            shape:     (B, T_traj, D) output shape
            condition: scene embedding
            device:    torch device

        Returns:
            (B, T_traj, D) denoised trajectories
        """
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            pred_noise = model(x, t, condition)

            alpha = self.alphas[i]
            alpha_bar = self.alpha_bar[i]
            beta = self.betas[i]

            # μ_θ(x_t, t)
            mean = (1.0 / alpha.sqrt()) * (
                x - (beta / (1.0 - alpha_bar).sqrt()) * pred_noise
            )

            if i > 0:
                sigma = beta.sqrt()
                x = mean + sigma * torch.randn_like(x)
            else:
                x = mean

        return x

    @torch.no_grad()
    def ddim_sample(self, model, shape, condition, device="cuda",
                    num_steps: int = 20, eta: float = 0.0):
        """DDIM sampling (faster, deterministic when η=0).

        Args:
            num_steps: number of denoising steps (≤ self.T)
            eta:       stochasticity (0 = deterministic DDIM, 1 = DDPM)
        """
        # Sub-sample timestep sequence
        step_size = self.T // num_steps
        timesteps = list(range(0, self.T, step_size))[::-1]

        x = torch.randn(shape, device=device)

        for i, t_cur in enumerate(timesteps):
            t = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            pred_noise = model(x, t, condition)

            alpha_bar_t = self.alpha_bar[t_cur]

            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.alpha_bar[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # Predicted x_0
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()

            # Direction pointing to x_t
            sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) *
                           (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
            dir_xt = (1 - alpha_bar_prev - sigma ** 2).sqrt() * pred_noise

            x = alpha_bar_prev.sqrt() * x0_pred + dir_xt
            if sigma > 0:
                x = x + sigma * torch.randn_like(x)

        return x
