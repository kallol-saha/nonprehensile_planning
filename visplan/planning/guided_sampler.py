"""Guidance-augmented DDPM and DDIM sampling for CBS trajectory planning.

This module provides drop-in replacements for ``GaussianDiffusion.ddpm_sample``
and ``GaussianDiffusion.ddim_sample`` that inject sphere-constraint guidance
at every denoising step.

Guidance mechanism (MMD paper Eq. 3 / 4)
-----------------------------------------
At denoising step k, after computing the noise-predicted mean μ_θ(x_t, t), we
apply a gradient step that steers x away from any active sphere constraints:

    μ_guided = μ_θ  −  λ_c · β_{k−1} · ∇_x J_c(μ_θ)

The minus sign arises because J_c is a **cost** (penalty when inside the
exclusion zone), so we descend on it.  The paper's Eq. 3 adds +η·β·∇J where J
is a reward; our formulation is equivalent with J ← −J_c.

The gradient ∇_x J_c is computed with a local torch.enable_grad() context on a
detached copy of μ_θ so that the main @torch.no_grad() context of the sampling
loop is not broken.  Only the xy dimensions (columns 0 and 1 of the trajectory)
receive non-zero gradients from J_c; the (cos θ, sin θ) columns are unaffected.

Batch sampling
--------------
Both functions accept a batch_size argument and produce (batch_size, T, 4)
outputs in normalised coordinates.  The caller (LowLevelPlanner) then converts
all B samples to world coordinates and passes them to CTNode.update_representative().
"""

from __future__ import annotations

from typing import List

import torch

from visplan.training.diffusion import GaussianDiffusion
from visplan.planning.constraints import SphereConstraint, guidance_gradient


# ---------------------------------------------------------------------------
#  Guided DDPM sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def guided_ddpm_sample(
    diffusion: GaussianDiffusion,
    noise_pred_net: torch.nn.Module,
    condition,
    traj_len: int,
    traj_dim: int,
    constraints: List[SphereConstraint],
    guidance_weight: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Guidance-augmented DDPM reverse sampling.

    Args:
        diffusion:        GaussianDiffusion instance (holds schedule buffers).
        noise_pred_net:   The noise-prediction network (TemporalUNet or
                          TrajectoryDiT).  Called as net(x_t, t, condition).
        condition:        Scene conditioning — either a (1, D) tensor (UNet) or
                          a tuple ((1, D), (1, S, D)) (DiT).  The leading batch
                          dimension of 1 will be broadcast to batch_size.
        traj_len:         T — number of trajectory timesteps.
        traj_dim:         D — trajectory feature dimension (4 for SE(2)).
        constraints:      SphereConstraints for this piece (normalised coords).
        guidance_weight:  λ_c — scales the gradient step.
        batch_size:       B — number of parallel trajectory samples.
        device:           Torch device.

    Returns:
        (B, T, D) denoised trajectories in normalised coordinates.
    """
    # Expand condition to batch_size if it came in with batch dim = 1
    cond = _expand_condition(condition, batch_size)

    x = torch.randn((batch_size, traj_len, traj_dim), device=device)

    for i in reversed(range(diffusion.T)):
        t_batch = torch.full((batch_size,), i, device=device, dtype=torch.long)
        pred_noise = noise_pred_net(x, t_batch, cond)

        alpha = diffusion.alphas[i]
        alpha_bar = diffusion.alpha_bar[i]
        beta = diffusion.betas[i]

        # Posterior mean μ_θ(x_t, t)
        mean = (1.0 / alpha.sqrt()) * (
            x - (beta / (1.0 - alpha_bar).sqrt()) * pred_noise
        )

        # Guidance: steer mean away from constraint violations
        if constraints:
            grad = _batch_guidance_gradient(mean, constraints, device)
            # Scale: λ_c · β_{k-1}  (use β_i which is β_{k} at step i)
            mean = mean - guidance_weight * beta * grad

        if i > 0:
            sigma = beta.sqrt()
            x = mean + sigma * torch.randn_like(mean)
        else:
            x = mean

    return x  # (B, T, D)


# ---------------------------------------------------------------------------
#  Guided DDIM sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def guided_ddim_sample(
    diffusion: GaussianDiffusion,
    noise_pred_net: torch.nn.Module,
    condition,
    traj_len: int,
    traj_dim: int,
    constraints: List[SphereConstraint],
    guidance_weight: float,
    batch_size: int,
    device: torch.device,
    num_steps: int = 20,
    eta: float = 0.0,
) -> torch.Tensor:
    """Guidance-augmented DDIM reverse sampling (faster, η=0 is deterministic).

    The guidance step is applied to the predicted x_0 (``x0_pred``) rather
    than the posterior mean, which is the standard formulation for classifier
    guidance in DDIM (Ho & Salimans 2021; Dhariwal & Nichol 2021).

    Args:
        num_steps: Number of denoising steps (must be ≤ diffusion.T).
        eta:       Stochasticity (0 = deterministic DDIM, 1 → DDPM).
        (other args same as guided_ddpm_sample)

    Returns:
        (B, T, D) denoised trajectories in normalised coordinates.
    """
    cond = _expand_condition(condition, batch_size)

    # Sub-sample timestep sequence (same as GaussianDiffusion.ddim_sample)
    step_size = diffusion.T // num_steps
    timesteps = list(range(0, diffusion.T, step_size))[::-1]

    x = torch.randn((batch_size, traj_len, traj_dim), device=device)

    for i, t_cur in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t_cur, device=device, dtype=torch.long)
        pred_noise = noise_pred_net(x, t_batch, cond)

        alpha_bar_t = diffusion.alpha_bar[t_cur]

        if i + 1 < len(timesteps):
            t_prev = timesteps[i + 1]
            alpha_bar_prev = diffusion.alpha_bar[t_prev]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        # Predicted clean trajectory x_0
        x0_pred = (x - (1.0 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()

        # Guidance on x0_pred: steer toward constraint-satisfying region
        if constraints:
            grad = _batch_guidance_gradient(x0_pred, constraints, device)
            beta_t = diffusion.betas[t_cur]
            x0_pred = x0_pred - guidance_weight * beta_t * grad

        # DDIM update
        sigma = eta * (
            (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) *
            (1.0 - alpha_bar_t / alpha_bar_prev)
        ).sqrt()
        dir_xt = (1.0 - alpha_bar_prev - sigma ** 2).sqrt() * pred_noise
        x = alpha_bar_prev.sqrt() * x0_pred + dir_xt

        if sigma > 0:
            x = x + sigma * torch.randn_like(x)

    return x  # (B, T, D)


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _expand_condition(condition, batch_size: int):
    """Broadcast condition tensors from batch-dim 1 to batch_size.

    Handles both:
      - A single tensor (B=1, D) → (batch_size, D)  [UNet]
      - A tuple ((B=1, D), (B=1, S, D))             [DiT]
    """
    if isinstance(condition, tuple):
        # DiT: (global_emb, spatial_tokens)
        return tuple(_expand_tensor(c, batch_size) for c in condition)
    else:
        return _expand_tensor(condition, batch_size)


def _expand_tensor(t: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Repeat a (1, ...) tensor along dim-0 to (batch_size, ...)."""
    if t.shape[0] == batch_size:
        return t
    assert t.shape[0] == 1, (
        f"Expected condition batch dim 1, got {t.shape[0]}"
    )
    return t.expand(batch_size, *t.shape[1:])


def _batch_guidance_gradient(
    x: torch.Tensor,
    constraints: List[SphereConstraint],
    device: torch.device,
) -> torch.Tensor:
    """Compute guidance gradient for a batch of trajectories.

    Evaluates ∇_x J_c independently for each sample in the batch and stacks
    the results.  Uses a local torch.enable_grad() context so this can be
    called safely from inside a @torch.no_grad() loop.

    Args:
        x:           (B, T, D) trajectory batch (detached from main graph).
        constraints: Constraints for this piece.
        device:      Torch device.

    Returns:
        (B, T, D) gradient tensor.
    """
    from visplan.planning.constraints import guidance_gradient as _guidance_grad

    B = x.shape[0]
    grads = []
    for b in range(B):
        # guidance_gradient handles the enable_grad context internally via
        # detach + requires_grad_(True) + backward()
        g = _guidance_grad(x[b], constraints)  # (T, D)
        grads.append(g)
    return torch.stack(grads, dim=0)  # (B, T, D)
