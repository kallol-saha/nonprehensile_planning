"""Sphere constraints for CBS-guided diffusion planning.

A SphereConstraint encodes the instruction "piece i must not be within radius r
of world point p during timestep interval [t_start, t_end]".

All positions are stored in **normalised** coordinates (xy divided by
visible_range so values lie in [-1, 1]) to match the trajectory representation
used by the diffusion models.  The radius is also in normalised units.

Guidance cost (Eq. 4 of MMD paper, adapted for SE(2))
------------------------------------------------------
For a trajectory tensor x of shape (B, T, 4) [x̃, ỹ, cos θ, sin θ]:

    J_c(x) = Σ_{t ∈ [t_start, t_end]}  max(ε·r  −  ‖x[t, :2] − p̃‖₂ ,  0)

where ε ≥ 1 is a padding factor.  The cost is zero when the piece is
sufficiently far from p̃, and increases linearly as it moves closer.

Guidance gradient (used inside the DDIM/DDPM loop)
---------------------------------------------------
During the k-th denoising step the guidance term modifies the mean:

    μ_guided = μ_θ  −  λ_c · β_{k-1} · ∇_x J_c(μ_θ)

where λ_c is the guidance weight and β_{k-1} is the diffusion noise schedule
coefficient at step k−1 (matching Eq. 3 of the MMD paper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class SphereConstraint:
    """A single sphere (disk in 2D) constraint for one piece.

    Attributes
    ----------
    piece_idx : int
        Index of the piece this constraint applies to.
    center_norm : tuple[float, float]
        Constraint centre (x̃, ỹ) in normalised coordinates.
    radius_norm : float
        Exclusion radius in normalised coordinates.
    t_start : int
        First trajectory timestep (inclusive) the constraint is active.
    t_end : int
        Last trajectory timestep (inclusive) the constraint is active.
        Use t_end = T-1 to cover the whole trajectory.
    epsilon : float
        Padding factor ε ≥ 1 that inflates the effective exclusion radius
        during cost evaluation (matches the ε in Eq. 4 of MMD).
    """

    piece_idx: int
    center_norm: tuple[float, float]
    radius_norm: float
    t_start: int
    t_end: int
    epsilon: float = 1.2


def compute_guidance_cost(
    x: torch.Tensor,
    constraints: List[SphereConstraint],
) -> torch.Tensor:
    """Compute the total sphere-constraint guidance cost for one piece's trajectory.

    Args:
        x:           (T, 4) or (B, T, 4) trajectory tensor [x̃, ỹ, cos θ, sin θ].
                     Must have requires_grad=True when gradient guidance is needed.
        constraints: List of SphereConstraints that apply to this piece.
                     The caller is responsible for passing only the constraints
                     that belong to the piece being planned.

    Returns:
        Scalar cost tensor (sum over all active timesteps and constraints).
        Returns zero if the constraint list is empty.
    """
    if not constraints:
        return torch.zeros(1, device=x.device, dtype=x.dtype).squeeze()

    # Accept both (T, 4) and (B, T, 4); normalise to (B, T, 4)
    squeezed = x.dim() == 2
    if squeezed:
        x = x.unsqueeze(0)  # (1, T, 4)

    B, T, _ = x.shape
    total_cost = torch.zeros(1, device=x.device, dtype=x.dtype).squeeze()

    for c in constraints:
        # Build a mask tensor of the active timestep indices
        t_start = max(0, c.t_start)
        t_end = min(T - 1, c.t_end)
        if t_start > t_end:
            continue  # constraint window outside trajectory length

        # Center in a tensor on the same device/dtype
        center = torch.tensor(
            [c.center_norm[0], c.center_norm[1]],
            device=x.device, dtype=x.dtype,
        )  # (2,)

        # xy positions at active timesteps: (B, window, 2)
        xy = x[:, t_start:t_end + 1, :2]

        # Euclidean distance from constraint centre: (B, window)
        dist = torch.norm(xy - center.unsqueeze(0).unsqueeze(0), dim=-1)

        # Hinge loss: cost > 0 only when inside ε·r
        effective_radius = c.epsilon * c.radius_norm
        cost = torch.clamp(effective_radius - dist, min=0.0)  # (B, window)

        total_cost = total_cost + cost.sum()

    return total_cost


def guidance_gradient(
    x: torch.Tensor,
    constraints: List[SphereConstraint],
) -> torch.Tensor:
    """Compute ∇_x J_c(x) for use in guided diffusion sampling.

    This function evaluates J_c(x) with a local autograd context and returns
    the gradient w.r.t. x.  It is designed to be called on a detached copy of
    the current mean trajectory so the main computation graph is not affected.

    Args:
        x:           (T, 4) or (B, T, 4) float32 tensor (will be detached and
                     re-enabled for grad internally).
        constraints: SphereConstraints applicable to this piece.

    Returns:
        Gradient tensor of same shape as x.  Zero tensor when no constraints.
    """
    if not constraints:
        return torch.zeros_like(x)

    # Work on a leaf copy so we can call backward without polluting the graph
    x_leaf = x.detach().requires_grad_(True)

    cost = compute_guidance_cost(x_leaf, constraints)

    # If cost has no grad_fn (e.g. all constraints fell outside the trajectory
    # window), backward() would fail — return zeros instead.
    if not cost.requires_grad:
        return torch.zeros_like(x)

    cost.backward()

    grad = x_leaf.grad.detach()  # same shape as x
    return grad
