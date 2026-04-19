"""Low-level planner: wraps a trained diffusion model and produces trajectory
batches for a single piece under a set of sphere constraints.

DiffusionLowLevel is the interface between CBS (which reasons about constraints
and conflicts) and the generative model (which produces trajectory samples).

Responsibilities
----------------
1. Accept a pre-built ``batch`` dict (same format as the training dataset) that
   contains the scene images and piece mask for one piece.
2. Encode the scene via the model's vision encoder to produce a conditioning
   vector.
3. Call the appropriate guided sampler (DDPM or DDIM) with the current
   SphereConstraints to produce B normalised trajectory samples (B, T, 4).
4. Denormalise the samples to world SE(2) coordinates (B, T, 3) in metres.
5. Return the batch for CTNode.update_representative() to pick the best one.

Coordinate conventions
-----------------------
The diffusion model operates in normalised space:
    x̃ = x / visible_range,   ỹ = y / visible_range,   θ → (cos θ, sin θ)

DiffusionLowLevel converts:
  • world → normalised  before passing to the sampler
  • normalised → world  after sampling

The constraints stored in CTNode are already in normalised coords
(SphereConstraint.center_norm, radius_norm), so they can be passed directly to
the guided sampler without conversion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

from visplan.planning.constraints import SphereConstraint
from visplan.planning.guided_sampler import guided_ddim_sample, guided_ddpm_sample


# ---------------------------------------------------------------------------
#  Protocol: shared interface for all low-level planners
# ---------------------------------------------------------------------------

@runtime_checkable
class LowLevelPlannerProtocol(Protocol):
    """Structural interface that all low-level planners must satisfy.

    Both DiffusionLowLevel and RRTLowLevel implement this protocol so that
    CBS and compare_methods.py can use either interchangeably.

    Attributes
    ----------
    device : Any
        The device (or device string) on which tensors should be placed.
        DiffusionLowLevel sets this to a torch.device; RRTLowLevel uses "cpu".
    """

    device: Any

    def plan_piece(
        self,
        scene_batch: dict,
        constraints: List[SphereConstraint],
    ) -> np.ndarray:
        """Sample a batch of trajectories for one piece under the given constraints.

        Args:
            scene_batch: Dict with at minimum keys:
                ``start_image`` (1, 3, H, W) float32 in [0, 1]
                ``goal_image``  (1, 3, H, W) float32 in [0, 1]
                ``piece_mask``  (1, 1, H, W) float32 in {0, 1}
                plus optional keys ``start_pose``, ``goal_pose``, ``outline``,
                ``piece_idx`` used by non-diffusion planners.
            constraints: SphereConstraints (normalised coords) for this piece.

        Returns:
            (B, T, 3) numpy float32 array of SE(2) trajectories in world metres
            [x, y, theta].
        """
        ...


# ---------------------------------------------------------------------------
#  DiffusionLowLevel (formerly LowLevelPlanner)
# ---------------------------------------------------------------------------

class DiffusionLowLevel:
    """Wraps a trained diffusion model as a CBS low-level planner.

    Args:
        model:            Trained DiffusionTransformer or DiffusionUNet instance
                          (already on the target device, in eval mode).
        visible_range:    Camera half-extent in metres (used for
                          normalisation / denormalisation).
        batch_size:       B — number of trajectory samples per planning call.
        use_ddim:         If True use DDIM (faster); otherwise full DDPM.
        ddim_steps:       Number of DDIM denoising steps.
        guidance_weight:  λ_c — scale of the constraint guidance gradient.
        device:           Torch device.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        visible_range: float = 0.6,
        batch_size: int = 8,
        use_ddim: bool = True,
        ddim_steps: int = 20,
        guidance_weight: float = 0.2,
        device: Union[str, torch.device] = "cuda",
        tail_blend_steps: int = 4,
    ):
        self.model = model
        self.visible_range = visible_range
        self.batch_size = batch_size
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.guidance_weight = guidance_weight
        self.device = torch.device(device)
        # Number of final waypoints linearly blended toward the exact goal pose
        # so the trajectory lands on goal. 0 disables. See _blend_tail_to_goal.
        self.tail_blend_steps = int(tail_blend_steps)

        # Infer traj_len and traj_dim from the model attributes
        self.traj_len: int = model.traj_len
        self.traj_dim: int = model.traj_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_piece(
        self,
        scene_batch: Dict[str, torch.Tensor],
        constraints: List[SphereConstraint],
    ) -> np.ndarray:
        """Sample a batch of trajectories for one piece under the given constraints.

        Args:
            scene_batch: Dict with keys:
                ``start_image`` (1, 3, H, W) float32 in [0, 1]
                ``goal_image``  (1, 3, H, W) float32 in [0, 1]
                ``piece_mask``  (1, 1, H, W) float32 in {0, 1}
                All tensors on self.device.
            constraints: SphereConstraints (normalised coords) applicable to
                this piece.  May be empty.

        Returns:
            (B, T, 3) numpy float32 array of SE(2) trajectories in world metres
            [x, y, theta].
        """
        # Encode scene once (no grad needed for the encoder itself)
        with torch.no_grad():
            condition = self._encode_scene(scene_batch)

        # Sample B trajectories in normalised space
        traj_norm = self._sample_normalised(condition, constraints)
        # traj_norm: (B, T, 4)

        # Denormalise to world SE(2) metres
        traj_world = self._denormalise(traj_norm)  # (B, T, 3)

        # Blend the tail toward the exact goal pose so the trajectory lands
        # precisely at goal (mitigates the learned policy's terminal drift).
        goal_pose = scene_batch.get("goal_pose")
        if self.tail_blend_steps > 0 and goal_pose is not None:
            traj_world = self._blend_tail_to_goal(traj_world, goal_pose)
        return traj_world

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_scene(self, scene_batch: Dict[str, torch.Tensor]):
        """Run the model's vision encoder and return the conditioning object.

        Handles both DiffusionUNet (returns a tensor) and DiffusionTransformer
        (returns a (global, spatial) tuple).
        """
        # Both DiffusionUNet and DiffusionTransformer expose _encode_scene()
        return self.model._encode_scene(scene_batch)

    def _sample_normalised(
        self,
        condition,
        constraints: List[SphereConstraint],
    ) -> torch.Tensor:
        """Run the guided sampler and return (B, T, 4) normalised trajectories."""
        sampler_kwargs = dict(
            diffusion=self.model.diffusion,
            noise_pred_net=self.model.noise_pred_net,
            condition=condition,
            traj_len=self.traj_len,
            traj_dim=self.traj_dim,
            constraints=constraints,
            guidance_weight=self.guidance_weight,
            batch_size=self.batch_size,
            device=self.device,
        )

        if self.use_ddim:
            return guided_ddim_sample(
                **sampler_kwargs,
                num_steps=self.ddim_steps,
                eta=0.0,
            )
        else:
            return guided_ddpm_sample(**sampler_kwargs)

    def _blend_tail_to_goal(
        self,
        traj_world: np.ndarray,
        goal_pose,
    ) -> np.ndarray:
        """Linearly blend the last K waypoints toward the exact goal pose.

        Weight ramps from 0 at t = T-K to 1 at t = T-1, so the last waypoint
        equals goal and earlier waypoints are only slightly nudged. Theta is
        blended along the shortest arc to avoid wrap-around artefacts.

        Args:
            traj_world: (B, T, 3) SE(2) trajectories in world metres.
            goal_pose:  (3,) numpy array or tensor [x, y, theta] in world metres.

        Returns:
            (B, T, 3) float32 trajectory with endpoint clamped to goal.
        """
        K = min(self.tail_blend_steps, self.traj_len)
        if K <= 0:
            return traj_world

        if isinstance(goal_pose, torch.Tensor):
            goal = goal_pose.detach().cpu().numpy().astype(np.float32).reshape(3)
        else:
            goal = np.asarray(goal_pose, dtype=np.float32).reshape(3)

        out = traj_world.copy()
        T = out.shape[1]
        # Weights 1/K, 2/K, ..., K/K over the last K timesteps
        alphas = np.linspace(1.0 / K, 1.0, K, dtype=np.float32)
        tail_idx = np.arange(T - K, T)

        # xy: simple linear interpolation
        tail_xy = out[:, tail_idx, :2]                               # (B, K, 2)
        out[:, tail_idx, :2] = (
            (1.0 - alphas)[None, :, None] * tail_xy
            + alphas[None, :, None] * goal[:2][None, None, :]
        )

        # theta: shortest-arc blend, then wrap to [-π, π]
        tail_th = out[:, tail_idx, 2]                                # (B, K)
        dtheta = (goal[2] - tail_th + np.pi) % (2 * np.pi) - np.pi    # (B, K)
        blended = tail_th + alphas[None, :] * dtheta
        out[:, tail_idx, 2] = (blended + np.pi) % (2 * np.pi) - np.pi

        return out.astype(np.float32)

    def _denormalise(self, traj_norm: torch.Tensor) -> np.ndarray:
        """Convert (B, T, 4) normalised tensor → (B, T, 3) world numpy array.

        Normalised format: [x̃, ỹ, cos θ, sin θ]
        World format:      [x (m), y (m), theta (rad)]
        """
        traj_np = traj_norm.cpu().numpy()  # (B, T, 4)
        xy = traj_np[..., :2] * self.visible_range           # (B, T, 2)
        theta = np.arctan2(traj_np[..., 3], traj_np[..., 2])  # (B, T)
        return np.concatenate(
            [xy, theta[..., np.newaxis]], axis=-1
        ).astype(np.float32)  # (B, T, 3)


# ---------------------------------------------------------------------------
#  Backward-compatibility alias
# ---------------------------------------------------------------------------

# Keep old name importable so existing code that references LowLevelPlanner
# continues to work during the transition period.
LowLevelPlanner = DiffusionLowLevel
