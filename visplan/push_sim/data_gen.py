"""Per-piece push trajectory dataset generation.

Generates quasi-static push trajectories for *individual* convex polygon
pieces.  Each trajectory is a sequence of SE(2) poses produced by random
pushes applied to a single piece.

Trajectory lengths are *variable* — sampled from a configurable range,
biased toward longer sequences.

Two output formats:
    * Python dicts (for inspection / debugging)
    * Compact NumPy arrays suitable for ML dataloaders (variable-length,
      stored as a list of arrays)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from visplan.push_sim.geometry import ConvexPolygon
from visplan.push_sim.push_model import push_transition
from visplan.push_sim.polygons import random_convex_polygon


# ------------------------------------------------------------------ #
#  Single-trajectory generation
# ------------------------------------------------------------------ #

def random_push_trajectory(
    polygon: ConvexPolygon,
    start_pose: np.ndarray,
    num_steps: int = 60,
    push_distance_range: tuple[float, float] = (0.01, 0.06),
    rng: np.random.RandomState | None = None,
) -> dict:
    """Generate one push trajectory for a single piece.

    Args:
        polygon:              the piece geometry.
        start_pose:           (3,) initial [x, y, θ].
        num_steps:            number of push actions to apply.
        push_distance_range:  (min, max) push distance per step (metres).
        rng:                  random state for reproducibility.

    Returns:
        dict with keys:
            poses    — (T+1, 3) SE(2) poses including start.
            actions  — (T, 4)   per-step [edge_idx, contact_t,
                                          push_distance, push_angle].
                       push_angle is the world-frame angle of the push
                       direction (for convenience; redundant with edge_idx
                       + contact_t + pose, but handy for ML).
            polygon_vertices — (V, 2) local-frame vertices.
    """
    if rng is None:
        rng = np.random.RandomState()

    poses = [np.asarray(start_pose, dtype=np.float64)]
    actions = []

    for _ in range(num_steps):
        pose = poses[-1]

        edge_idx = rng.randint(0, polygon.num_vertices)
        contact_t = rng.uniform(0.1, 0.9)  # avoid exact corners
        push_dist = rng.uniform(*push_distance_range)

        new_pose = push_transition(polygon, pose, edge_idx, contact_t, push_dist)

        # Record the world-frame push direction angle (inward normal rotated by θ)
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[c, -s], [s, c]])
        n_world = R @ (-polygon.edge_normals[edge_idx])  # inward
        push_angle = float(np.arctan2(n_world[1], n_world[0]))

        poses.append(new_pose)
        actions.append([edge_idx, contact_t, push_dist, push_angle])

    return {
        "poses": np.array(poses, dtype=np.float64),            # (T+1, 3)
        "actions": np.array(actions, dtype=np.float64),         # (T, 4)
        "polygon_vertices": polygon.vertices.copy(),            # (V, 2)
    }


# ------------------------------------------------------------------ #
#  Variable-length step sampling
# ------------------------------------------------------------------ #

def _sample_num_steps(
    steps_range: tuple[int, int],
    rng: np.random.RandomState,
) -> int:
    """Sample a trajectory length, biased toward longer trajectories.

    Uses a Beta(2, 1) distribution mapped onto the range — most samples
    land in the upper half.
    """
    lo, hi = steps_range
    t = rng.beta(2.0, 1.0)  # skewed toward 1.0
    return int(lo + t * (hi - lo))


# ------------------------------------------------------------------ #
#  Dataset generation
# ------------------------------------------------------------------ #

def generate_dataset(
    polygons: Sequence[ConvexPolygon],
    num_trajectories_per_piece: int = 50,
    steps_range: tuple[int, int] = (20, 80),
    start_pose_range: float = 0.15,
    push_distance_range: tuple[float, float] = (0.01, 0.06),
    seed: int = 0,
) -> dict:
    """Generate a dataset of push trajectories, one piece at a time.

    For each polygon in *polygons*, generates *num_trajectories_per_piece*
    independent trajectories starting from random poses.  Trajectory
    lengths are variable, sampled from *steps_range* biased toward longer.

    Args:
        polygons:                  list of ConvexPolygon pieces.
        num_trajectories_per_piece: trajectories per piece.
        steps_range:               (min, max) number of push steps per trajectory.
        start_pose_range:          uniform sampling bound for x, y (metres).
        push_distance_range:       (min, max) push distance per step.
        seed:                      base RNG seed.

    Returns:
        dict::

            piece_idx      (N,)             which piece each trajectory belongs to
            poses          list of (T_i+1, 3)  SE(2) trajectory poses (variable length)
            actions        list of (T_i, 4)    push actions per step (variable length)
            num_steps      (N,)             number of push steps in each trajectory
            polygon_verts  list of (V_j, 2)  per-piece local vertices
    """
    rng = np.random.RandomState(seed)

    all_piece_idx = []
    all_poses = []
    all_actions = []
    all_num_steps = []
    polygon_verts = []

    for pi, polygon in enumerate(polygons):
        polygon_verts.append(polygon.vertices.copy())

        for _ in range(num_trajectories_per_piece):
            x0 = rng.uniform(-start_pose_range, start_pose_range)
            y0 = rng.uniform(-start_pose_range, start_pose_range)
            theta0 = rng.uniform(-np.pi, np.pi)
            start_pose = np.array([x0, y0, theta0])

            num_steps = _sample_num_steps(steps_range, rng)

            traj = random_push_trajectory(
                polygon, start_pose, num_steps=num_steps,
                push_distance_range=push_distance_range, rng=rng,
            )

            all_piece_idx.append(pi)
            all_poses.append(traj["poses"])
            all_actions.append(traj["actions"])
            all_num_steps.append(num_steps)

    return {
        "piece_idx": np.array(all_piece_idx, dtype=np.int32),
        "poses": all_poses,
        "actions": all_actions,
        "num_steps": np.array(all_num_steps, dtype=np.int32),
        "polygon_verts": polygon_verts,
    }


def generate_random_dataset(
    num_pieces: int = 10,
    n_verts_range: tuple[int, int] = (4, 8),
    scale: float = 0.05,
    num_trajectories_per_piece: int = 50,
    steps_range: tuple[int, int] = (20, 80),
    push_distance_range: tuple[float, float] = (0.01, 0.06),
    seed: int = 0,
) -> dict:
    """Convenience wrapper: generate random polygons then build a dataset."""
    rng = np.random.RandomState(seed)
    polygons = []
    for _ in range(num_pieces):
        nv = rng.randint(*n_verts_range)
        polygons.append(random_convex_polygon(n_verts=nv, scale=scale, rng=rng))
    return generate_dataset(
        polygons,
        num_trajectories_per_piece=num_trajectories_per_piece,
        steps_range=steps_range,
        push_distance_range=push_distance_range,
        seed=seed + 1,
    )
