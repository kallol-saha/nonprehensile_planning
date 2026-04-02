"""Pairwise polygon conflict detection for CBS trajectory planning.

A *conflict* occurs when two pieces physically overlap at some trajectory
timestep.  Given SE(2) trajectories for all pieces and their polygon outlines
(in local frame, centred at origin as produced by `get_polygon_outlines`), this
module checks every pair of pieces at every timestep and returns a list of
Conflict objects describing where and when they collide.

Conflict → SphereConstraint conversion
---------------------------------------
CBS resolves a conflict ⟨i, j, t⟩ by creating two child CT nodes:

  • In one child, piece i is constrained away from collision point p at time t.
  • In the other, piece j is constrained away from the same point at time t.

Following the MMD paper (§3.1) the constraint is a sphere (disk in 2D) centred
at the **world position of the collision** with radius equal to the bounding
radius of the larger of the two pieces, multiplied by a margin.  This is a
conservative but simple choice that subsumes the original MAPF vertex constraint.

All positions used in conflict detection are in **world** (unnormalised, metres)
coordinates because the polygon outlines are in metres.  The resulting
SphereConstraint centres are then converted to normalised coordinates before
being returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import rotate as shapely_rotate, translate as shapely_translate

from visplan.planning.constraints import SphereConstraint


# ---------------------------------------------------------------------------
#  Data types
# ---------------------------------------------------------------------------

@dataclass
class Conflict:
    """A pairwise collision between two pieces at a specific timestep.

    Attributes
    ----------
    piece_i : int
        Index of first piece.
    piece_j : int
        Index of second piece.
    timestep : int
        Trajectory timestep at which the collision occurs.
    world_pos : tuple[float, float]
        World (x, y) position of the conflict point in metres.  This is the
        centroid of the intersection polygon of the two pieces, or the midpoint
        of their centroids if a centroid cannot be computed.
    constraint_radius_world : float
        Radius of the sphere constraint in world metres.  Sized to cover the
        bounding circles of both pieces with a margin.
    """

    piece_i: int
    piece_j: int
    timestep: int
    world_pos: tuple[float, float]
    constraint_radius_world: float


# ---------------------------------------------------------------------------
#  Geometry helpers
# ---------------------------------------------------------------------------

def _apply_se2(outline: np.ndarray, pose: np.ndarray) -> ShapelyPolygon:
    """Transform a local-frame polygon outline by an SE(2) pose and return a
    Shapely Polygon.

    Args:
        outline: (V, 2) float64 polygon vertices in local frame.
        pose:    (3,) [x, y, theta] world pose.

    Returns:
        Shapely Polygon in world coordinates.
    """
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    world_verts = (R @ outline.T).T + np.array([x, y])
    return ShapelyPolygon(world_verts)


def _bounding_radius(outline: np.ndarray) -> float:
    """Return the circumscribed radius of a local-frame polygon outline."""
    return float(np.max(np.linalg.norm(outline, axis=1)))


# ---------------------------------------------------------------------------
#  Core detection
# ---------------------------------------------------------------------------

def detect_conflicts(
    trajectories_world: np.ndarray,
    outlines: List[np.ndarray],
    visible_range: float,
    constraint_margin: float = 1.2,
    max_conflicts: int = 1,
) -> List[Conflict]:
    """Find pairwise piece collisions across all trajectory timesteps.

    Iterates over every pair (i, j) and every timestep t.  Returns up to
    ``max_conflicts`` Conflict objects.  CBS calls this with max_conflicts=1
    (it only needs *one* conflict to branch on), which makes detection cheap.

    Args:
        trajectories_world: (N, T, 3) SE(2) trajectories in world metres
                             [x, y, theta].
        outlines:           List of N (V_i, 2) local-frame polygon outlines.
        visible_range:      Camera half-extent in metres.  Used to convert the
                            constraint centre from world to normalised coords.
        constraint_margin:  Multiplicative margin applied to the bounding radius
                            when sizing the sphere constraint (ε in Eq. 4).
                            Note: this is a separate concept from the ε inside
                            SphereConstraint – this controls the *radius* of the
                            constraint, not the padding during cost evaluation.
        max_conflicts:      Stop after finding this many conflicts.  Pass None
                            to find all.

    Returns:
        List of Conflict objects, at most ``max_conflicts`` long.
    """
    N, T, _ = trajectories_world.shape
    conflicts: List[Conflict] = []

    # Pre-compute bounding radii for all pieces
    bounding_radii = [_bounding_radius(ol) for ol in outlines]

    for i in range(N):
        for j in range(i + 1, N):
            for t in range(T):
                pose_i = trajectories_world[i, t]  # (3,)
                pose_j = trajectories_world[j, t]  # (3,)

                poly_i = _apply_se2(outlines[i], pose_i)
                poly_j = _apply_se2(outlines[j], pose_j)

                if not poly_i.is_valid:
                    poly_i = poly_i.buffer(0)
                if not poly_j.is_valid:
                    poly_j = poly_j.buffer(0)

                if poly_i.intersects(poly_j):
                    # Compute conflict world position: centroid of intersection
                    try:
                        intersection = poly_i.intersection(poly_j)
                        cx = intersection.centroid.x
                        cy = intersection.centroid.y
                    except Exception:
                        # Fallback: midpoint of the two piece centroids
                        cx = (pose_i[0] + pose_j[0]) / 2.0
                        cy = (pose_i[1] + pose_j[1]) / 2.0

                    # Constraint radius: larger bounding radius + margin
                    r_world = constraint_margin * max(
                        bounding_radii[i], bounding_radii[j]
                    )

                    conflicts.append(Conflict(
                        piece_i=i,
                        piece_j=j,
                        timestep=t,
                        world_pos=(cx, cy),
                        constraint_radius_world=r_world,
                    ))

                    if max_conflicts is not None and len(conflicts) >= max_conflicts:
                        return conflicts

    return conflicts


def conflict_to_sphere_constraint(
    conflict: Conflict,
    piece_idx: int,
    visible_range: float,
    epsilon: float = 1.2,
) -> SphereConstraint:
    """Convert a Conflict into a SphereConstraint for a specific piece.

    The constraint centre and radius are converted from world metres to the
    normalised coordinate system used by the diffusion models (dividing by
    visible_range).

    Args:
        conflict:      The Conflict to convert.
        piece_idx:     Which of the two colliding pieces to constrain
                       (must be conflict.piece_i or conflict.piece_j).
        visible_range: Camera half-extent in metres (used for normalisation).
        epsilon:       Padding factor inside the SphereConstraint cost function.

    Returns:
        SphereConstraint in normalised coordinates.
    """
    assert piece_idx in (conflict.piece_i, conflict.piece_j), (
        f"piece_idx {piece_idx} is not part of conflict "
        f"({conflict.piece_i}, {conflict.piece_j})"
    )

    cx_norm = conflict.world_pos[0] / visible_range
    cy_norm = conflict.world_pos[1] / visible_range
    r_norm = conflict.constraint_radius_world / visible_range

    # The constraint covers the single conflicting timestep (tight window)
    return SphereConstraint(
        piece_idx=piece_idx,
        center_norm=(cx_norm, cy_norm),
        radius_norm=r_norm,
        t_start=conflict.timestep,
        t_end=conflict.timestep,
        epsilon=epsilon,
    )


def count_conflicts(
    trajectories_world: np.ndarray,
    outlines: List[np.ndarray],
) -> int:
    """Count the total number of pairwise piece-timestep collisions.

    Used by CBS to rank CT nodes by number of remaining conflicts.

    Args:
        trajectories_world: (N, T, 3) SE(2) trajectories in world metres.
        outlines:           List of N (V_i, 2) local-frame polygon outlines.

    Returns:
        Total collision count (int).
    """
    N, T, _ = trajectories_world.shape
    total = 0
    for i in range(N):
        for j in range(i + 1, N):
            for t in range(T):
                poly_i = _apply_se2(outlines[i], trajectories_world[i, t])
                poly_j = _apply_se2(outlines[j], trajectories_world[j, t])
                if not poly_i.is_valid:
                    poly_i = poly_i.buffer(0)
                if not poly_j.is_valid:
                    poly_j = poly_j.buffer(0)
                if poly_i.intersects(poly_j):
                    total += 1
    return total
