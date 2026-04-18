"""Quasi-static push transition model.

Implements the ellipsoidal limit-surface approximation (Lynch & Mason 1996).
A push at contact point *c* with inward-normal direction *n* produces a
generalised velocity (twist) whose translational and rotational components
are governed by the polygon's radius of gyration.

The single entry point is :func:`push_transition`.
"""

from __future__ import annotations

import numpy as np

from visplan.push_sim.geometry import ConvexPolygon


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def push_transition(
    polygon: ConvexPolygon,
    pose: np.ndarray,
    edge_idx: int,
    contact_t: float,
    push_distance: float,
) -> np.ndarray:
    """Compute the next SE(2) pose after a quasi-static push.

    Args:
        polygon:       the polygon being pushed (local-frame geometry).
        pose:          (3,) current pose [x, y, θ].
        edge_idx:      which edge the pusher contacts (0 … V-1).
        contact_t:     parameter ∈ [0, 1] along that edge.
        push_distance: how far the pusher travels (metres).  Positive =
                       push inward (into the polygon).

    Returns:
        (3,) new pose [x', y', θ'].

    Model
    -----
    1.  Resolve the contact point and inward normal in the world frame.
    2.  Compute the moment arm  r × n  (scalar, 2D cross product).
    3.  The limit-surface twist is::

            vx = nx,   vy = ny,   ω = (r × n) / c²

        where  c² = I/m  is the polygon's squared radius of gyration
        (``polygon.moment_of_inertia``).
    4.  Normalise so that the *contact point* moves by ``push_distance``
        in the push direction, then integrate as a constant-twist SE(2)
        motion.
    """
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    R = _rotation_matrix(theta)

    # Contact point and inward normal in world frame
    contact_local = polygon.sample_edge_point(edge_idx, contact_t)  # (2,)
    normal_local = -polygon.edge_normals[edge_idx]                   # inward
    contact_world = R @ contact_local + np.array([x, y])
    n_world = R @ normal_local  # unit inward normal in world frame

    # Moment arm: vector from CoM to contact point (world frame)
    r = contact_world - np.array([x, y])

    # Squared radius of gyration
    c_sq = polygon.moment_of_inertia  # I/m

    # 2D cross product  r × n  (scalar)
    torque = r[0] * n_world[1] - r[1] * n_world[0]

    # Generalised velocity (twist): [vx, vy, ω]
    vx, vy = n_world[0], n_world[1]
    omega = torque / c_sq

    # Scale so the contact-point displacement along the push direction
    # equals push_distance.  The contact-point velocity in the push
    # direction is:  v_contact · n  =  (v + ω × r) · n
    #   where  ω × r = [-ω·ry, ω·rx]  in 2D.
    v_contact_along_n = (
        vx * n_world[0] + vy * n_world[1]
        + omega * (-r[1] * n_world[0] + r[0] * n_world[1])
    )

    if abs(v_contact_along_n) < 1e-12:
        # Degenerate: push produces no motion along its own direction
        return pose.copy()

    dt = push_distance / v_contact_along_n

    # Integrate constant twist
    dx = vx * dt
    dy = vy * dt
    dtheta = omega * dt

    return np.array([x + dx, y + dy, theta + dtheta], dtype=np.float64)
