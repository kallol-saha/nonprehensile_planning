"""Polygon-based 2D trajectory visualisation helpers.

Shared by plan_cbs.py (inline rendering during evaluation) and
visualize_plan.py (post-hoc rendering from saved .npz files).
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
#  Colour palette
# ---------------------------------------------------------------------------

# 12-colour Paired palette (ColorBrewer), perceptually distinct
_PALETTE: list[tuple[int, int, int]] = [
    (166, 206, 227), (31,  120, 180), (178, 223, 138),
    (51,  160,  44), (251, 154, 153), (227,  26,  28),
    (253, 191, 111), (255, 127,   0), (202, 178, 214),
    (106,  61, 154), (255, 255, 153), (177,  89,  40),
]


def build_conflict_map(
    trajectories_world: np.ndarray,
    outlines: list[np.ndarray],
) -> dict[int, set[int]]:
    """Return a mapping from timestep → set of piece indices involved in a conflict.

    Uses detect_conflicts with max_conflicts=None to find all collisions, then
    indexes them by timestep for O(1) lookup during frame rendering.

    Args:
        trajectories_world: (N, T, 3) SE(2) trajectories in world metres.
        outlines:           List of N (V_i, 2) local-frame polygon outlines.

    Returns:
        Dict mapping each conflicted timestep to the set of piece indices that
        overlap at that timestep.  Timesteps with no conflict are absent.
    """
    from visplan.planning.conflict_detector import detect_conflicts

    conflicts = detect_conflicts(
        trajectories_world=trajectories_world,
        outlines=outlines,
        visible_range=1.0,  # only used for normalisation, not needed here
        max_conflicts=None,
    )
    cmap: dict[int, set[int]] = {}
    for c in conflicts:
        t = c.timestep
        if t not in cmap:
            cmap[t] = set()
        cmap[t].add(c.piece_i)
        cmap[t].add(c.piece_j)
    return cmap


def piece_colors(n: int, seed: int = 0) -> list[tuple[int, int, int]]:
    """Return n distinct RGB colours."""
    if n <= len(_PALETTE):
        return _PALETTE[:n]
    rng = np.random.RandomState(seed + 123)
    return [tuple(int(v) for v in row) for row in rng.randint(60, 220, size=(n, 3))]


# ---------------------------------------------------------------------------
#  Geometry helpers
# ---------------------------------------------------------------------------

def transform_polygon(outline: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply SE(2) pose to local-frame polygon outline.

    Args:
        outline: (V, 2) local-frame vertices in metres.
        pose:    (3,) [x, y, theta] world pose.
    Returns:
        (V, 2) world-frame vertices.
    """
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return (rot @ outline.T).T + np.array([x, y])


def world_to_pixel(xy: np.ndarray, image_size: int, half_range: float) -> np.ndarray:
    """World (x, y) → integer pixel (u, v).  World +y is image up."""
    u = (xy[:, 0] + half_range) / (2.0 * half_range) * image_size
    v = (half_range - xy[:, 1]) / (2.0 * half_range) * image_size
    return np.stack([u, v], axis=-1).astype(np.int32)


def _lighter(color: tuple[int, int, int], alpha: float = 0.35) -> tuple[int, int, int]:
    return tuple(int(c * alpha + 255 * (1 - alpha)) for c in color)


# ---------------------------------------------------------------------------
#  Frame renderer
# ---------------------------------------------------------------------------

def render_frame(
    outlines: list[np.ndarray],
    current_poses: np.ndarray,
    traj_history: np.ndarray,
    goal_poses: np.ndarray,
    colors: list[tuple],
    image_size: int,
    half_range: float,
    label: str = "",
    show_step: int | None = None,
    total_steps: int | None = None,
    conflicted_pieces: set | None = None,
) -> np.ndarray:
    """Render one video frame as an RGB numpy array (image_size × image_size × 3).

    Draws (back-to-front):
      1. Near-white background
      2. Goal polygon outlines (lighter shade of piece colour)
      3. Trajectory history paths
      4. Current filled polygons with black outline
      5. Red thick border on conflicted pieces (if conflicted_pieces provided)
    """
    img = np.full((image_size, image_size, 3), 245, dtype=np.uint8)

    # Goal outlines
    for i, outline in enumerate(outlines):
        world_v = transform_polygon(outline, goal_poses[i])
        px = world_to_pixel(world_v, image_size, half_range).reshape(-1, 1, 2)
        cv2.polylines(img, [px], True, _lighter(colors[i], alpha=0.5), 2, cv2.LINE_AA)

    # Trajectory history
    for i in range(len(outlines)):
        pts_world = traj_history[i, :, :2]
        if len(pts_world) > 1:
            px = world_to_pixel(pts_world, image_size, half_range)
            path_color = _lighter(colors[i], alpha=0.7)
            for k in range(len(px) - 1):
                cv2.line(img, tuple(px[k]), tuple(px[k + 1]), path_color, 1, cv2.LINE_AA)

    # Current filled polygons
    for i, outline in enumerate(outlines):
        world_v = transform_polygon(outline, current_poses[i])
        px = world_to_pixel(world_v, image_size, half_range).reshape(-1, 1, 2)
        cv2.fillPoly(img, [px], colors[i])
        cv2.polylines(img, [px], True, (30, 30, 30), 1, cv2.LINE_AA)

    # Conflict highlight: thick red border on colliding pieces
    if conflicted_pieces:
        for i, outline in enumerate(outlines):
            if i in conflicted_pieces:
                world_v = transform_polygon(outline, current_poses[i])
                px = world_to_pixel(world_v, image_size, half_range).reshape(-1, 1, 2)
                cv2.polylines(img, [px], True, (220, 30, 30), 3, cv2.LINE_AA)

    if label:
        cv2.putText(img, label, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    if show_step is not None and total_steps is not None:
        cv2.putText(img, f"t={show_step}/{total_steps - 1}", (8, image_size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
#  Video writers
# ---------------------------------------------------------------------------

def render_solo_video(
    outlines: list[np.ndarray],
    trajectories: np.ndarray,
    goal_poses: np.ndarray,
    colors: list[tuple],
    image_size: int,
    half_range: float,
    output_path: str,
    fps: int,
    label: str = "",
    conflict_map: dict | None = None,
) -> None:
    """Animate a single trajectory set and save as MP4.

    Args:
        outlines:     N local-frame polygon outlines.
        trajectories: (N, T, 3) SE(2) trajectories.
        goal_poses:   (N, 3) goal SE(2) poses.
        colors:       N RGB colour tuples.
        image_size:   Output frame resolution (square).
        half_range:   Camera half-extent in metres.
        output_path:  Output .mp4 path.
        fps:          Video frame rate.
        label:        Text label drawn on frames.
        conflict_map: Optional dict mapping timestep → set of conflicted piece indices.
                      Conflicted pieces are highlighted with a red border.
    """
    T = trajectories.shape[1]
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (image_size, image_size)
    )

    def _write(t: int) -> None:
        cp = conflict_map.get(t) if conflict_map else None
        frame = render_frame(
            outlines, trajectories[:, t], trajectories[:, : t + 1],
            goal_poses, colors, image_size, half_range,
            label=label, show_step=t, total_steps=T,
            conflicted_pieces=cp,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    for _ in range(fps):      # hold start 1 s
        _write(0)
    for t in range(T):
        _write(t)
    for _ in range(fps):      # hold end 1 s
        _write(T - 1)

    writer.release()


def render_methods_video(
    outlines: list[np.ndarray],
    panels: list[tuple],
    goal_poses: np.ndarray,
    colors: list[tuple],
    image_size: int,
    half_range: float,
    output_path: str,
    fps: int,
) -> None:
    """Side-by-side comparison video for an arbitrary number of methods.

    Args:
        outlines: N local-frame polygon outlines.
        panels:   List of (label, trajectories, conflict_map) tuples, one per method.
                  trajectories is (N, T, 3); conflict_map is dict[int, set[int]] or None.
        goal_poses: (N, 3) goal SE(2) poses.
        colors:   N RGB colour tuples.
        image_size: Per-panel resolution (square).
        half_range: Camera half-extent in metres.
        output_path: Output .mp4 path.
        fps:      Video frame rate.
    """
    if not panels:
        return
    T = panels[0][1].shape[1]
    n_panels = len(panels)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (image_size * n_panels, image_size)
    )

    def _row(t: int) -> np.ndarray:
        frames = []
        for label, trajs, cmap in panels:
            cp = cmap.get(t) if cmap else None
            frames.append(render_frame(
                outlines, trajs[:, t], trajs[:, : t + 1],
                goal_poses, colors, image_size, half_range,
                label=label, show_step=t, total_steps=T,
                conflicted_pieces=cp,
            ))
        return cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR)

    for _ in range(fps):
        writer.write(_row(0))
    for t in range(T):
        writer.write(_row(t))
    for _ in range(fps):
        writer.write(_row(T - 1))

    writer.release()


def render_comparison_video(
    outlines: list[np.ndarray],
    unconstrained_trajs: np.ndarray,
    planned_trajs: np.ndarray,
    goal_poses: np.ndarray,
    colors: list[tuple],
    image_size: int,
    half_range: float,
    output_path: str,
    fps: int,
    left_label: str = "Unconstrained",
    right_label: str = "Planned",
    left_conflict_map: dict | None = None,
    right_conflict_map: dict | None = None,
) -> None:
    """Side-by-side comparison video: unconstrained (left) vs planned (right).

    Thin wrapper around render_methods_video kept for backward compatibility
    (used by visualize_plan.py).
    """
    render_methods_video(
        outlines=outlines,
        panels=[
            (left_label, unconstrained_trajs, left_conflict_map),
            (right_label, planned_trajs, right_conflict_map),
        ],
        goal_poses=goal_poses,
        colors=colors,
        image_size=image_size,
        half_range=half_range,
        output_path=output_path,
        fps=fps,
    )
