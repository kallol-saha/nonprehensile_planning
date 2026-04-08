"""Generate Voronoi shard reassembly trajectory data for diffusion model training.

For each episode this script:
  1. Creates a random Voronoi tessellation of a square (varying seed & point count)
  2. Records the goal (assembled) state
  3. Scatters pieces outward via physics-based edge-force pushes
  4. Records the start (scattered) state with overhead image and per-piece masks
  5. Generates per-piece SE(2) trajectories via linear interpolation
  6. Saves everything to disk as a compressed .npz

Data format (per episode .npz file)
------------------------------------
  start_image        (H, W, 3)    uint8    Overhead view of scattered pieces
  goal_image         (H, W, 3)    uint8    Overhead view of assembled pieces
  piece_masks        (N, H, W)    bool     Per-piece binary masks (scattered config)
  start_poses        (N, 3)       float32  Per-piece [x, y, theta] start poses
  goal_poses         (N, 3)       float32  Per-piece [x, y, theta] goal poses
  trajectories       (N, T, 3)    float32  Per-piece [x, y, theta] trajectories
  centroids          (N, 2)       float64  Original centroid positions
  num_pieces         scalar       int32    Number of pieces
  voronoi_seed       scalar       int32    Seed used for Voronoi generation
  num_voronoi_points scalar       int32    Number of Voronoi seed points
  side_length        scalar       float32  Side length of the square
  outline_<i>        (V_i, 2)    float64  Polygon vertices for piece i (local frame)

Training usage
--------------
  For each training sample, pick a random episode and a random piece index *i*:
    - Model input:  (start_image, piece_masks[i])          (+ optionally goal_image)
    - Model target: trajectories[i]   (T, 3)  SE(2) waypoints

Example
-------
  python scripts/data_gen/generate_voronoi_reassembly_data.py \
      --output_dir data/voronoi_reassembly \
      --num_episodes 1000 \
      --episodes_per_voronoi 10
"""

import argparse
import json
import os

import cv2
import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import visplan.voronoi_env  # noqa: F401 – registers the environment


# ------------------------------------------------------------------ #
#  Geometry helpers
# ------------------------------------------------------------------ #

def get_polygon_outlines(meshes):
    """Extract ordered 2D polygon outlines from 3D extruded meshes.

    Each mesh is an extruded shapely polygon centred at the origin (as
    produced by ``generate_voronoi_meshes``).  The bottom-face vertices
    are extracted, de-duplicated, and angle-sorted so they form a proper
    polygon ring.

    Returns:
        list of (V_i, 2) float64 arrays – one outline per piece.
    """
    outlines = []
    for mesh in meshes:
        verts = mesh.vertices
        z_min = verts[:, 2].min()
        bottom = verts[np.abs(verts[:, 2] - z_min) < 1e-5][:, :2]
        bottom = np.unique(np.round(bottom, 6), axis=0)
        c = bottom.mean(axis=0)
        angles = np.arctan2(bottom[:, 1] - c[1], bottom[:, 0] - c[0])
        outlines.append(bottom[np.argsort(angles)])
    return outlines


def transform_outline(outline, pose):
    """Rotate + translate a local-frame outline by an SE(2) pose [x, y, θ]."""
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ outline.T).T + np.array([x, y])


def world_to_pixel(xy, image_size, visible_range):
    """Map world (x, y) → pixel (u, v) for an overhead orthographic camera.

    Convention: world +x → image right, world +y → image up.
    Camera is centred at the origin and ``visible_range`` is the half-extent
    of the visible area in metres (≈ 0.6 m for the default overhead cam).
    """
    u = (xy[:, 0] + visible_range) / (2.0 * visible_range) * image_size
    v = (visible_range - xy[:, 1]) / (2.0 * visible_range) * image_size
    return np.stack([u, v], axis=-1).astype(np.int32)


# ------------------------------------------------------------------ #
#  Rendering helpers
# ------------------------------------------------------------------ #

def _piece_colors(n, seed=0):
    """Return *n* distinct RGB colours as a list of 3-tuples (uint8)."""
    rng = np.random.RandomState(seed + 123)
    return [tuple(int(v) for v in row) for row in rng.randint(80, 220, size=(n, 3))]


def render_overhead(outlines, poses, image_size, visible_range, colors,
                    background=(200, 200, 200)):
    """Programmatic top-down rendering of polygon pieces.

    Returns an (H, W, 3) uint8 BGR image (OpenCV convention).
    """
    img = np.full((image_size, image_size, 3), background, dtype=np.uint8)
    for outline, pose, color in zip(outlines, poses, colors):
        world = transform_outline(outline, pose)
        px = world_to_pixel(world, image_size, visible_range)
        cv2.fillPoly(img, [px], color)
        cv2.polylines(img, [px], isClosed=True, color=(0, 0, 0), thickness=1)
    return img


def generate_piece_masks(outlines, poses, image_size, visible_range):
    """Per-piece binary masks.  Returns (N, H, W) bool array."""
    masks = np.zeros((len(outlines), image_size, image_size), dtype=np.uint8)
    for i, (outline, pose) in enumerate(zip(outlines, poses)):
        world = transform_outline(outline, pose)
        px = world_to_pixel(world, image_size, visible_range)
        cv2.fillPoly(masks[i], [px], 1)
    return masks.astype(bool)


# ------------------------------------------------------------------ #
#  Pose helpers
# ------------------------------------------------------------------ #

def quat_to_yaw(q_wxyz):
    """Extract yaw (rotation about Z) from a (w, x, y, z) quaternion."""
    w, x, y, z = q_wxyz
    return float(Rotation.from_quat([x, y, z, w]).as_euler("xyz")[2])


def get_piece_poses(env):
    """Return (N, 3) float32 array of [x, y, θ] for every piece."""
    poses = []
    for name in env.unwrapped.piece_names:
        p7 = env.unwrapped.get_object_pose(name).cpu().numpy()[0]  # (7,)
        poses.append([p7[0], p7[1], quat_to_yaw(p7[3:7])])
    return np.array(poses, dtype=np.float32)


# ------------------------------------------------------------------ #
#  Trajectory generation
# ------------------------------------------------------------------ #

def interpolate_se2(start, goal, T):
    """Linearly interpolate N SE(2) poses over T steps.

    Handles angular wrapping so the shortest arc is always taken for θ.

    Args:
        start: (N, 3) [x, y, θ]
        goal:  (N, 3) [x, y, θ]
        T:     number of timesteps

    Returns:
        (N, T, 3) float32 trajectories.
    """
    t = np.linspace(0.0, 1.0, T).reshape(1, -1, 1)  # (1, T, 1)
    s = start[:, None, :]  # (N, 1, 3)
    g = goal[:, None, :]   # (N, 1, 3)

    traj = s + t * (g - s)

    # Shortest-arc interpolation for θ
    dtheta = g[..., 2:] - s[..., 2:]
    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
    traj[..., 2:] = s[..., 2:] + t * dtheta

    return traj.astype(np.float32)


# ------------------------------------------------------------------ #
#  Physics-based scattering
# ------------------------------------------------------------------ #

def compute_edge_push_force(outline, pose_se2, outward_direction, force_magnitude, rng):
    """Select a push edge and return the force vector and world-frame application point.

    Selects uniformly at random from the subset of edges whose inward normal
    has a positive dot product with ``outward_direction`` (guaranteeing the
    piece will be propelled outward).  Falls back to the single best-aligned
    edge if none qualify.

    For a CCW-ordered polygon, the inward normal of the edge from v1 to v2 is
        n = (-d_y, d_x) / |d|,   where  d = v2 - v1.

    The force is applied at the edge midpoint.  Rotation naturally emerges from
    the physics engine via the torque  τ = (midpoint - CM) × F  — no manual
    angular velocity calculation is required.

    Args:
        outline:          (V, 2) CCW-ordered local-frame polygon vertices
                          (no closing/repeated vertex).
        pose_se2:         [x, y, theta] current world-frame SE(2) pose.
        outward_direction:(2,) unit vector for the desired outward direction in
                          world frame (used only for edge filtering, may be noisy).
        force_magnitude:  Magnitude of the applied force in Newtons.
        rng:              numpy Generator (for reproducible edge selection).

    Returns:
        force_world: (3,) ndarray [Fx, Fy, 0.0] force vector in world frame (N).
        midpoint_xy: (2,) ndarray world-frame XY position of the edge midpoint.
        edge_world:  (2, 2) ndarray world-frame [v1, v2] endpoints of the chosen
                     edge, for visualisation.  None for degenerate polygons.
    """
    x, y, theta = pose_se2
    c_th, s_th = np.cos(theta), np.sin(theta)
    R_mat = np.array([[c_th, -s_th], [s_th, c_th]])
    pos = np.array([x, y])

    n_verts = len(outline)

    # Collect per-edge inward normals (world frame), local endpoints, midpoints
    inward_normals_world = []
    endpoints_local = []
    midpoints_local = []
    dots = []

    for i in range(n_verts):
        v1 = outline[i]
        v2 = outline[(i + 1) % n_verts]
        d = v2 - v1
        edge_len = np.linalg.norm(d)
        if edge_len < 1e-10:
            continue
        # Inward normal for a CCW polygon (left of travel direction)
        inward_local = np.array([-d[1], d[0]]) / edge_len
        inward_world = R_mat @ inward_local
        dot = float(np.dot(inward_world, outward_direction))
        inward_normals_world.append(inward_world)
        endpoints_local.append((v1, v2))
        midpoints_local.append((v1 + v2) / 2.0)
        dots.append(dot)

    if len(dots) == 0:
        # Degenerate polygon – fall back to a centroid-directed force
        force_world = np.array([outward_direction[0] * force_magnitude,
                                outward_direction[1] * force_magnitude, 0.0])
        return force_world, pos.copy(), None

    # Keep only edges that propel the piece outward (dot > 0).
    # Fall back to the single best edge if none qualify.
    valid_indices = [i for i, dot in enumerate(dots) if dot > 0.0]
    if len(valid_indices) == 0:
        valid_indices = [int(np.argmax(dots))]

    chosen = valid_indices[rng.integers(len(valid_indices))]
    push_dir_world = inward_normals_world[chosen]
    midpoint_local = midpoints_local[chosen]
    v1_local, v2_local = endpoints_local[chosen]

    # Edge endpoints in world frame (XY only — for visualisation)
    edge_world = np.array([
        (R_mat @ v1_local) + pos,
        (R_mat @ v2_local) + pos,
    ])  # (2, 2)

    # Edge midpoint in world frame (XY)
    midpoint_xy = (R_mat @ midpoint_local) + pos  # (2,)

    # Force along the chosen edge's inward normal
    force_world = np.array([push_dir_world[0] * force_magnitude,
                            push_dir_world[1] * force_magnitude, 0.0])

    return force_world, midpoint_xy, edge_world


def apply_force_at_world_point(uw, actor, force_3d, point_xy, piece_z):
    """Apply a 3-D force at a world XY point on an actor for one physics step.

    Decomposes the off-centre force into:
      • A linear force applied at the CM  (via actor.apply_force)
      • A torque  τ = r × F  (r = application point − CM)  to match the
        physical effect of contacting the body at that point.

    For GPU simulation this writes directly to ``cuda_rigid_body_torque`` and
    calls ``gpu_apply_rigid_dynamic_torque()``.  For CPU simulation it uses
    ``body.add_force_at_point()`` which handles the decomposition internally.

    Args:
        uw:         Unwrapped VoronoiReassembly env.
        actor:      The ManiSkill Actor to push.
        force_3d:   (3,) ndarray force in world frame [Fx, Fy, Fz] (Newtons).
        point_xy:   (2,) ndarray XY world position of the application point.
        piece_z:    Z coordinate (metres) to use for the application point.
                    Pass the current CM z so the moment arm has no z-component,
                    keeping the induced torque purely about the Z axis.
    """
    point_3d = np.array([point_xy[0], point_xy[1], piece_z], dtype=np.float64)
    force_np = np.asarray(force_3d, dtype=np.float64)

    if uw.gpu_sim_enabled:
        # --- Linear force at CM ---
        force_t = torch.tensor(force_np, dtype=torch.float32,
                               device=uw.device).unsqueeze(0)  # (1, 3)
        actor.apply_force(force_t)  # calls gpu_apply_rigid_dynamic_force internally

        # --- Torque from off-centre application: τ = r × F ---
        cm_pos = actor.pose.p[0].cpu().numpy()  # (3,) world CM position
        r = point_3d - cm_pos
        torque = np.cross(r, force_np)
        torque_t = torch.tensor(torque, dtype=torch.float32,
                                device=uw.device).unsqueeze(0)  # (1, 3)
        uw.scene.px.cuda_rigid_body_torque.torch()[
            actor._body_data_index, :3
        ] = torque_t
        uw.scene.px.gpu_apply_rigid_dynamic_torque()
    else:
        # CPU: physx body handles force-at-point → force + torque decomposition
        for body in actor._bodies:
            body.add_force_at_point(force=force_np.tolist(),
                                    point=point_3d.tolist())


def scatter_pieces(env, outlines, rng,
                   force_min=5.0, force_max=15.0,
                   direction_noise=0.35,
                   scatter_steps=150, settle_steps=350,
                   non_target_threshold=10):
    """Scatter each piece via a physics-based edge push, one piece at a time.

    For each piece i the function:
      1. Computes a desired outward direction from the assembly centre,
         perturbed with Gaussian noise (σ = ``direction_noise``) and re-normalised.
      2. Selects a random edge of piece i whose inward normal has a positive
         component along that outward direction (so the push propels the piece
         outward rather than inward).
      3. Applies a constant force at the edge midpoint for ``scatter_steps``
         physics steps, then allows ``settle_steps`` steps to settle.
         Rotation naturally emerges from the off-centre torque τ = r × F.
      4. Compares all-object poses before and after: if any non-target piece
         moved more than ``non_target_threshold`` metres, the episode is
         considered invalid and the function returns False immediately.

    Pieces are scattered sequentially (not simultaneously) so that each
    push can be checked in isolation.

    Args:
        env:                    Gymnasium wrapper around VoronoiReassembly.
        outlines:               list of (V_i, 2) CCW local-frame vertex arrays,
                                one per piece (no closing vertex).
        rng:                    numpy Generator for reproducibility.
        force_min / force_max:  Force magnitude range (Newtons).
        direction_noise:        Std-dev of Gaussian noise added to the clean
                                outward direction before edge selection.
        scatter_steps:          Physics steps during which force is applied.
        settle_steps:           Physics steps to let the piece settle after push.
        non_target_threshold:   Maximum allowed position displacement (metres)
                                of any non-target piece per push.

    Returns:
        valid (bool): True if every push caused no collateral displacement
                      exceeding ``non_target_threshold``; False means the
                      episode should be discarded.
        pushed_edges (list[np.ndarray | None]): One (2, 2) world-frame edge
                      array per piece (or None for degenerate pieces).
                      Empty list if valid is False.
    """
    uw = env.unwrapped
    pushed_edges = []
    zero_action = torch.zeros(uw.num_envs, uw.action_space.shape[-1], device=uw.device)

    for i in range(uw.num_pieces):
        # Current SE(2) pose of this piece
        p7 = uw.get_object_pose(uw.piece_names[i]).cpu().numpy()[0]
        pose_se2 = [p7[0], p7[1], quat_to_yaw(p7[3:7])]
        piece_z = float(p7[2])

        # Clean outward direction (from assembly centre toward original centroid)
        cx, cy = uw.centroids[i]
        outward = np.array([cx, cy], dtype=float)
        outward_norm = np.linalg.norm(outward)
        if outward_norm < 1e-8:
            # Centroid is at the origin; choose a random direction
            outward = rng.standard_normal(2)
            outward /= np.linalg.norm(outward)
        else:
            outward /= outward_norm

        force_magnitude = float(rng.uniform(force_min, force_max))

        # Add directional noise then re-normalise.  Edge filtering still uses
        # this (noisy) direction, so both the chosen edge and the moment-arm
        # contribute diversity to the resulting motion.
        noisy_outward = outward + rng.standard_normal(2) * direction_noise
        noisy_norm = np.linalg.norm(noisy_outward)
        if noisy_norm < 1e-8:
            noisy_outward = outward
        else:
            noisy_outward /= noisy_norm

        force_world, midpoint_xy, edge_world = compute_edge_push_force(
            outlines[i], pose_se2, noisy_outward, force_magnitude, rng
        )
        pushed_edges.append(edge_world)

        # Record all object poses BEFORE the push
        poses_before = uw.object_poses_tensor.clone()

        # Apply force at edge midpoint each physics step for scatter_steps steps.
        # Call uw.step() (unwrapped) to avoid triggering the gymnasium wrapper's
        # auto-reset when max_episode_steps is reached.
        for _ in range(scatter_steps):
            apply_force_at_world_point(uw, uw.actors[i], force_world, midpoint_xy, piece_z)
            uw.step(zero_action)

        # Let the piece settle (no force applied)
        uw.sim_step(settle_steps)

        # Record all object poses AFTER settle
        poses_after = uw.object_poses_tensor.clone()

        # Check whether any non-target piece was displaced beyond the threshold
        exceeded = uw.non_target_objects_moved_beyond_threshold(
            poses_before,
            poses_after,
            uw.actors[i],
            non_target_threshold,
        )
        if exceeded.any().item():
            return False, []  # Collateral movement detected – discard episode

    return True, pushed_edges  # All pushes clean


def validate_poses(env, xy_bound=0.45):
    """Return True if every piece is within *xy_bound* of the origin and
    above the table surface."""
    for name in env.unwrapped.piece_names:
        p = env.unwrapped.get_object_pose(name).cpu().numpy()[0]
        if abs(p[0]) > xy_bound or abs(p[1]) > xy_bound or p[2] < -0.01:
            return False
    return True


# ------------------------------------------------------------------ #
#  Visualization
# ------------------------------------------------------------------ #

def save_episode_visualization(start_image, goal_image, piece_masks,
                               trajectories, outlines, start_poses,
                               goal_poses, image_size, visible_range,
                               colors, output_path, pushed_edges=None):
    """Save a composite debug visualisation for one episode.

    Layout (left to right):
      - Goal (assembled) image with pushed edges highlighted
      - Start (scattered) image with trajectory waypoints overlaid
      - Montage of per-piece masks

    Args:
        pushed_edges: list of (2, 2) world-frame edge endpoints per piece
                      (as returned by scatter_pieces), or None to skip.
    """
    # --- goal image: highlight pushed edges in assembled configuration ---
    goal_viz = goal_image.copy()
    if pushed_edges is not None:
        for i, edge in enumerate(pushed_edges):
            if edge is None:
                continue
            ep = world_to_pixel(edge, image_size, visible_range)  # (2, 2)
            cv2.line(goal_viz, tuple(ep[0]), tuple(ep[1]), (255, 255, 255), 3)
            # Arrow from edge midpoint outward (using goal pose as centroid)
            mid_w = edge.mean(axis=0)
            cx_w = float(goal_poses[i, 0])
            cy_w = float(goal_poses[i, 1])
            push_dir = mid_w - np.array([cx_w, cy_w])
            push_norm = np.linalg.norm(push_dir)
            if push_norm > 1e-8:
                push_dir /= push_norm
                arrow_len_m = 0.03  # 3 cm arrow
                mid_px = world_to_pixel(mid_w[None], image_size, visible_range)[0]
                tip_world = mid_w + push_dir * arrow_len_m
                tip_px = world_to_pixel(tip_world[None], image_size, visible_range)[0]
                cv2.arrowedLine(goal_viz, tuple(mid_px), tuple(tip_px),
                                (255, 255, 255), 2, tipLength=0.4)

    # --- start image: trajectory waypoints only ---
    traj_img = start_image.copy()
    for i in range(len(outlines)):
        color = colors[i]
        pts = trajectories[i][:, :2]  # (T, 2)
        px = world_to_pixel(pts, image_size, visible_range)
        for j in range(len(px) - 1):
            cv2.line(traj_img, tuple(px[j]), tuple(px[j + 1]), color, 2)
        # mark start
        cv2.circle(traj_img, tuple(px[0]), 4, (0, 0, 255), -1)
        # mark goal
        cv2.circle(traj_img, tuple(px[-1]), 4, (0, 255, 0), -1)

    # --- mask montage ---
    n = piece_masks.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cell = image_size // max(cols, rows, 1)
    montage = np.full((rows * cell, cols * cell, 3), 200, dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        mask_rgb = np.stack([piece_masks[i].astype(np.uint8) * 255] * 3, axis=-1)
        mask_rgb = cv2.resize(mask_rgb, (cell, cell), interpolation=cv2.INTER_NEAREST)
        montage[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = mask_rgb
    montage = cv2.resize(montage, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    # --- compose ---
    composite = np.concatenate([goal_viz, traj_img, montage], axis=1)
    cv2.imwrite(output_path, composite)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Voronoi reassembly trajectory data for "
                    "diffusion model training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write episode .npz files.")
    p.add_argument("--num_episodes", type=int, default=1000,
                   help="Total number of episodes to generate.")
    p.add_argument("--episodes_per_voronoi", type=int, default=10,
                   help="Scatter variations per Voronoi configuration.")
    p.add_argument("--num_points_min", type=int, default=5,
                   help="Min number of Voronoi seed points.")
    p.add_argument("--num_points_max", type=int, default=12,
                   help="Max number of Voronoi seed points.")
    p.add_argument("--side_length", type=float, default=0.2,
                   help="Side length of the square to tessellate (metres).")
    p.add_argument("--trajectory_length", type=int, default=32,
                   help="Number of timesteps per trajectory.")
    p.add_argument("--image_size", type=int, default=256,
                   help="Resolution of overhead images and masks.")
    p.add_argument("--scatter_force_min", type=float, default=5.0,
                   help="Min force magnitude applied to scatter each piece (Newtons).")
    p.add_argument("--scatter_force_max", type=float, default=15.0,
                   help="Max force magnitude applied to scatter each piece (Newtons).")
    p.add_argument("--scatter_direction_noise", type=float, default=0.35,
                   help="Std-dev of Gaussian noise added to the outward scatter "
                        "direction before normalisation. Higher values make "
                        "scatter less radially predictable.")
    p.add_argument("--scatter_steps", type=int, default=150,
                   help="Physics steps while applying scatter velocity per piece.")
    p.add_argument("--settle_steps", type=int, default=350,
                   help="Physics steps to let each piece settle after its push. "
                        "Because pieces are scattered one at a time, total physics "
                        "steps per episode ≈ num_pieces × (scatter_steps + settle_steps).")
    p.add_argument("--non_target_threshold", type=float, default=0.005,
                   help="Maximum allowed position displacement (metres) of any "
                        "non-target piece during a push.  Episodes where any push "
                        "knocks a bystander piece beyond this distance are discarded.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base random seed for reproducibility.")
    p.add_argument("--save_viz", type=int, default=0,
                   help="Save debug visualisations for the first N episodes.")
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-label subfolder so multiple datasets can coexist under output_dir.
    # Format: ep<N>_pts<min>-<max>_seed<S>
    subfolder = (
        f"ep{args.num_episodes}"
        f"_pts{args.num_points_min}-{args.num_points_max}"
        f"_seed{args.seed}"
    )
    output_dir = os.path.join(args.output_dir, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "viz")
    if args.save_viz > 0:
        os.makedirs(viz_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Overhead camera geometry (matches voronoi_env.py overhead_cam config:
    # eye=[0,0,0.6], FOV=π/2, 256×256).
    cam_height = 0.6
    visible_range = cam_height * np.tan(np.pi / 4)  # ≈ 0.6 m

    episode_idx = 0
    skipped = 0
    cfg_idx = 0  # incremented each time a new Voronoi config is created

    pbar = tqdm(total=args.num_episodes, desc="Generating episodes")

    while episode_idx < args.num_episodes:
        voronoi_seed = args.seed + cfg_idx
        num_points = int(rng.integers(args.num_points_min, args.num_points_max + 1))

        # ---- Create environment for this Voronoi config ----------------
        env = gym.make(
            "VoronoiReassembly-v1",
            num_envs=1,
            render_mode=None,
            num_voronoi_points=num_points,
            side_length=args.side_length,
            placement_mode="assembled",
            voronoi_seed=voronoi_seed,
        )

        print("DEBUG: Created environment with Voronoi seed", voronoi_seed)

        outlines = get_polygon_outlines(env.unwrapped.meshes)
        colors = _piece_colors(env.unwrapped.num_pieces, seed=voronoi_seed)

        # ---- Record goal (assembled) state -----------------------------
        env.reset()
        env.unwrapped.sim_step(50)  # let pieces settle onto the table

        print("DEBUG: Recorded goal state for Voronoi seed", voronoi_seed)
        goal_poses = get_piece_poses(env)
        goal_image = render_overhead(
            outlines, goal_poses, args.image_size, visible_range, colors,
        )

        # ---- Generate scatter episodes --------------------------------
        for _ in range(args.episodes_per_voronoi):
            print("DEBUG: Scattering pieces for episode", episode_idx, "with Voronoi seed", voronoi_seed)
            if episode_idx >= args.num_episodes:
                break  # done with this Voronoi config; outer while will also exit

            # Reset to assembled, then scatter one piece at a time
            env.reset()
            env.unwrapped.sim_step(50)
            print("DEBUG: Environment reset to assembled for episode", episode_idx)
            scatter_valid, pushed_edges = scatter_pieces(
                env, outlines, rng,
                force_min=args.scatter_force_min,
                force_max=args.scatter_force_max,
                direction_noise=args.scatter_direction_noise,
                scatter_steps=args.scatter_steps,
                settle_steps=args.settle_steps,
                non_target_threshold=args.non_target_threshold,
            )
            print("DEBUG: Scatter completed for episode", episode_idx, "valid =", scatter_valid)

            # if not scatter_valid or not validate_poses(env):
            #     skipped += 1
            #     continue

            print("DEBUG: Recording start state for episode", episode_idx)

            start_poses = get_piece_poses(env)
            start_image = render_overhead(
                outlines, start_poses, args.image_size, visible_range, colors,
            )
            piece_masks = generate_piece_masks(
                outlines, start_poses, args.image_size, visible_range,
            )
            trajectories = interpolate_se2(
                start_poses, goal_poses, args.trajectory_length,
            )

            # ---- Save episode ------------------------------------------
            fname = os.path.join(output_dir, f"episode_{episode_idx:06d}.npz")
            save_dict = dict(
                start_image=start_image,
                goal_image=goal_image,
                piece_masks=piece_masks,
                start_poses=start_poses,
                goal_poses=goal_poses,
                trajectories=trajectories,
                centroids=np.array(env.unwrapped.centroids),
                num_pieces=np.int32(env.unwrapped.num_pieces),
                voronoi_seed=np.int32(voronoi_seed),
                num_voronoi_points=np.int32(num_points),
                side_length=np.float32(args.side_length),
            )
            # Store per-piece polygon outlines (variable vertex count)
            for i, ol in enumerate(outlines):
                save_dict[f"outline_{i}"] = ol
            np.savez_compressed(fname, **save_dict)

            # ---- Optional debug visualisation --------------------------
            if episode_idx < args.save_viz:
                save_episode_visualization(
                    start_image, goal_image, piece_masks,
                    trajectories, outlines, start_poses, goal_poses,
                    args.image_size, visible_range, colors,
                    os.path.join(viz_dir, f"episode_{episode_idx:06d}.png"),
                    pushed_edges=pushed_edges,
                )

            episode_idx += 1
            pbar.update(1)

        env.close()
        cfg_idx += 1

    pbar.close()

    # ---- Global metadata -----------------------------------------------
    meta = dict(
        trajectory_length=args.trajectory_length,
        image_size=args.image_size,
        side_length=args.side_length,
        visible_range=float(visible_range),
        num_points_min=args.num_points_min,
        num_points_max=args.num_points_max,
        scatter_force_min=args.scatter_force_min,
        scatter_force_max=args.scatter_force_max,
        scatter_direction_noise=args.scatter_direction_noise,
        scatter_steps=args.scatter_steps,
        settle_steps=args.settle_steps,
        non_target_threshold=args.non_target_threshold,
        seed=args.seed,
        total_episodes=episode_idx,
        skipped=skipped,
    )
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone – {episode_idx} episodes saved to {output_dir}"
          f" ({skipped} scattered states skipped)")


if __name__ == "__main__":
    main()
