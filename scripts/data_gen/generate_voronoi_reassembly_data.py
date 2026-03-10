"""Generate Voronoi shard reassembly trajectory data for diffusion model training.

For each episode this script:
  1. Creates a random Voronoi tessellation of a square (varying seed & point count)
  2. Records the goal (assembled) state
  3. Scatters pieces outward via physics-based velocity perturbations
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

def scatter_pieces(env, rng,
                   speed_min=0.3, speed_max=0.8,
                   scatter_steps=100, settle_steps=200):
    """Push every piece outward from the centre with random velocity, then
    let physics settle.

    Each piece's velocity direction is its centroid direction plus isotropic
    noise; the speed is sampled uniformly in ``[speed_min, speed_max]``.
    """
    uw = env.unwrapped
    for i, (cx, cy) in enumerate(uw.centroids):
        direction = np.array([cx, cy, 0.0])
        direction[:2] += rng.normal(0, 0.15, size=2)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            direction = np.array([rng.normal(), rng.normal(), 0.0])
            norm = np.linalg.norm(direction)
        direction /= norm

        speed = rng.uniform(speed_min, speed_max)
        vel = torch.tensor(direction * speed, dtype=torch.float32).unsqueeze(0)
        uw.set_piece_velocities(vel, piece_indices=[i])

    uw.sim_step(scatter_steps)
    uw.sim_step(settle_steps)


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
                               colors, output_path):
    """Save a composite debug visualisation for one episode.

    Layout (left to right):
      - Start image with trajectory waypoints overlaid
      - Goal image
      - Montage of per-piece masks
    """
    # --- trajectory overlay on start image ---
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
    composite = np.concatenate([traj_img, goal_image, montage], axis=1)
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
    p.add_argument("--num_points_min", type=int, default=3,
                   help="Min number of Voronoi seed points.")
    p.add_argument("--num_points_max", type=int, default=8,
                   help="Max number of Voronoi seed points.")
    p.add_argument("--side_length", type=float, default=0.2,
                   help="Side length of the square to tessellate (metres).")
    p.add_argument("--trajectory_length", type=int, default=32,
                   help="Number of timesteps per trajectory.")
    p.add_argument("--image_size", type=int, default=256,
                   help="Resolution of overhead images and masks.")
    p.add_argument("--scatter_speed_min", type=float, default=0.3,
                   help="Min outward scatter speed (m/s).")
    p.add_argument("--scatter_speed_max", type=float, default=0.8,
                   help="Max outward scatter speed (m/s).")
    p.add_argument("--scatter_steps", type=int, default=100,
                   help="Physics steps while applying scatter velocity.")
    p.add_argument("--settle_steps", type=int, default=200,
                   help="Physics steps to let pieces settle after scatter.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base random seed for reproducibility.")
    p.add_argument("--save_viz", type=int, default=0,
                   help="Save debug visualisations for the first N episodes.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, "viz")
    if args.save_viz > 0:
        os.makedirs(viz_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Overhead camera geometry (matches voronoi_env.py overhead_cam config:
    # eye=[0,0,0.6], FOV=π/2, 256×256).
    cam_height = 0.6
    visible_range = cam_height * np.tan(np.pi / 4)  # ≈ 0.6 m

    num_configs = -(-args.num_episodes // args.episodes_per_voronoi)  # ceil div

    episode_idx = 0
    skipped = 0

    pbar = tqdm(total=args.num_episodes, desc="Generating episodes")

    for cfg_idx in range(num_configs):
        if episode_idx >= args.num_episodes:
            break

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

        outlines = get_polygon_outlines(env.unwrapped.meshes)
        colors = _piece_colors(env.unwrapped.num_pieces, seed=voronoi_seed)

        # ---- Record goal (assembled) state -----------------------------
        env.reset()
        env.unwrapped.sim_step(50)  # let pieces settle onto the table
        goal_poses = get_piece_poses(env)
        goal_image = render_overhead(
            outlines, goal_poses, args.image_size, visible_range, colors,
        )

        # ---- Generate scatter episodes --------------------------------
        for _ in range(args.episodes_per_voronoi):
            if episode_idx >= args.num_episodes:
                break

            # Reset to assembled, then scatter
            env.reset()
            env.unwrapped.sim_step(50)
            scatter_pieces(
                env, rng,
                speed_min=args.scatter_speed_min,
                speed_max=args.scatter_speed_max,
                scatter_steps=args.scatter_steps,
                settle_steps=args.settle_steps,
            )

            if not validate_poses(env):
                skipped += 1
                continue

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
            fname = os.path.join(args.output_dir, f"episode_{episode_idx:06d}.npz")
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
                )

            episode_idx += 1
            pbar.update(1)

        env.close()

    pbar.close()

    # ---- Global metadata -----------------------------------------------
    meta = dict(
        trajectory_length=args.trajectory_length,
        image_size=args.image_size,
        side_length=args.side_length,
        visible_range=float(visible_range),
        num_points_min=args.num_points_min,
        num_points_max=args.num_points_max,
        scatter_speed_min=args.scatter_speed_min,
        scatter_speed_max=args.scatter_speed_max,
        scatter_steps=args.scatter_steps,
        settle_steps=args.settle_steps,
        seed=args.seed,
        total_episodes=episode_idx,
        skipped=skipped,
    )
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone – {episode_idx} episodes saved to {args.output_dir}"
          f" ({skipped} scattered states skipped)")


if __name__ == "__main__":
    main()
