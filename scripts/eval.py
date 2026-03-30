"""Evaluate trajectory prediction models and generate comparison videos.

For each model × {train, val}, produces an MP4 video stepping through
trajectory timesteps with ground-truth (green) and predicted (red) waypoints
overlaid on the start image.

Supports two rendering modes:
  - Default (2D overlay): draws trajectory lines on saved images with OpenCV.
  - ManiSkill (--maniskill): replays trajectories in the 3D ManiSkill
    environment by teleporting pieces frame-by-frame and capturing top-view
    renders.  Produces side-by-side GT | Predicted videos.

Usage:
    python scripts/eval.py \
        --data_dir assets/data \
        --checkpoint_dir checkpoints \
        --output_dir eval_output \
        --max_episodes 20 \
        --fps 8

    # ManiSkill rendering
    python scripts/eval.py --maniskill --image_size 512

Output structure:
    eval_output/
      unet_train.mp4
      unet_val.mp4
      dit_train.mp4
      dit_val.mp4
      regression_train.mp4
      regression_val.mp4
      metrics.txt
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from visplan.training.dataset import VoronoiReassemblyDataset
from visplan.training.models.diffusion_unet import DiffusionUNet
from visplan.training.models.diffusion_transformer import DiffusionTransformer
from visplan.training.models.regression_baseline import RegressionBaseline

# Reuse rendering helpers from data generation
from scripts.data_gen.generate_voronoi_reassembly_data import (
    render_overhead, world_to_pixel, _piece_colors,
)


MODEL_REGISTRY = {
    "unet": DiffusionUNet,
    "dit": DiffusionTransformer,
    "regression": RegressionBaseline,
}


# ------------------------------------------------------------------ #
#  Denormalization
# ------------------------------------------------------------------ #

def denormalize_traj(traj_norm, visible_range=0.6):
    """Convert normalised (T, 4) [x̃, ỹ, cos θ, sin θ] → (T, 3) [x, y, θ].

    Args:
        traj_norm: (T, 4) or (N, T, 4) numpy array
    Returns:
        (T, 3) or (N, T, 3) numpy array in world coordinates
    """
    xy = traj_norm[..., :2] * visible_range
    theta = np.arctan2(traj_norm[..., 3], traj_norm[..., 2])
    return np.concatenate([xy, theta[..., None]], axis=-1).astype(np.float32)


# ------------------------------------------------------------------ #
#  Per-episode rendering
# ------------------------------------------------------------------ #

def render_trajectory_frame(start_image, outlines, start_poses, visible_range,
                            gt_waypoints, pred_waypoints, timestep, colors,
                            image_size=256):
    """Render one frame showing pieces at start + trajectory progress.

    Returns an (H, 2W, 3) uint8 image: [GT overlay | Pred overlay].

    Args:
        start_image:   (H, W, 3) uint8
        outlines:      list of (V_i, 2) polygon outlines
        start_poses:   (N, 3) start SE(2) poses
        gt_waypoints:  (N, T, 3) ground-truth SE(2) trajectories
        pred_waypoints:(N, T, 3) predicted SE(2) trajectories
        timestep:      current timestep to highlight
        colors:        list of RGB tuples for each piece
    """
    N = len(outlines)
    T = gt_waypoints.shape[1]

    def _draw_overlay(base, waypoints, label):
        img = base.copy()
        for i in range(N):
            color = colors[i]
            pts = waypoints[i, :timestep + 1, :2]  # up to current step
            if len(pts) < 1:
                continue
            px = world_to_pixel(pts, image_size, visible_range)

            # Draw trajectory line
            for j in range(len(px) - 1):
                cv2.line(img, tuple(px[j]), tuple(px[j + 1]), color, 2)

            # Start marker (blue circle)
            start_px = world_to_pixel(
                waypoints[i, 0:1, :2], image_size, visible_range)
            cv2.circle(img, tuple(start_px[0]), 5, (255, 0, 0), -1)

            # Current position marker (filled)
            cur_px = world_to_pixel(
                waypoints[i, timestep:timestep + 1, :2], image_size,
                visible_range)
            cv2.circle(img, tuple(cur_px[0]), 5, color, -1)

            # Goal marker (green diamond)
            goal_px = world_to_pixel(
                waypoints[i, -1:, :2], image_size, visible_range)
            cv2.drawMarker(img, tuple(goal_px[0]), (0, 200, 0),
                           cv2.MARKER_DIAMOND, 8, 2)

        # Label
        cv2.putText(img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2)
        cv2.putText(img, f"t={timestep}/{T-1}", (5, image_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img

    gt_frame = _draw_overlay(start_image, gt_waypoints, "Ground Truth")
    pred_frame = _draw_overlay(start_image, pred_waypoints, "Predicted")

    return np.concatenate([gt_frame, pred_frame], axis=1)


def render_episode_summary(start_image, goal_image, outlines, start_poses,
                           gt_traj, pred_traj, colors, visible_range,
                           image_size=256):
    """Render a static summary frame: start | goal | GT full traj | pred full traj."""
    T = gt_traj.shape[1]

    def _draw_full_traj(base, waypoints, label):
        img = base.copy()
        N = len(outlines)
        for i in range(N):
            color = colors[i]
            pts = waypoints[i, :, :2]
            px = world_to_pixel(pts, image_size, visible_range)
            for j in range(len(px) - 1):
                cv2.line(img, tuple(px[j]), tuple(px[j + 1]), color, 2)
            cv2.circle(img, tuple(px[0]), 4, (255, 0, 0), -1)
            cv2.drawMarker(img, tuple(px[-1]), (0, 200, 0),
                           cv2.MARKER_DIAMOND, 8, 2)
        cv2.putText(img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2)
        return img

    gt_full = _draw_full_traj(start_image, gt_traj, "GT Trajectories")
    pred_full = _draw_full_traj(start_image, pred_traj, "Pred Trajectories")

    # Label the start/goal
    si = start_image.copy()
    cv2.putText(si, "Start", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 2)
    gi = goal_image.copy()
    cv2.putText(gi, "Goal", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 2)

    return np.concatenate([si, gi, gt_full, pred_full], axis=1)


# ------------------------------------------------------------------ #
#  Prediction helpers
# ------------------------------------------------------------------ #

def predict_episode(model, model_name, episode_data, visible_range, device,
                    use_ddim=True, ddim_steps=20):
    """Run model inference for all pieces in one episode.

    Args:
        episode_data: loaded .npz dict
        Returns: (N, T, 3) predicted trajectories in world SE(2) coords
    """
    N = int(episode_data["num_pieces"])
    start_img = episode_data["start_image"].astype(np.float32) / 255.0
    start_img = np.transpose(start_img, (2, 0, 1))  # (3, H, W)
    goal_img = episode_data["goal_image"].astype(np.float32) / 255.0
    goal_img = np.transpose(goal_img, (2, 0, 1))

    all_pred = []
    for pi in range(N):
        mask = episode_data["piece_masks"][pi].astype(np.float32)[None]  # (1, H, W)

        batch = {
            "start_image": torch.from_numpy(start_img).unsqueeze(0).to(device),
            "goal_image": torch.from_numpy(goal_img).unsqueeze(0).to(device),
            "piece_mask": torch.from_numpy(mask).unsqueeze(0).to(device),
        }

        with torch.no_grad():
            if model_name == "regression":
                pred = model.sample(batch)
            else:
                if use_ddim:
                    pred = model.sample(batch, device=device, use_ddim=True,
                                        ddim_steps=ddim_steps)
                else:
                    pred = model.sample(batch, device=device, use_ddim=False)

        pred_np = pred[0].cpu().numpy()  # (T, 4) normalised
        all_pred.append(pred_np)

    pred_traj_norm = np.stack(all_pred, axis=0)  # (N, T, 4)
    return denormalize_traj(pred_traj_norm, visible_range)


# ------------------------------------------------------------------ #
#  Video generation
# ------------------------------------------------------------------ #

def generate_video(model, model_name, episode_files, output_path,
                   visible_range, device, max_episodes=20, fps=8,
                   use_ddim=True, ddim_steps=20):
    """Generate a comparison video for a list of episode files.

    Video layout per episode:
      1. Summary frame (held for 1 second): start | goal | GT traj | pred traj
      2. Animated frames stepping through timesteps: GT overlay | pred overlay
    """
    image_size = 256
    frame_w = image_size * 2  # GT side-by-side with pred for animated frames
    frame_h = image_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Use summary width (4 panels) as the video width
    video_w = image_size * 4
    video_h = image_size

    writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))

    episodes_used = min(max_episodes, len(episode_files))
    metrics_xy = []
    metrics_theta = []
    metrics_final = []

    for ei in range(episodes_used):
        d = np.load(episode_files[ei])
        N = int(d["num_pieces"])
        T = d["trajectories"].shape[1]
        voronoi_seed = int(d["voronoi_seed"])
        colors = _piece_colors(N, seed=voronoi_seed)

        # Get outlines
        outlines = [d[f"outline_{i}"] for i in range(N)]

        gt_traj = d["trajectories"]          # (N, T, 3)
        start_poses = d["start_poses"]       # (N, 3)
        start_image = d["start_image"]       # (H, W, 3)
        goal_image = d["goal_image"]         # (H, W, 3)

        # Run prediction
        pred_traj = predict_episode(model, model_name, d, visible_range,
                                    device, use_ddim=use_ddim,
                                    ddim_steps=ddim_steps)

        # Compute metrics for this episode
        xy_err = np.sqrt(((gt_traj[:, :, :2] - pred_traj[:, :, :2]) ** 2).sum(-1)).mean()
        theta_err = np.abs(
            np.arctan2(np.sin(gt_traj[:, :, 2] - pred_traj[:, :, 2]),
                       np.cos(gt_traj[:, :, 2] - pred_traj[:, :, 2]))
        ).mean()
        final_xy_err = np.sqrt(
            ((gt_traj[:, -1, :2] - pred_traj[:, -1, :2]) ** 2).sum(-1)).mean()
        metrics_xy.append(xy_err)
        metrics_theta.append(theta_err)
        metrics_final.append(final_xy_err)

        # 1) Summary frame (hold for 1 second)
        summary = render_episode_summary(
            start_image, goal_image, outlines, start_poses,
            gt_traj, pred_traj, colors, visible_range, image_size)
        for _ in range(fps):  # hold for 1 second
            writer.write(summary)

        # 2) Animated trajectory frames
        for t in range(T):
            frame = render_trajectory_frame(
                start_image, outlines, start_poses, visible_range,
                gt_traj, pred_traj, t, colors, image_size)
            # Pad to match video width (4 panels → add padding on right)
            padded = np.full((video_h, video_w, 3), 200, dtype=np.uint8)
            padded[:, :frame_w] = frame
            # Add episode info on the padding area
            cv2.putText(padded, f"Ep {ei}", (frame_w + 10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(padded, f"xy err: {xy_err:.4f}m",
                        (frame_w + 10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(padded, f"theta err: {np.degrees(theta_err):.1f}deg",
                        (frame_w + 10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(padded, f"final xy: {final_xy_err:.4f}m",
                        (frame_w + 10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            writer.write(padded)

        # Hold last frame briefly
        for _ in range(fps // 2):
            writer.write(padded)

        d.close()

    writer.release()

    return {
        "mean_xy_error": float(np.mean(metrics_xy)),
        "mean_theta_error_deg": float(np.degrees(np.mean(metrics_theta))),
        "mean_final_xy_error": float(np.mean(metrics_final)),
        "num_episodes": episodes_used,
    }


# ------------------------------------------------------------------ #
#  ManiSkill trajectory replay
# ------------------------------------------------------------------ #

def se2_to_pose7d(x, y, theta, z=0.016):
    """Convert SE(2) pose [x, y, θ] to ManiSkill 7D pose [x, y, z, qw, qx, qy, qz]."""
    quat_xyzw = Rotation.from_euler("xyz", [0, 0, theta]).as_quat()  # (x, y, z, w)
    qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
    return np.array([x, y, z, qw, qx, qy, qz], dtype=np.float32)


def _set_all_piece_poses(env, poses_se2, z=0.016):
    """Teleport all pieces to SE(2) poses.

    Args:
        env: unwrapped VoronoiReassembly env
        poses_se2: (N, 3) array of [x, y, θ]
        z: height to place pieces at
    """
    for i, name in enumerate(env.piece_names):
        pose7d = se2_to_pose7d(poses_se2[i, 0], poses_se2[i, 1],
                               poses_se2[i, 2], z=z)
        env.set_object_pose(name, torch.tensor(pose7d, device=env.device))


def _create_maniskill_env(episode_data, render_mode="human"):
    """Create a ManiSkill env matching an episode's Voronoi configuration."""
    import gymnasium as gym
    import visplan.voronoi_env  # noqa: F401 — triggers env registration

    num_points = int(episode_data["num_voronoi_points"])
    voronoi_seed = int(episode_data["voronoi_seed"])
    side_length = float(episode_data["side_length"])

    env = gym.make(
        "VoronoiReassembly-v1",
        parallel_in_single_scene=False,
        num_envs=1,
        viewer_camera_configs=dict(shader_pack="rt-fast"),
        render_mode=render_mode,
        num_voronoi_points=num_points,
        side_length=side_length,
        placement_mode="assembled",
        voronoi_seed=voronoi_seed,
    )
    env.reset()
    return env


def _replay_trajectory(env, trajectory, image_size=256):
    """Replay an (N, T, 3) SE(2) trajectory and capture frames.

    Returns:
        List of (image_size, image_size, 3) uint8 frames, one per timestep.
    """
    uw = env.unwrapped
    z = uw.extrusion_height / 2 + 0.001
    N, T, _ = trajectory.shape
    frames = []
    for t in range(T):
        _set_all_piece_poses(uw, trajectory[:, t, :], z=z)
        if env.render_mode is not None:
            env.render()
        frames.append(uw.get_top_view(image_size=image_size))
    return frames


def generate_maniskill_video(model, model_name, episode_files, output_path,
                             visible_range, device, max_episodes=20, fps=8,
                             image_size=256, use_ddim=True, ddim_steps=20,
                             render_mode="human"):
    """Generate a GT-vs-predicted video by replaying trajectories in ManiSkill.

    Video layout per episode:
      1. Summary row (held 1 s): GT-start | GT-goal | label
      2. Animated side-by-side: GT replay | Predicted replay
    """
    video_w = image_size * 2
    video_h = image_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))

    episodes_used = min(max_episodes, len(episode_files))
    metrics_xy, metrics_theta, metrics_final = [], [], []

    env = None
    prev_voronoi_seed = None
    prev_num_points = None

    for ei in range(episodes_used):
        d = np.load(episode_files[ei])
        N = int(d["num_pieces"])
        T = d["trajectories"].shape[1]
        voronoi_seed = int(d["voronoi_seed"])
        num_points = int(d["num_voronoi_points"])

        gt_traj = d["trajectories"]       # (N, T, 3)

        # Run prediction
        pred_traj = predict_episode(model, model_name, d, visible_range,
                                    device, use_ddim=use_ddim,
                                    ddim_steps=ddim_steps)

        # Compute metrics
        xy_err = np.sqrt(((gt_traj[:, :, :2] - pred_traj[:, :, :2]) ** 2).sum(-1)).mean()
        theta_err = np.abs(
            np.arctan2(np.sin(gt_traj[:, :, 2] - pred_traj[:, :, 2]),
                       np.cos(gt_traj[:, :, 2] - pred_traj[:, :, 2]))
        ).mean()
        final_xy_err = np.sqrt(
            ((gt_traj[:, -1, :2] - pred_traj[:, -1, :2]) ** 2).sum(-1)).mean()
        metrics_xy.append(xy_err)
        metrics_theta.append(theta_err)
        metrics_final.append(final_xy_err)

        # (Re)create env only when voronoi config changes
        if (env is None or voronoi_seed != prev_voronoi_seed
                or num_points != prev_num_points):
            if env is not None:
                env.close()
            env = _create_maniskill_env(d, render_mode=render_mode)
            prev_voronoi_seed = voronoi_seed
            prev_num_points = num_points

        # Replay GT and predicted trajectories
        gt_frames = _replay_trajectory(env, gt_traj, image_size=image_size)
        pred_frames = _replay_trajectory(env, pred_traj, image_size=image_size)

        # 1) Summary frame: GT-start | GT-goal (held for 1 second)
        summary = np.concatenate([gt_frames[0], gt_frames[-1]], axis=1)
        cv2.putText(summary, "GT Start", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(summary, "GT Goal", (image_size + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(summary, f"Ep {ei}  xy={xy_err:.4f}m  "
                    f"th={np.degrees(theta_err):.1f}deg", (5, image_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        for _ in range(fps):
            writer.write(cv2.cvtColor(summary, cv2.COLOR_RGB2BGR))

        # 2) Animated side-by-side: GT | Predicted
        for t in range(T):
            gt_f = gt_frames[t].copy()
            pred_f = pred_frames[t].copy()

            cv2.putText(gt_f, "Ground Truth", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(pred_f, "Predicted", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            label = f"t={t}/{T-1}"
            cv2.putText(gt_f, label, (5, image_size - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(pred_f, label, (5, image_size - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            frame = np.concatenate([gt_f, pred_f], axis=1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Hold last frame briefly
        for _ in range(fps // 2):
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        d.close()
        print(f"  Episode {ei+1}/{episodes_used}  "
              f"xy={xy_err:.4f}m  θ={np.degrees(theta_err):.1f}°  "
              f"final_xy={final_xy_err:.4f}m")

    if env is not None:
        env.close()
    writer.release()

    return {
        "mean_xy_error": float(np.mean(metrics_xy)),
        "mean_theta_error_deg": float(np.degrees(np.mean(metrics_theta))),
        "mean_final_xy_error": float(np.mean(metrics_final)),
        "num_episodes": episodes_used,
    }


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", type=str, default="assets/data")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--output_dir", type=str, default="eval_output")
    p.add_argument("--models", nargs="+",
                   default=["unet", "dit", "regression"],
                   choices=list(MODEL_REGISTRY))
    p.add_argument("--max_episodes", type=int, default=20,
                   help="Max episodes per split to include in video.")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--visible_range", type=float, default=0.6)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--traj_len", type=int, default=32)
    p.add_argument("--diffusion_steps", type=int, default=100)
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--no_ddim", action="store_true",
                   help="Use full DDPM sampling instead of DDIM.")
    p.add_argument("--maniskill", action="store_true",
                   help="Replay trajectories in ManiSkill 3D environment "
                        "instead of 2D overlay rendering.")
    p.add_argument("--image_size", type=int, default=256,
                   help="Resolution for ManiSkill top-view renders.")
    p.add_argument("--render_mode", type=str, default="human",
                   choices=["human", "rgb_array"],
                   help="ManiSkill render mode. 'human' opens a GUI viewer, "
                        "'rgb_array' runs headless.")
    return p.parse_args()


def get_episode_split(data_dir, val_frac=0.1):
    """Return (train_files, val_files) using the same split as training."""
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    n_val = max(1, int(len(files) * val_frac))
    n_train = len(files) - n_val
    return files[:n_train], files[n_train:]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_files, val_files = get_episode_split(args.data_dir, args.val_frac)
    print(f"Split: {len(train_files)} train, {len(val_files)} val episodes")

    all_metrics = {}

    for model_name in args.models:
        ckpt_path = os.path.join(args.checkpoint_dir, f"{model_name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"Skipping {model_name}: {ckpt_path} not found")
            continue

        # Build model
        model_cls = MODEL_REGISTRY[model_name]
        model_kwargs = dict(traj_dim=4, traj_len=args.traj_len,
                            embed_dim=args.embed_dim)
        if model_name in ("unet", "dit"):
            model_kwargs["num_diffusion_steps"] = args.diffusion_steps
        model = model_cls(**model_kwargs).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=True))
        model.eval()
        print(f"\nLoaded {model_name} from {ckpt_path}")

        use_ddim = not args.no_ddim and model_name != "regression"

        for split_name, files in [("train", train_files), ("val", val_files)]:
            suffix = "_maniskill" if args.maniskill else ""
            out_path = os.path.join(args.output_dir,
                                    f"{model_name}_{split_name}{suffix}.mp4")
            print(f"  Generating {out_path} ({min(args.max_episodes, len(files))} episodes)...")

            if args.maniskill:
                metrics = generate_maniskill_video(
                    model, model_name, files, out_path,
                    args.visible_range, device,
                    max_episodes=args.max_episodes, fps=args.fps,
                    image_size=args.image_size,
                    use_ddim=use_ddim, ddim_steps=args.ddim_steps,
                    render_mode=args.render_mode)
            else:
                metrics = generate_video(
                    model, model_name, files, out_path,
                    args.visible_range, device,
                    max_episodes=args.max_episodes, fps=args.fps,
                    use_ddim=use_ddim, ddim_steps=args.ddim_steps)

            key = f"{model_name}_{split_name}"
            all_metrics[key] = metrics
            print(f"    xy={metrics['mean_xy_error']:.4f}m  "
                  f"θ={metrics['mean_theta_error_deg']:.1f}°  "
                  f"final_xy={metrics['mean_final_xy_error']:.4f}m")

    # Save metrics summary
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"{'Model':20s} {'Split':6s} {'XY Err (m)':>12s} "
                f"{'θ Err (°)':>10s} {'Final XY (m)':>13s} {'Episodes':>9s}\n")
        f.write("-" * 75 + "\n")
        for key, m in all_metrics.items():
            model_name, split = key.rsplit("_", 1)
            f.write(f"{model_name:20s} {split:6s} "
                    f"{m['mean_xy_error']:12.4f} "
                    f"{m['mean_theta_error_deg']:10.1f} "
                    f"{m['mean_final_xy_error']:13.4f} "
                    f"{m['num_episodes']:9d}\n")
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
