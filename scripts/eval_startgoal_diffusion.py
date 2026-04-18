"""Inference, visualization, and metrics for the start/goal diffusion model.

The model takes only start + goal poses as input (no polygon) and produces
a clamped trajectory. The polygon is still drawn in visualizations so you
can see the piece that was moved.

Usage:
    python scripts/eval_startgoal_diffusion.py \
        --data_dir assets/push_data \
        --checkpoint checkpoints/startgoal_diffusion/best.pt \
        --num_samples 6 --video
"""

import argparse
import json
import os

import matplotlib.animation as manim
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon as MplPolygon

from visplan.training.push_dataset import PushTrajectoryDataset
from visplan.training.models.startgoal_diffusion import StartGoalDiffusionUNet


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def denormalize_trajectory(traj_norm, xy_norm):
    xy = traj_norm[..., :2] * xy_norm
    theta = np.arctan2(traj_norm[..., 3], traj_norm[..., 2])
    return np.concatenate([xy, theta[..., None]], axis=-1)


def polygon_world(verts_local, pose):
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ verts_local.T).T + np.array([x, y])


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def per_sample_metrics(gt, pr):
    dxy = np.linalg.norm(gt[:, :2] - pr[:, :2], axis=1)
    dth = np.abs(wrap_angle(gt[:, 2] - pr[:, 2]))
    return {"xy_mean": dxy.mean(), "xy_final": dxy[-1],
            "theta_mean": dth.mean(), "theta_final": dth[-1]}


# ------------------------------------------------------------------ #
#  Plotting
# ------------------------------------------------------------------ #

def draw_polygon_outline(ax, verts_local, pose, color, linestyle="-",
                         linewidth=1.2, alpha=0.9, label=None):
    ax.add_patch(MplPolygon(
        polygon_world(verts_local, pose), closed=True, facecolor="none",
        edgecolor=color, linewidth=linewidth, alpha=alpha,
        linestyle=linestyle, label=label))


def plot_trajectory(ax, polygon_verts, poses, color, label=None,
                    alpha_range=(0.08, 0.85)):
    T = len(poses)
    draw_every = max(1, T // 10)
    for t in range(0, T, draw_every):
        alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * t / max(T - 1, 1)
        ax.add_patch(MplPolygon(polygon_world(polygon_verts, poses[t]),
                                closed=True, facecolor=color,
                                edgecolor="black", linewidth=0.3, alpha=alpha))
    ax.add_patch(MplPolygon(polygon_world(polygon_verts, poses[-1]),
                            closed=True, facecolor=color,
                            edgecolor="black", linewidth=0.8, alpha=0.95))
    ax.plot(poses[:, 0], poses[:, 1], "-", color=color, linewidth=1.2,
            alpha=0.7, label=label)
    ax.plot(poses[0, 0], poses[0, 1], "o", color="white", markersize=5,
            markeredgecolor=color, markeredgewidth=1.5, zorder=5)
    ax.plot(poses[-1, 0], poses[-1, 1], "s", color=color, markersize=6, zorder=5)


def plot_sample(ax, polygon_verts, gt_poses, pred_poses, start_pose, goal_pose,
                title):
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    draw_polygon_outline(ax, polygon_verts, start_pose, color="black",
                         linewidth=1.6, alpha=0.9, label="start (input)")
    draw_polygon_outline(ax, polygon_verts, goal_pose, color="goldenrod",
                         linestyle="--", linewidth=1.6, alpha=0.9,
                         label="goal (input)")
    plot_trajectory(ax, polygon_verts, gt_poses, color="tab:green", label="GT")
    plot_trajectory(ax, polygon_verts, pred_poses, color="tab:red", label="Pred")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="best")
    ax.autoscale_view()
    m = 0.03
    xl, xr = ax.get_xlim(); yl, yr = ax.get_ylim()
    ax.set_xlim(xl - m, xr + m); ax.set_ylim(yl - m, yr + m)


def make_video(polygon_verts, gt_poses, pred_poses, start_pose, goal_pose,
               save_path, fps=15, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, name, color in zip(axes, ["Ground truth", "Predicted"],
                               ["tab:green", "tab:red"]):
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(name, fontsize=11, color=color)

    all_xy = np.concatenate([gt_poses[:, :2], pred_poses[:, :2],
                             goal_pose[None, :2]], axis=0)
    radius = np.max(np.linalg.norm(polygon_verts, axis=1))
    x_lo, x_hi = all_xy[:, 0].min() - radius - 0.03, all_xy[:, 0].max() + radius + 0.03
    y_lo, y_hi = all_xy[:, 1].min() - radius - 0.03, all_xy[:, 1].max() + radius + 0.03
    for ax, poses, color in zip(axes, [gt_poses, pred_poses],
                                ["tab:green", "tab:red"]):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        draw_polygon_outline(ax, polygon_verts, start_pose, color="black",
                             linewidth=1.4, alpha=0.7)
        draw_polygon_outline(ax, polygon_verts, goal_pose, color="goldenrod",
                             linestyle="--", linewidth=1.6, alpha=0.9)
        ax.plot(poses[:, 0], poses[:, 1], "-", color=color,
                linewidth=1.0, alpha=0.5)

    T = min(len(gt_poses), len(pred_poses))
    patches = [None, None]
    traces = [None, None]

    def init():
        for i, (ax, poses, color) in enumerate(zip(axes, [gt_poses, pred_poses],
                                                   ["tab:green", "tab:red"])):
            world = polygon_world(polygon_verts, poses[0])
            patches[i] = MplPolygon(world, closed=True, facecolor=color,
                                    edgecolor="black", linewidth=0.8, alpha=0.9)
            ax.add_patch(patches[i])
            traces[i], = ax.plot([poses[0, 0]], [poses[0, 1]], "-",
                                 color=color, linewidth=2.0, alpha=0.9)
        return patches + traces

    def update(t):
        for i, poses in enumerate([gt_poses, pred_poses]):
            patches[i].set_xy(polygon_world(polygon_verts, poses[t]))
            traces[i].set_data(poses[:t + 1, 0], poses[:t + 1, 1])
        return patches + traces

    if title:
        fig.suptitle(title, fontsize=11)

    anim = manim.FuncAnimation(fig, update, frames=T, init_func=init,
                               blit=True, interval=1000 / fps)
    ext = os.path.splitext(save_path)[1].lower()
    writer = (manim.FFMpegWriter(fps=fps, bitrate=2400)
              if ext in (".mp4", ".mov")
              else manim.PillowWriter(fps=fps))
    anim.save(save_path, writer=writer, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Inference
# ------------------------------------------------------------------ #

@torch.no_grad()
def run_inference(model, dataset, indices, xy_norm, device, args,
                  batch_size=64):
    results = []
    for b in range(0, len(indices), batch_size):
        chunk = indices[b:b + batch_size]
        samples = [dataset[i] for i in chunk]
        batch = {
            "start_pose":       torch.stack([s["start_pose"] for s in samples]).to(device),
            "goal_pose":        torch.stack([s["goal_pose"] for s in samples]).to(device),
            "trajectory_mask":  torch.stack([s["trajectory_mask"] for s in samples]).to(device),
        }
        pred = model.sample(batch, device=device,
                            use_ddim=args.use_ddim,
                            ddim_steps=args.ddim_steps).cpu().numpy()

        for s, p in zip(samples, pred):
            T = int(s["traj_len"])
            V = int(s["polygon_mask"].sum().item())
            verts = s["polygon_vertices"].numpy()[:V]
            gt = denormalize_trajectory(s["trajectory"].numpy()[:T], xy_norm)
            pr = denormalize_trajectory(p[:T], xy_norm)
            start = denormalize_trajectory(s["start_pose"].numpy()[None], xy_norm)[0]
            goal  = denormalize_trajectory(s["goal_pose"].numpy()[None],  xy_norm)[0]
            results.append({
                "verts": verts, "gt": gt, "pr": pr,
                "start": start, "goal": goal, "T": T,
            })
    return results


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--metrics_samples", type=int, default=512)
    p.add_argument("--diffusion_steps", type=int, default=100)
    p.add_argument("--use_ddim", action="store_true")
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="startgoal_diffusion_eval.png")
    p.add_argument("--metrics_save", type=str,
                   default="startgoal_diffusion_metrics.png")
    p.add_argument("--video", action="store_true")
    p.add_argument("--video_dir", type=str, default="startgoal_diffusion_videos")
    p.add_argument("--video_format", choices=["mp4", "gif"], default="mp4")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--goal_xy_thresh", type=float, default=0.03)
    p.add_argument("--goal_theta_thresh_deg", type=float, default=15.0)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    traj_len = meta["max_steps"] + 1
    xy_norm = meta["start_pose_range"]

    ds = PushTrajectoryDataset(args.data_dir, augment=False)
    n_files = len(ds.files)
    n_val = max(1, int(n_files * args.val_frac))
    n_train = n_files - n_val
    val_indices = [i for i, (fi, _) in enumerate(ds.index) if fi >= n_train]
    if not val_indices:
        val_indices = list(range(len(ds)))
    print(f"Val pool: {len(val_indices)} samples")

    model = StartGoalDiffusionUNet(
        traj_dim=4, traj_len=traj_len,
        num_diffusion_steps=args.diffusion_steps,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Metrics ---
    n_metric = min(args.metrics_samples, len(val_indices))
    metric_idx = list(rng.choice(val_indices, size=n_metric, replace=False))
    print(f"\nComputing metrics over {n_metric} val samples "
          f"(ddim={args.use_ddim})...")
    metric_results = run_inference(model, ds, metric_idx, xy_norm, device,
                                   args, batch_size=args.batch_size)

    xy_mean, xy_final, theta_mean, theta_final = [], [], [], []
    goal_xy_err, goal_theta_err = [], []
    for r in metric_results:
        m = per_sample_metrics(r["gt"], r["pr"])
        xy_mean.append(m["xy_mean"])
        xy_final.append(m["xy_final"])
        theta_mean.append(m["theta_mean"])
        theta_final.append(m["theta_final"])
        goal_xy_err.append(np.linalg.norm(r["pr"][-1, :2] - r["goal"][:2]))
        goal_theta_err.append(abs(wrap_angle(r["pr"][-1, 2] - r["goal"][2])))

    xy_mean = np.array(xy_mean); xy_final = np.array(xy_final)
    theta_mean = np.array(theta_mean); theta_final = np.array(theta_final)
    goal_xy_err = np.array(goal_xy_err); goal_theta_err = np.array(goal_theta_err)
    success = ((goal_xy_err < args.goal_xy_thresh) &
               (goal_theta_err < np.deg2rad(args.goal_theta_thresh_deg)))

    print("\n=== Aggregate metrics (vs ground truth) ===")
    print(f"  xy error    — mean {xy_mean.mean():.4f}  median {np.median(xy_mean):.4f}  "
          f"p95 {np.percentile(xy_mean, 95):.4f}  (m)")
    print(f"  final xy    — mean {xy_final.mean():.4f}  median {np.median(xy_final):.4f}  "
          f"p95 {np.percentile(xy_final, 95):.4f}  (m)")
    print(f"  θ error     — mean {np.degrees(theta_mean.mean()):.2f}  "
          f"median {np.degrees(np.median(theta_mean)):.2f}  "
          f"p95 {np.degrees(np.percentile(theta_mean, 95)):.2f}  (°)")
    print(f"  final θ     — mean {np.degrees(theta_final.mean()):.2f}  "
          f"median {np.degrees(np.median(theta_final)):.2f}  "
          f"p95 {np.degrees(np.percentile(theta_final, 95)):.2f}  (°)")
    print("\n=== Goal-reaching metrics (pred_final vs input goal) ===")
    print(f"  goal xy  — mean {goal_xy_err.mean():.4f}  median {np.median(goal_xy_err):.4f}  (m)")
    print(f"  goal θ   — mean {np.degrees(goal_theta_err.mean()):.2f}  "
          f"median {np.degrees(np.median(goal_theta_err)):.2f}  (°)")
    print(f"  success (xy<{args.goal_xy_thresh}m & θ<{args.goal_theta_thresh_deg}°): "
          f"{success.mean() * 100:.1f}%  ({success.sum()}/{len(success)})")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].hist(xy_mean, bins=40, color="tab:blue", alpha=0.8)
    axes[0, 0].set_title(f"Mean xy error per traj (m)  μ={xy_mean.mean():.3f}")
    axes[0, 1].hist(goal_xy_err, bins=40, color="tab:orange", alpha=0.8)
    axes[0, 1].axvline(args.goal_xy_thresh, color="red", linestyle="--",
                       label=f"thresh {args.goal_xy_thresh}")
    axes[0, 1].set_title(f"Final xy vs goal (m)  μ={goal_xy_err.mean():.3f}")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].hist(np.degrees(theta_mean), bins=40, color="tab:green", alpha=0.8)
    axes[1, 0].set_title(f"Mean θ error per traj (°)  μ={np.degrees(theta_mean.mean()):.2f}")
    axes[1, 1].hist(np.degrees(goal_theta_err), bins=40, color="tab:red", alpha=0.8)
    axes[1, 1].axvline(args.goal_theta_thresh_deg, color="red", linestyle="--",
                       label=f"thresh {args.goal_theta_thresh_deg}°")
    axes[1, 1].set_title(f"Final θ vs goal (°)  μ={np.degrees(goal_theta_err.mean()):.2f}")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"StartGoal diffusion — val metrics over {n_metric} samples  "
        f"(ddim={args.use_ddim})  "
        f"success={success.mean() * 100:.1f}%", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.metrics_save, dpi=140)
    plt.close(fig)
    print(f"\nSaved metrics histogram → {args.metrics_save}")

    # --- Static visualisation ---
    n_vis = min(args.num_samples, len(val_indices))
    vis_idx = list(rng.choice(val_indices, size=n_vis, replace=False))
    vis_results = run_inference(model, ds, vis_idx, xy_norm, device, args,
                                batch_size=args.batch_size)

    cols = min(3, n_vis)
    rows = (n_vis + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax_i, r in enumerate(vis_results):
        m = per_sample_metrics(r["gt"], r["pr"])
        gxy = np.linalg.norm(r["pr"][-1, :2] - r["goal"][:2])
        title = (f"T={r['T']}  xy={m['xy_mean']:.3f}  "
                 f"finalXY={m['xy_final']:.3f}  goalXY={gxy:.3f}  "
                 f"θ={np.degrees(m['theta_mean']):.1f}°")
        plot_sample(axes[ax_i], r["verts"], r["gt"], r["pr"],
                    r["start"], r["goal"], title)
    for k in range(n_vis, len(axes)):
        axes[k].axis("off")
    fig.suptitle(
        f"StartGoal diffusion  ddim={args.use_ddim}  "
        f"mean_xy={xy_mean.mean():.3f}m  "
        f"goal_xy={goal_xy_err.mean():.3f}m  "
        f"θ={np.degrees(theta_mean.mean()):.1f}°  "
        f"success={success.mean() * 100:.1f}%", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.save, dpi=150)
    plt.close(fig)
    print(f"Saved static figure → {args.save}")

    # --- Videos ---
    if args.video:
        os.makedirs(args.video_dir, exist_ok=True)
        print(f"\nRendering {len(vis_results)} videos → {args.video_dir}/ ...")
        for i, r in enumerate(vis_results):
            m = per_sample_metrics(r["gt"], r["pr"])
            gxy = np.linalg.norm(r["pr"][-1, :2] - r["goal"][:2])
            title = (f"sample {i}  T={r['T']}  xy={m['xy_mean']:.3f}m  "
                     f"goalXY={gxy:.3f}m  θ={np.degrees(m['theta_mean']):.1f}°")
            path = os.path.join(args.video_dir,
                                f"sample_{i:02d}.{args.video_format}")
            try:
                make_video(r["verts"], r["gt"], r["pr"], r["start"], r["goal"],
                           path, fps=args.fps, title=title)
                print(f"  wrote {path}")
            except Exception as e:
                print(f"  [warn] failed to write {path}: {e}")


if __name__ == "__main__":
    main()
