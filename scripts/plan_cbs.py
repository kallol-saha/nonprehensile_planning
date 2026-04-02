"""CBS / MMD-PP multi-piece trajectory planning script.

Loads a trained diffusion model and one or more saved episode .npz files,
runs CBS or MMD-PP to produce collision-aware reassembly trajectories, and
saves both a visualisation video and a .npz of the planned trajectories.

Output (per episode)
--------------------
  <output_dir>/
    episode_XXXXXX_cbs.npz          – planned trajectories + metadata
    episode_XXXXXX_cbs.mp4          – side-by-side: unconstrained | CBS planned
    planning_summary.txt            – aggregate metrics across all episodes

.npz contents
-------------
  planned_trajectories  (N, T, 3)  float32  world SE(2) [x, y, theta]
  unconstrained_trajs   (N, T, 3)  float32  unconstrained baseline (no CBS)
  gt_trajectories       (N, T, 3)  float32  ground-truth from episode file
  start_poses           (N, 3)     float32  [x, y, theta]
  goal_poses            (N, 3)     float32  [x, y, theta]
  num_pieces            scalar     int32
  num_conflicts_planned scalar     int32    conflicts in planned trajectories
  num_conflicts_unconstrained scalar int32  conflicts in unconstrained baseline
  nodes_expanded        scalar     int32    CBS nodes expanded (0 for PP)
  planning_time_s       scalar     float32

Usage
-----
    python scripts/plan_cbs.py \\
        --data_dir assets/data \\
        --checkpoint checkpoints/dit_best.pt \\
        --output_dir plan_output \\
        --model dit \\
        --algorithm cbs \\
        --max_episodes 10 \\
        --batch_size 8 \\
        --guidance_weight 0.2 \\
        --max_nodes 1000 \\
        --save_viz

    # MMD-PP instead of CBS:
    python scripts/plan_cbs.py ... --algorithm pp
"""

from __future__ import annotations

import argparse
import glob
import logging
import os

import cv2
import numpy as np
import torch

from visplan.planning.cbs import run_cbs, run_pp
from visplan.planning.low_level_planner import LowLevelPlanner
from visplan.training.models.diffusion_transformer import DiffusionTransformer
from visplan.training.models.diffusion_unet import DiffusionUNet


# ---------------------------------------------------------------------------
#  Rendering helpers (duplicated here to avoid importing from non-package scripts/)
# ---------------------------------------------------------------------------

def world_to_pixel(
    xy: np.ndarray, image_size: int, visible_range: float
) -> np.ndarray:
    """Map world (x, y) → pixel (u, v) for an overhead orthographic camera.

    World +x → image right, world +y → image up.  Camera is centred at the
    origin and ``visible_range`` is the half-extent in metres.
    """
    u = (xy[:, 0] + visible_range) / (2.0 * visible_range) * image_size
    v = (visible_range - xy[:, 1]) / (2.0 * visible_range) * image_size
    return np.stack([u, v], axis=-1).astype(np.int32)


def _piece_colors(n: int, seed: int = 0) -> list:
    """Return n distinct RGB colours as a list of 3-tuples (uint8)."""
    rng = np.random.RandomState(seed + 123)
    return [tuple(int(v) for v in row) for row in rng.randint(80, 220, size=(n, 3))]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_REGISTRY = {
    "unet": DiffusionUNet,
    "dit": DiffusionTransformer,
}


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CBS / MMD-PP multi-piece trajectory planning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing episode_XXXXXX.npz files.")
    p.add_argument("--max_episodes", type=int, default=10,
                   help="Number of episodes to plan.")
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="Fraction of episodes reserved for validation "
                        "(same split as training).")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "all"],
                   help="Which data split to evaluate on.")
    # Model
    p.add_argument("--model", type=str, default="dit",
                   choices=list(MODEL_REGISTRY),
                   help="Model architecture to use as the low-level planner.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (.pt file).")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--traj_len", type=int, default=32)
    p.add_argument("--diffusion_steps", type=int, default=100)
    # Sampler
    p.add_argument("--use_ddim", action="store_true", default=True,
                   help="Use DDIM sampling (faster).")
    p.add_argument("--no_ddim", action="store_true",
                   help="Use full DDPM sampling instead of DDIM.")
    p.add_argument("--ddim_steps", type=int, default=20,
                   help="Number of DDIM denoising steps.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Trajectories sampled per planning call (B).")
    p.add_argument("--guidance_weight", type=float, default=0.2,
                   help="λ_c: scale of sphere-constraint gradient guidance.")
    # CBS
    p.add_argument("--algorithm", type=str, default="cbs",
                   choices=["cbs", "pp"],
                   help="Planning algorithm: 'cbs' (CBS) or 'pp' (MMD-PP).")
    p.add_argument("--max_nodes", type=int, default=1000,
                   help="Max CT nodes to expand before giving up (CBS only).")
    p.add_argument("--constraint_margin", type=float, default=1.2,
                   help="Multiplier on bounding radius for sphere constraint size.")
    p.add_argument("--constraint_epsilon", type=float, default=1.2,
                   help="Epsilon padding inside SphereConstraint cost function.")
    # Output
    p.add_argument("--output_dir", type=str, default="plan_output",
                   help="Directory for output .npz files and videos.")
    p.add_argument("--save_viz", action="store_true",
                   help="Generate comparison MP4 videos.")
    p.add_argument("--fps", type=int, default=8,
                   help="Video frame rate.")
    p.add_argument("--visible_range", type=float, default=0.6,
                   help="Camera half-extent in metres.")
    p.add_argument("--image_size", type=int, default=256,
                   help="Overhead image resolution.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for reproducibility.")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def get_episode_split(
    data_dir: str,
    val_frac: float,
    split: str,
) -> list[str]:
    """Return the episode file list for the requested split."""
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found in {data_dir}")
    n_val = max(1, int(len(files) * val_frac))
    n_train = len(files) - n_val
    if split == "train":
        return files[:n_train]
    elif split == "val":
        return files[n_train:]
    else:
        return files


def load_model(
    model_name: str,
    checkpoint: str,
    traj_len: int,
    embed_dim: int,
    diffusion_steps: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load and return a trained model on the given device."""
    cls = MODEL_REGISTRY[model_name]
    model = cls(
        traj_dim=4,
        traj_len=traj_len,
        embed_dim=embed_dim,
        num_diffusion_steps=diffusion_steps,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded %s from %s", model_name, checkpoint)
    return model


def denormalise_traj(traj_norm: np.ndarray, visible_range: float) -> np.ndarray:
    """(*, T, 4) normalised → (*, T, 3) world metres SE(2)."""
    xy = traj_norm[..., :2] * visible_range
    theta = np.arctan2(traj_norm[..., 3], traj_norm[..., 2])
    return np.concatenate([xy, theta[..., np.newaxis]], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
#  Per-episode unconstrained baseline
# ---------------------------------------------------------------------------

def run_unconstrained(
    episode_data: dict,
    planner: LowLevelPlanner,
) -> np.ndarray:
    """Sample one trajectory per piece with no constraints (baseline).

    Returns:
        (N, T, 3) world SE(2) trajectories.
    """
    from visplan.planning.cbs import _build_scene_batch

    N = int(episode_data["num_pieces"])
    trajs = []
    for i in range(N):
        scene_batch = _build_scene_batch(episode_data, i, planner.device)
        batch_world = planner.plan_piece(scene_batch, constraints=[])
        # Pick the first sample as the representative (no conflict info yet)
        trajs.append(batch_world[0])
    return np.stack(trajs, axis=0)  # (N, T, 3)


# ---------------------------------------------------------------------------
#  Visualisation
# ---------------------------------------------------------------------------

def render_comparison_video(
    episode_data: dict,
    outlines: list,
    unconstrained_trajs: np.ndarray,
    planned_trajs: np.ndarray,
    visible_range: float,
    image_size: int,
    output_path: str,
    fps: int,
) -> None:
    """Save a side-by-side video: unconstrained (left) vs CBS planned (right).

    Each frame shows both sets of trajectories overlaid on the start image,
    with the current timestep highlighted.  A summary frame (full trajectories)
    is prepended.

    Args:
        episode_data:        Raw episode dict (.npz).
        outlines:            List of N (V_i, 2) polygon outlines.
        unconstrained_trajs: (N, T, 3) unconstrained trajectories (world metres).
        planned_trajs:       (N, T, 3) CBS/PP-planned trajectories (world metres).
        visible_range:       Camera half-extent (metres).
        image_size:          Pixel resolution.
        output_path:         Output .mp4 path.
        fps:                 Video frame rate.
    """
    N = int(episode_data["num_pieces"])
    voronoi_seed = int(episode_data["voronoi_seed"])
    colors = _piece_colors(N, seed=voronoi_seed)

    start_image = episode_data["start_image"]   # (H, W, 3) uint8
    goal_image = episode_data["goal_image"]

    T = planned_trajs.shape[1]

    # Video dimensions: two panels side by side
    video_w = image_size * 2
    video_h = image_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))

    def _draw_all_trajs(base_img: np.ndarray, trajs: np.ndarray,
                        label: str, step: int | None = None) -> np.ndarray:
        """Draw trajectories up to ``step`` onto a copy of base_img."""
        img = base_img.copy()
        end_t = T if step is None else step + 1
        for i in range(N):
            color = colors[i]
            pts = trajs[i, :end_t, :2]  # (end_t, 2)
            px = world_to_pixel(pts, image_size, visible_range)
            for k in range(len(px) - 1):
                cv2.line(img, tuple(px[k]), tuple(px[k + 1]), color, 2)
            # Start marker (blue)
            start_px = world_to_pixel(trajs[i, 0:1, :2], image_size, visible_range)
            cv2.circle(img, tuple(start_px[0]), 4, (255, 0, 0), -1)
            # Goal marker (green diamond)
            goal_px = world_to_pixel(trajs[i, -1:, :2], image_size, visible_range)
            cv2.drawMarker(img, tuple(goal_px[0]), (0, 200, 0),
                           cv2.MARKER_DIAMOND, 8, 2)
            if step is not None and step < T:
                cur_px = world_to_pixel(
                    trajs[i, step:step + 1, :2], image_size, visible_range)
                cv2.circle(img, tuple(cur_px[0]), 5, color, -1)
        cv2.putText(img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 2)
        if step is not None:
            cv2.putText(img, f"t={step}/{T-1}", (5, image_size - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        return img

    # 1. Summary frame: full trajectories, held for 1 second
    left_summary = _draw_all_trajs(start_image, unconstrained_trajs, "Unconstrained")
    right_summary = _draw_all_trajs(start_image, planned_trajs, "CBS Planned")
    summary_frame = np.concatenate([left_summary, right_summary], axis=1)
    for _ in range(fps):
        writer.write(summary_frame)

    # 2. Animated frames
    for t in range(T):
        left = _draw_all_trajs(start_image, unconstrained_trajs,
                               "Unconstrained", step=t)
        right = _draw_all_trajs(start_image, planned_trajs,
                                "CBS Planned", step=t)
        frame = np.concatenate([left, right], axis=1)
        writer.write(frame)

    # Hold last frame briefly
    for _ in range(fps // 2):
        writer.write(frame)

    writer.release()


# ---------------------------------------------------------------------------
#  Save planned episode .npz
# ---------------------------------------------------------------------------

def save_planned_npz(
    episode_data: dict,
    planned_trajs: np.ndarray,
    unconstrained_trajs: np.ndarray,
    num_conflicts_planned: int,
    num_conflicts_unconstrained: int,
    nodes_expanded: int,
    planning_time_s: float,
    output_path: str,
) -> None:
    """Save planned trajectories and metadata to a compressed .npz file."""
    save_dict = dict(
        planned_trajectories=planned_trajs.astype(np.float32),
        unconstrained_trajs=unconstrained_trajs.astype(np.float32),
        gt_trajectories=episode_data["trajectories"].astype(np.float32),
        start_poses=episode_data["start_poses"].astype(np.float32),
        goal_poses=episode_data["goal_poses"].astype(np.float32),
        num_pieces=np.int32(episode_data["num_pieces"]),
        num_conflicts_planned=np.int32(num_conflicts_planned),
        num_conflicts_unconstrained=np.int32(num_conflicts_unconstrained),
        nodes_expanded=np.int32(nodes_expanded),
        planning_time_s=np.float32(planning_time_s),
    )
    np.savez_compressed(output_path, **save_dict)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    use_ddim = args.use_ddim and not args.no_ddim
    model = load_model(
        model_name=args.model,
        checkpoint=args.checkpoint,
        traj_len=args.traj_len,
        embed_dim=args.embed_dim,
        diffusion_steps=args.diffusion_steps,
        device=device,
    )

    planner = LowLevelPlanner(
        model=model,
        visible_range=args.visible_range,
        batch_size=args.batch_size,
        use_ddim=use_ddim,
        ddim_steps=args.ddim_steps,
        guidance_weight=args.guidance_weight,
        device=device,
    )

    # ------------------------------------------------------------------
    # Episode files
    # ------------------------------------------------------------------
    episode_files = get_episode_split(args.data_dir, args.val_frac, args.split)
    episode_files = episode_files[:args.max_episodes]
    logger.info(
        "Planning %d episodes (split=%s, algorithm=%s)",
        len(episode_files), args.split, args.algorithm,
    )

    # ------------------------------------------------------------------
    # Per-episode planning loop
    # ------------------------------------------------------------------
    summary_rows = []

    for ep_idx, ep_path in enumerate(episode_files):
        ep_name = os.path.splitext(os.path.basename(ep_path))[0]
        logger.info(
            "Episode %d/%d: %s", ep_idx + 1, len(episode_files), ep_name
        )

        episode_data = dict(np.load(ep_path, allow_pickle=False))
        N = int(episode_data["num_pieces"])

        # Extract polygon outlines from saved arrays
        outlines = [episode_data[f"outline_{i}"] for i in range(N)]

        # ------------------------------------------------------------------
        # Unconstrained baseline (no CBS guidance)
        # ------------------------------------------------------------------
        unconstrained_trajs = run_unconstrained(episode_data, planner)
        from visplan.planning.conflict_detector import count_conflicts
        n_conflicts_unconstrained = count_conflicts(unconstrained_trajs, outlines)
        logger.info(
            "  Unconstrained conflicts: %d", n_conflicts_unconstrained
        )

        # ------------------------------------------------------------------
        # CBS / PP planning
        # ------------------------------------------------------------------
        plan_kwargs = dict(
            episode_data=episode_data,
            outlines=outlines,
            planner=planner,
            visible_range=args.visible_range,
            constraint_margin=args.constraint_margin,
            constraint_epsilon=args.constraint_epsilon,
        )

        if args.algorithm == "cbs":
            result = run_cbs(**plan_kwargs, max_nodes=args.max_nodes)
        else:
            result = run_pp(**plan_kwargs)

        logger.info(
            "  Planned conflicts: %d | success: %s | nodes: %d | time: %.1fs",
            result.num_conflicts, result.success,
            result.nodes_expanded, result.planning_time_s,
        )

        # ------------------------------------------------------------------
        # Save .npz
        # ------------------------------------------------------------------
        npz_out = os.path.join(args.output_dir, f"{ep_name}_{args.algorithm}.npz")
        save_planned_npz(
            episode_data=episode_data,
            planned_trajs=result.trajectories_world,
            unconstrained_trajs=unconstrained_trajs,
            num_conflicts_planned=result.num_conflicts,
            num_conflicts_unconstrained=n_conflicts_unconstrained,
            nodes_expanded=result.nodes_expanded,
            planning_time_s=result.planning_time_s,
            output_path=npz_out,
        )
        logger.info("  Saved: %s", npz_out)

        # ------------------------------------------------------------------
        # Save visualisation video
        # ------------------------------------------------------------------
        if args.save_viz:
            vid_out = os.path.join(
                args.output_dir, f"{ep_name}_{args.algorithm}.mp4"
            )
            render_comparison_video(
                episode_data=episode_data,
                outlines=outlines,
                unconstrained_trajs=unconstrained_trajs,
                planned_trajs=result.trajectories_world,
                visible_range=args.visible_range,
                image_size=args.image_size,
                output_path=vid_out,
                fps=args.fps,
            )
            logger.info("  Video: %s", vid_out)

        summary_rows.append({
            "episode": ep_name,
            "n_pieces": N,
            "conflicts_unconstrained": n_conflicts_unconstrained,
            "conflicts_planned": result.num_conflicts,
            "success": result.success,
            "nodes_expanded": result.nodes_expanded,
            "planning_time_s": round(result.planning_time_s, 2),
        })

    # ------------------------------------------------------------------
    # Planning summary
    # ------------------------------------------------------------------
    summary_path = os.path.join(args.output_dir, "planning_summary.txt")
    _write_summary(summary_rows, summary_path, args.algorithm)
    logger.info("Summary saved to %s", summary_path)


def _write_summary(rows: list, path: str, algorithm: str) -> None:
    """Write a plaintext summary table to ``path``."""
    header = (
        f"{'Episode':30s}  {'N':>3}  {'Unconstrained':>15}  "
        f"{'Planned':>9}  {'Success':>7}  {'Nodes':>6}  {'Time(s)':>8}"
    )
    sep = "-" * len(header)

    lines = [
        f"Algorithm: {algorithm.upper()}",
        "",
        header,
        sep,
    ]
    for r in rows:
        lines.append(
            f"{r['episode']:30s}  {r['n_pieces']:3d}  "
            f"{r['conflicts_unconstrained']:15d}  "
            f"{r['conflicts_planned']:9d}  "
            f"{'YES' if r['success'] else 'no':>7}  "
            f"{r['nodes_expanded']:6d}  "
            f"{r['planning_time_s']:8.2f}"
        )
    lines.append(sep)

    # Aggregate stats
    if rows:
        success_rate = sum(r["success"] for r in rows) / len(rows) * 100
        avg_planned = sum(r["conflicts_planned"] for r in rows) / len(rows)
        avg_unconstrained = sum(r["conflicts_unconstrained"] for r in rows) / len(rows)
        avg_time = sum(r["planning_time_s"] for r in rows) / len(rows)
        lines += [
            "",
            f"Episodes:               {len(rows)}",
            f"Success rate:           {success_rate:.1f}%",
            f"Avg conflicts (unconstrained): {avg_unconstrained:.2f}",
            f"Avg conflicts (planned):       {avg_planned:.2f}",
            f"Avg planning time:      {avg_time:.2f}s",
        ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
