"""Run all 4 planning methods on fixed validation episodes and save results.csv.

Methods evaluated
-----------------
  rrt          – RRT-Connect per piece, no coordination (run_independent)
  rrt_cbs      – RRT-Connect + CBS (run_cbs with RRTLowLevel)
  diff         – Diffusion per piece, no coordination (run_independent)
  diff_cbs     – Diffusion + CBS (run_cbs with DiffusionLowLevel)

Episode selection
-----------------
n_eval episodes are sampled deterministically from data_dir using eval_seed,
shared across all methods so results are paired (same episode per method),
enabling Wilcoxon signed-rank tests.

Per-episode seed for reproducibility: GLOBAL_SEED + episode_idx (0-indexed
in the chosen subset, not the original file index).

Metrics collected per (episode, method)
----------------------------------------
  success              bool    True iff num_conflicts == 0
  num_conflicts        int     Total pairwise piece-timestep collisions
  conflict_time_ratio  float   Fraction of timesteps with any overlap ∈ [0,1]
  planning_time_s      float   Wall-clock seconds
  nodes_expanded       int     CT nodes expanded (0 for non-CBS methods)
  arc_length           float   Mean per-piece arc length (metres)
  smoothness           float   Mean per-piece sum of squared angular accelerations
  final_pose_error     float   Mean per-piece SE(2) dist to goal at t=T-1
  n_pieces             int     Number of Voronoi pieces in the episode

Output
------
  <output_dir>/results.csv          – one row per (episode, method)
  <output_dir>/run_log.txt          – plaintext log of per-episode progress
  <output_dir>/viz/                 – polygon animation videos (with --save_viz)
    episode_XXXXXX_<method>_solo.mp4   – per-method trajectory animation
    episode_XXXXXX_comparison.mp4      – all methods side-by-side

Usage
-----
    python scripts/eval/compare_methods.py \\
        --data_dir assets/data \\
        --checkpoint checkpoints/dit_best.pt \\
        --output_dir eval_output \\
        --methods rrt rrt_cbs diff diff_cbs \\
        --n_eval 50 \\
        --eval_seed 42 \\
        --max_nodes 500 \\
        --rrt_max_samples 5000 \\
        --rrt_max_time 10.0 \\
        --save_viz

    # RRT-only (no checkpoint needed):
    python scripts/eval/compare_methods.py \\
        --data_dir assets/data \\
        --output_dir eval_output \\
        --methods rrt rrt_cbs \\
        --n_eval 50 \\
        --save_viz
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore
    _CV2_AVAILABLE = False

from visplan.planning.cbs import run_cbs, run_independent
from visplan.planning.conflict_detector import (
    count_conflicts,
    compute_conflict_time_ratio,
)
from visplan.planning.low_level_planner import DiffusionLowLevel, LowLevelPlannerProtocol
from visplan.planning.rrt_low_level import RRTLowLevel
from visplan.planning.viz_utils import (
    build_conflict_map,
    piece_colors,
    render_methods_video,
    render_solo_video,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GLOBAL_SEED = 42  # base seed; per-episode seed = GLOBAL_SEED + episode_idx

CSV_FIELDS = [
    "episode",
    "method",
    "n_pieces",
    "success",
    "num_conflicts",
    "conflict_time_ratio",
    "planning_time_s",
    "nodes_expanded",
    "arc_length",
    "smoothness",
    "final_pose_error",
]

# Metric descriptions printed in the log summary
METRIC_LABELS = {
    "success": "Success rate",
    "num_conflicts": "Avg conflicts",
    "conflict_time_ratio": "Avg conflict time ratio",
    "planning_time_s": "Avg planning time (s)",
    "arc_length": "Avg arc length (m)",
    "smoothness": "Avg smoothness (rad²)",
    "final_pose_error": "Avg final pose error (m+rad)",
}


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate RRT / Diffusion × independent / CBS on val episodes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data_dir", type=str, default=None,
                   help="Directory containing episode_*.npz files.  Required unless "
                        "--randomize_start_poses is set, in which case synthetic "
                        "Voronoi episodes are generated on the fly.")
    p.add_argument("--n_eval", type=int, default=50,
                   help="Number of validation episodes to evaluate.")
    p.add_argument("--eval_seed", type=int, default=GLOBAL_SEED,
                   help="Seed for sampling val episodes.  Per-episode RNG seed "
                        "= eval_seed + episode_idx.")
    p.add_argument("--n_pieces", type=int, default=6,
                   help="Number of Voronoi pieces per synthetic episode "
                        "(used when --data_dir is omitted).")
    # Methods
    p.add_argument("--methods", type=str, nargs="+",
                   default=["rrt", "rrt_cbs", "diff", "diff_cbs"],
                   choices=["rrt", "rrt_cbs", "diff", "diff_cbs"],
                   help="Which methods to evaluate.")
    # Diffusion model (required only when diff or diff_cbs in --methods)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to diffusion model checkpoint (.pt).  Required "
                        "when 'diff' or 'diff_cbs' are in --methods.")
    p.add_argument("--model", type=str, default="dit",
                   choices=["dit", "unet"],
                   help="Diffusion model architecture.")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--traj_len", type=int, default=32)
    p.add_argument("--diffusion_steps", type=int, default=100)
    p.add_argument("--use_ddim", action="store_true", default=True)
    p.add_argument("--no_ddim", action="store_true")
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Diffusion trajectories sampled per planning call.")
    p.add_argument("--guidance_weight", type=float, default=0.2)
    # RRT
    p.add_argument("--rrt_max_samples", type=int, default=5000,
                   help="Total node budget per RRT-Connect call.")
    p.add_argument("--rrt_max_time", type=float, default=10.0,
                   help="Wall-clock budget per RRT-Connect call (seconds).")
    # CBS
    p.add_argument("--max_nodes", type=int, default=500,
                   help="Max CT nodes to expand per CBS call.")
    p.add_argument("--constraint_margin", type=float, default=1.2)
    p.add_argument("--constraint_epsilon", type=float, default=1.2)
    # OOD / randomisation
    p.add_argument("--randomize_start_poses", action="store_true",
                   help="Replace episode start poses with uniform-random SE(2) poses "
                        "inside the arena.  Useful for out-of-distribution testing. "
                        "Uses per-episode seed for reproducibility.")
    p.add_argument("--start_pose_margin", type=float, default=0.8,
                   help="Fraction of visible_range used as the sampling half-extent "
                        "for randomised start poses (avoids placing pieces at the edge).")
    # Misc
    p.add_argument("--visible_range", type=float, default=0.6)
    p.add_argument("--output_dir", type=str, default="eval_output")
    # Visualisation
    p.add_argument("--save_viz", action="store_true",
                   help="Generate polygon animation videos alongside results.")
    p.add_argument("--fps", type=int, default=8,
                   help="Video frame rate (used when --save_viz).")
    p.add_argument("--image_size", type=int, default=400,
                   help="Per-panel video resolution in pixels (used when --save_viz).")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Episode loading
# ---------------------------------------------------------------------------

def sample_eval_episodes(
    data_dir: str,
    n_eval: int,
    seed: int,
) -> List[str]:
    """Deterministically sample n_eval episodes from data_dir."""
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found in {data_dir}")
    rng = np.random.RandomState(seed)
    n = min(n_eval, len(files))
    idxs = rng.choice(len(files), size=n, replace=False)
    idxs.sort()
    chosen = [files[i] for i in idxs]
    logger.info(
        "Selected %d/%d val episodes (seed=%d).", len(chosen), len(files), seed
    )
    return chosen


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

def load_diffusion_model(
    model_name: str,
    checkpoint: str,
    traj_len: int,
    embed_dim: int,
    diffusion_steps: int,
    device: torch.device,
) -> torch.nn.Module:
    from visplan.training.models.diffusion_transformer import DiffusionTransformer
    from visplan.training.models.diffusion_unet import DiffusionUNet

    registry = {"dit": DiffusionTransformer, "unet": DiffusionUNet}
    cls = registry[model_name]
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


# ---------------------------------------------------------------------------
#  Planner factories
# ---------------------------------------------------------------------------

def build_planners(
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, LowLevelPlannerProtocol]:
    """Build and return the requested planners."""
    planners: Dict[str, LowLevelPlannerProtocol] = {}

    needs_diffusion = any(m in args.methods for m in ("diff", "diff_cbs"))
    needs_rrt = any(m in args.methods for m in ("rrt", "rrt_cbs"))

    if needs_diffusion:
        if args.checkpoint is None:
            raise ValueError(
                "--checkpoint is required when evaluating 'diff' or 'diff_cbs' methods"
            )
        use_ddim = args.use_ddim and not args.no_ddim
        model = load_diffusion_model(
            model_name=args.model,
            checkpoint=args.checkpoint,
            traj_len=args.traj_len,
            embed_dim=args.embed_dim,
            diffusion_steps=args.diffusion_steps,
            device=device,
        )
        planners["diffusion"] = DiffusionLowLevel(
            model=model,
            visible_range=args.visible_range,
            batch_size=args.batch_size,
            use_ddim=use_ddim,
            ddim_steps=args.ddim_steps,
            guidance_weight=args.guidance_weight,
            device=device,
        )

    if needs_rrt:
        planners["rrt"] = RRTLowLevel(
            visible_range=args.visible_range,
            traj_len=args.traj_len,
            seed=args.eval_seed,
            max_samples=args.rrt_max_samples,
            max_time_s=args.rrt_max_time,
        )

    return planners


# ---------------------------------------------------------------------------
#  Metrics computation
# ---------------------------------------------------------------------------

def compute_arc_length(trajectories_world: np.ndarray) -> float:
    """Mean per-piece arc length in metres.

    Args:
        trajectories_world: (N, T, 3) SE(2) array.
    Returns:
        Mean arc length across all pieces.
    """
    # Δxy differences between consecutive timesteps
    diffs = np.diff(trajectories_world[:, :, :2], axis=1)  # (N, T-1, 2)
    lengths = np.linalg.norm(diffs, axis=-1).sum(axis=1)    # (N,)
    return float(lengths.mean())


def compute_smoothness(trajectories_world: np.ndarray) -> float:
    """Mean per-piece sum of squared angular accelerations (rad²).

    Smoothness = Σ_t (θ_{t+1} - 2θ_t + θ_{t-1})²  averaged over pieces.
    Wrapping is applied to each angular difference before computing the
    second difference so that wrap-arounds don't inflate the metric.

    Args:
        trajectories_world: (N, T, 3) SE(2) array.
    Returns:
        Mean smoothness value.
    """
    thetas = trajectories_world[:, :, 2]  # (N, T)
    # First differences (wrapped)
    d1 = thetas[:, 1:] - thetas[:, :-1]
    d1 = (d1 + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
    # Second differences
    d2 = d1[:, 1:] - d1[:, :-1]
    return float((d2 ** 2).sum(axis=1).mean())


def compute_final_pose_error(
    trajectories_world: np.ndarray,
    goal_poses: np.ndarray,
) -> float:
    """Mean per-piece SE(2) distance from final waypoint to goal pose.

    SE(2) distance = ||Δxy||₂ + 0.1 * |Δθ|_wrapped.

    Args:
        trajectories_world: (N, T, 3) SE(2) array.
        goal_poses:         (N, 3) SE(2) goal poses.
    Returns:
        Mean final pose error.
    """
    N = trajectories_world.shape[0]
    final = trajectories_world[:, -1, :]   # (N, 3)
    errors = []
    for i in range(N):
        dxy = float(np.linalg.norm(final[i, :2] - goal_poses[i, :2]))
        dtheta = abs(float(((final[i, 2] - goal_poses[i, 2]) + np.pi) % (2 * np.pi) - np.pi))
        errors.append(dxy + 0.1 * dtheta)
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
#  Per-episode evaluation
# ---------------------------------------------------------------------------

def evaluate_episode(
    episode_data: dict,
    outlines: List[np.ndarray],
    method: str,
    planner: LowLevelPlannerProtocol,
    args: argparse.Namespace,
) -> Tuple[dict, object]:
    """Run one method on one episode and return (metrics dict, PlanResult).

    Args:
        episode_data: Loaded .npz dict.
        outlines:     List of polygon outlines.
        method:       One of 'rrt', 'rrt_cbs', 'diff', 'diff_cbs'.
        planner:      The appropriate low-level planner.
        args:         Parsed CLI arguments.

    Returns:
        Tuple of (metrics dict, PlanResult).
        metrics dict has keys matching CSV_FIELDS (except 'episode' and 'method').
        PlanResult carries .trajectories_world (N, T, 3) among other fields.
    """
    if method in ("rrt", "diff"):
        result = run_independent(
            episode_data=episode_data,
            outlines=outlines,
            planner=planner,
            visible_range=args.visible_range,
        )
    elif method in ("rrt_cbs", "diff_cbs"):
        result = run_cbs(
            episode_data=episode_data,
            outlines=outlines,
            planner=planner,
            visible_range=args.visible_range,
            constraint_margin=args.constraint_margin,
            constraint_epsilon=args.constraint_epsilon,
            max_nodes=args.max_nodes,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    trajs = result.trajectories_world  # (N, T, 3)
    goal_poses = episode_data["goal_poses"].astype(np.float32)  # (N, 3)

    ctr = compute_conflict_time_ratio(trajs, outlines)

    metrics = {
        "n_pieces": int(episode_data["num_pieces"]),
        "success": result.success,
        "num_conflicts": result.num_conflicts,
        "conflict_time_ratio": round(ctr, 6),
        "planning_time_s": round(result.planning_time_s, 4),
        "nodes_expanded": result.nodes_expanded,
        "arc_length": round(compute_arc_length(trajs), 6),
        "smoothness": round(compute_smoothness(trajs), 8),
        "final_pose_error": round(compute_final_pose_error(trajs, goal_poses), 6),
    }
    return metrics, result


# ---------------------------------------------------------------------------
#  OOD helpers
# ---------------------------------------------------------------------------

def generate_synthetic_episode(
    n_pieces: int,
    visible_range: float,
    seed: int,
) -> dict:
    """Generate a synthetic episode dict with Voronoi outlines and random poses.

    Uses Shapely to produce 2D Voronoi polygon outlines (no simulation required).
    Does NOT include image keys (``start_image``, ``goal_image``, ``piece_masks``),
    so this is only compatible with RRT-based planners.

    Args:
        n_pieces:      Number of Voronoi pieces to generate.
        visible_range: Camera half-extent in metres; poses are sampled inside this.
        seed:          RNG seed for reproducible generation.

    Returns:
        Dict with keys: num_pieces, voronoi_seed, outline_i, start_poses, goal_poses.
    """
    from shapely.geometry import MultiPoint
    from shapely.ops import voronoi_diagram
    import shapely.affinity

    rng = np.random.RandomState(seed)
    side = visible_range * 0.6  # piece size relative to arena

    from shapely.geometry import Polygon as ShapelyPolygon
    boundary = ShapelyPolygon([
        (-side / 2, -side / 2), (-side / 2, side / 2),
        (side / 2, side / 2), (side / 2, -side / 2),
    ])

    # Retry until we get exactly n_pieces valid cells
    for attempt in range(20):
        pts = rng.uniform(-side / 2, side / 2, size=(n_pieces, 2))
        vor = voronoi_diagram(MultiPoint(pts))
        cells = [boundary.intersection(g) for g in vor.geoms if not g.is_empty]
        cells = [c for c in cells if not c.is_empty and c.geom_type == "Polygon"]
        if len(cells) >= n_pieces:
            cells = cells[:n_pieces]
            break
    else:
        raise RuntimeError(f"Could not generate {n_pieces} Voronoi cells after 20 attempts")

    outlines = []
    centroids = []
    for cell in cells:
        scaled = shapely.affinity.scale(cell, xfact=0.9, yfact=0.9, origin="centroid")
        cx, cy = cell.centroid.x, cell.centroid.y
        centroids.append((cx, cy))
        centered = shapely.affinity.translate(scaled, xoff=-cx, yoff=-cy)
        verts = np.array(centered.exterior.coords[:-1], dtype=np.float32)
        outlines.append(verts)

    # Goal: reassemble the square — each piece at its original centroid, theta=0
    goal_poses = np.array(
        [[cx, cy, 0.0] for cx, cy in centroids], dtype=np.float32
    )

    half = visible_range * 0.8
    start_poses = np.column_stack([
        rng.uniform(-half, half, n_pieces),
        rng.uniform(-half, half, n_pieces),
        rng.uniform(-np.pi, np.pi, n_pieces),
    ]).astype(np.float32)

    ep: dict = {
        "num_pieces": np.int32(n_pieces),
        "voronoi_seed": np.int32(seed),
        "start_poses": start_poses,
        "goal_poses": goal_poses,
    }
    for i, ol in enumerate(outlines):
        ep[f"outline_{i}"] = ol
    return ep


def randomize_start_poses(
    episode_data: dict,
    visible_range: float,
    margin: float,
    rng: np.random.RandomState,
) -> dict:
    """Return a shallow copy of episode_data with start_poses replaced by random SE(2) poses.

    Each piece gets an independent uniform sample:
      x, y ∈ [-visible_range * margin, visible_range * margin]
      θ    ∈ [-π, π]

    Args:
        episode_data:  Original episode dict (not mutated).
        visible_range: Camera half-extent in metres.
        margin:        Fraction of visible_range for the sampling half-extent.
        rng:           Seeded RandomState for reproducibility.

    Returns:
        Modified copy with ``start_poses`` overwritten.
    """
    N = int(episode_data["num_pieces"])
    half = visible_range * margin
    xy = rng.uniform(-half, half, size=(N, 2)).astype(np.float32)
    theta = rng.uniform(-np.pi, np.pi, size=(N, 1)).astype(np.float32)
    new_start_poses = np.concatenate([xy, theta], axis=1)  # (N, 3)
    patched = dict(episode_data)
    patched["start_poses"] = new_start_poses
    return patched


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Episode selection / synthetic mode
    # ------------------------------------------------------------------
    _DIFFUSION_METHODS = {"diff", "diff_cbs"}
    if args.data_dir is None:
        if not args.randomize_start_poses:
            raise ValueError(
                "--data_dir is required unless --randomize_start_poses is set "
                "(synthetic mode requires randomised start poses)."
            )
        if _DIFFUSION_METHODS.intersection(args.methods):
            raise ValueError(
                "Diffusion methods ('diff', 'diff_cbs') require --data_dir "
                "(they need start/goal images). Use 'rrt' or 'rrt_cbs' for "
                "synthetic episodes."
            )
        episode_files = None
        logger.info(
            "Synthetic mode: generating %d episodes × %d methods (n_pieces=%d).",
            args.n_eval, len(args.methods), args.n_pieces,
        )
    else:
        episode_files = sample_eval_episodes(args.data_dir, args.n_eval, args.eval_seed)
        logger.info("Evaluating %d episodes × %d methods.", len(episode_files), len(args.methods))

    # ------------------------------------------------------------------
    # Build planners
    # ------------------------------------------------------------------
    planners = build_planners(args, device)

    # Map method name → planner instance
    method_planner_map = {
        "rrt":      planners.get("rrt"),
        "rrt_cbs":  planners.get("rrt"),
        "diff":     planners.get("diffusion"),
        "diff_cbs": planners.get("diffusion"),
    }

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.output_dir, "results.csv")
    log_path = os.path.join(args.output_dir, "run_log.txt")

    all_rows = []

    viz_dir = os.path.join(args.output_dir, "viz") if args.save_viz else None
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)

    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        n_episodes = args.n_eval if episode_files is None else len(episode_files)
        with open(log_path, "w") as log_f:
            for ep_idx in range(n_episodes):
                # Per-episode RNG seed for RRT reproducibility
                ep_seed = args.eval_seed + ep_idx
                if "rrt" in planners:
                    planners["rrt"]._base_seed = ep_seed
                    planners["rrt"]._call_count = 0

                if episode_files is not None:
                    ep_path = episode_files[ep_idx]
                    ep_name = os.path.splitext(os.path.basename(ep_path))[0]
                    episode_data = dict(np.load(ep_path, allow_pickle=False))
                else:
                    ep_name = f"synthetic_{ep_idx:06d}"
                    episode_data = generate_synthetic_episode(
                        args.n_pieces, args.visible_range, seed=ep_seed
                    )

                N = int(episode_data["num_pieces"])
                outlines = [episode_data[f"outline_{i}"] for i in range(N)]

                if args.randomize_start_poses and episode_files is not None:
                    # synthetic episodes already have random poses baked in
                    ep_rng = np.random.RandomState(args.eval_seed + ep_idx + 9999)
                    episode_data = randomize_start_poses(
                        episode_data, args.visible_range, args.start_pose_margin, ep_rng
                    )
                    logger.info("  Randomised start poses (margin=%.2f)", args.start_pose_margin)

                log_f.write(
                    f"\n=== Episode {ep_idx + 1}/{n_episodes}: {ep_name} "
                    f"(N={N}) ===\n"
                )
                logger.info(
                    "Episode %d/%d: %s (N=%d)",
                    ep_idx + 1, n_episodes, ep_name, N,
                )

                ep_results: dict = {}

                for method in args.methods:
                    planner = method_planner_map[method]
                    if planner is None:
                        logger.warning("Planner for method '%s' not built; skipping.", method)
                        continue

                    # Reset RRT call counter per method to ensure reproducibility
                    if hasattr(planner, "_call_count"):
                        planner._call_count = 0

                    t0 = time.time()
                    try:
                        metrics, result = evaluate_episode(
                            episode_data=episode_data,
                            outlines=outlines,
                            method=method,
                            planner=planner,
                            args=args,
                        )
                    except Exception as exc:
                        logger.error(
                            "Episode %s method %s failed: %s", ep_name, method, exc
                        )
                        # Write a row of NaNs so the CSV stays aligned
                        metrics = {
                            "n_pieces": N,
                            "success": False,
                            "num_conflicts": -1,
                            "conflict_time_ratio": float("nan"),
                            "planning_time_s": round(time.time() - t0, 4),
                            "nodes_expanded": 0,
                            "arc_length": float("nan"),
                            "smoothness": float("nan"),
                            "final_pose_error": float("nan"),
                        }
                        result = None

                    row = {"episode": ep_name, "method": method, **metrics}
                    all_rows.append(row)
                    writer.writerow(row)
                    csv_f.flush()

                    log_line = (
                        f"  {method:12s}  success={str(metrics['success']):5s}  "
                        f"conflicts={metrics['num_conflicts']:4d}  "
                        f"ctr={metrics['conflict_time_ratio']:.3f}  "
                        f"time={metrics['planning_time_s']:.2f}s  "
                        f"nodes={metrics['nodes_expanded']}"
                    )
                    log_f.write(log_line + "\n")
                    logger.info(log_line)

                    if viz_dir and result is not None:
                        ep_results[method] = result

                # After all methods: render solo + comparison videos
                if viz_dir and ep_results:
                    _save_viz(
                        episode_data=episode_data,
                        outlines=outlines,
                        ep_results=ep_results,
                        args=args,
                        viz_dir=viz_dir,
                        ep_name=ep_name,
                    )

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    _print_summary(all_rows, args.methods)
    logger.info("Results saved to %s", csv_path)
    logger.info("Run log saved to %s", log_path)


def _save_viz(
    episode_data: dict,
    outlines: List[np.ndarray],
    ep_results: dict,
    args: argparse.Namespace,
    viz_dir: str,
    ep_name: str,
) -> None:
    """Render per-method solo videos and one multi-panel comparison video."""
    colors = piece_colors(
        int(episode_data["num_pieces"]),
        seed=int(episode_data["voronoi_seed"]),
    )
    goal_poses = episode_data["goal_poses"].astype(np.float32)

    panels = []
    for method, result in ep_results.items():
        trajs = result.trajectories_world
        cmap = build_conflict_map(trajs, outlines)

        solo_path = os.path.join(viz_dir, f"{ep_name}_{method}_solo.mp4")
        try:
            render_solo_video(
                outlines=outlines,
                trajectories=trajs,
                goal_poses=goal_poses,
                colors=colors,
                image_size=args.image_size,
                half_range=args.visible_range,
                output_path=solo_path,
                fps=args.fps,
                label=method.upper(),
                conflict_map=cmap,
            )
            logger.info("    viz solo:  %s", solo_path)
        except Exception as exc:
            logger.warning("    viz solo failed (%s): %s", method, exc)

        panels.append((method.upper(), trajs, cmap))

    if len(panels) > 1:
        cmp_path = os.path.join(viz_dir, f"{ep_name}_comparison.mp4")
        try:
            render_methods_video(
                outlines=outlines,
                panels=panels,
                goal_poses=goal_poses,
                colors=colors,
                image_size=args.image_size,
                half_range=args.visible_range,
                output_path=cmp_path,
                fps=args.fps,
            )
            logger.info("    viz comparison: %s", cmp_path)
        except Exception as exc:
            logger.warning("    viz comparison failed: %s", exc)


def _print_summary(rows: List[dict], methods: List[str]) -> None:
    """Print a compact aggregate table to stdout."""
    from collections import defaultdict

    by_method: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    print("\n" + "=" * 72)
    print(f"{'Method':12s}  {'N_ep':>4}  {'Succ%':>6}  {'Conflicts':>9}  "
          f"{'CTR':>6}  {'Time(s)':>7}  {'FinalErr':>8}")
    print("-" * 72)
    for method in methods:
        mrs = by_method.get(method, [])
        if not mrs:
            continue
        valid = [r for r in mrs if r["num_conflicts"] >= 0]
        n = len(valid)
        if n == 0:
            continue
        succ_pct = 100.0 * sum(r["success"] for r in valid) / n
        avg_conf = sum(r["num_conflicts"] for r in valid) / n
        ctr_vals = [r["conflict_time_ratio"] for r in valid
                    if not np.isnan(r["conflict_time_ratio"])]
        avg_ctr = sum(ctr_vals) / len(ctr_vals) if ctr_vals else float("nan")
        avg_time = sum(r["planning_time_s"] for r in valid) / n
        err_vals = [r["final_pose_error"] for r in valid
                    if not np.isnan(r["final_pose_error"])]
        avg_err = sum(err_vals) / len(err_vals) if err_vals else float("nan")
        print(f"{method:12s}  {n:4d}  {succ_pct:6.1f}  {avg_conf:9.2f}  "
              f"{avg_ctr:6.3f}  {avg_time:7.2f}  {avg_err:8.4f}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
