"""Visualize planned trajectories as animated polygon videos (post-hoc).

Reads planned .npz files produced by plan_cbs.py alongside the original
episode .npz files, and generates MP4 animations showing polygon pieces
moving from start to goal.

Two output files per episode
-----------------------------
  <output_dir>/
    episode_XXXXXX_<alg>_solo.mp4        – single-panel: planned trajectory
    episode_XXXXXX_<alg>_comparison.mp4  – side-by-side: unconstrained | planned

Usage (single episode)
----------------------
    python scripts/visualize_plan.py \\
        --planned_npz plan_output/episode_000671_cbs.npz \\
        --episode_npz data/voronoi_reassembly/ep1000_pts5-12_seed0/episode_000671.npz \\
        --output_dir viz_output

Usage (batch — process all planned npz in a directory)
-------------------------------------------------------
    python scripts/visualize_plan.py \\
        --planned_dir plan_output \\
        --data_dir data/voronoi_reassembly/ep1000_pts5-12_seed0 \\
        --output_dir viz_output
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

from visplan.planning.viz_utils import piece_colors, render_comparison_video, render_solo_video


# ---------------------------------------------------------------------------
#  Episode processing
# ---------------------------------------------------------------------------

def process_episode(
    planned_npz: str,
    episode_npz: str,
    output_dir: str,
    fps: int,
    image_size: int,
    half_range: float,
    algorithm: str,
) -> None:
    planned = dict(np.load(planned_npz, allow_pickle=False))
    episode = dict(np.load(episode_npz, allow_pickle=False))

    N = int(episode["num_pieces"])
    outlines = [episode[f"outline_{i}"] for i in range(N)]
    goal_poses = episode["goal_poses"]
    colors = piece_colors(N, seed=int(episode["voronoi_seed"]))

    planned_trajs = planned["planned_trajectories"]
    unconstrained_trajs = planned["unconstrained_trajs"]

    # Strip trailing _<algorithm> suffix to get a clean base name
    ep_stem = os.path.splitext(os.path.basename(planned_npz))[0]
    base_name = re.sub(r"_(cbs|pp|independent|rrt)$", "", ep_stem)

    os.makedirs(output_dir, exist_ok=True)

    solo_path = os.path.join(output_dir, f"{base_name}_solo.mp4")
    render_solo_video(
        outlines=outlines,
        trajectories=planned_trajs,
        goal_poses=goal_poses,
        colors=colors,
        image_size=image_size,
        half_range=half_range,
        output_path=solo_path,
        fps=fps,
        label=algorithm.upper(),
    )
    print(f"  Solo:       {solo_path}")

    cmp_path = os.path.join(output_dir, f"{base_name}_comparison.mp4")
    render_comparison_video(
        outlines=outlines,
        unconstrained_trajs=unconstrained_trajs,
        planned_trajs=planned_trajs,
        goal_poses=goal_poses,
        colors=colors,
        image_size=image_size,
        half_range=half_range,
        output_path=cmp_path,
        fps=fps,
        left_label="Unconstrained",
        right_label=algorithm.upper(),
    )
    print(f"  Comparison: {cmp_path}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _detect_algorithm(path: str) -> str:
    m = re.search(r"_(cbs|pp|independent|rrt)\.npz$", path)
    return m.group(1) if m else "planned"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render planned trajectories as animated polygon videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--planned_npz", type=str, default=None,
                   help="Path to a single planned .npz file.")
    p.add_argument("--episode_npz", type=str, default=None,
                   help="Path to the matching original episode .npz file.")
    p.add_argument("--planned_dir", type=str, default=None,
                   help="Directory containing planned episode_*_<alg>.npz files.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Directory containing original episode_*.npz files.")
    p.add_argument("--algorithm", type=str, default=None,
                   help="Algorithm label override (auto-detected from filename if omitted).")
    p.add_argument("--output_dir", type=str, default="viz_output")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--image_size", type=int, default=400)
    p.add_argument("--visible_range", type=float, default=0.6,
                   help="Camera half-extent in metres.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.planned_npz and args.episode_npz:
        alg = args.algorithm or _detect_algorithm(args.planned_npz)
        print(f"Processing: {os.path.basename(args.planned_npz)}  [{alg}]")
        process_episode(
            planned_npz=args.planned_npz,
            episode_npz=args.episode_npz,
            output_dir=args.output_dir,
            fps=args.fps,
            image_size=args.image_size,
            half_range=args.visible_range,
            algorithm=alg,
        )

    elif args.planned_dir and args.data_dir:
        planned_files = sorted(glob.glob(os.path.join(args.planned_dir, "episode_*.npz")))
        if not planned_files:
            raise FileNotFoundError(f"No episode_*.npz files in {args.planned_dir}")
        for pf in planned_files:
            m = re.match(r"(episode_\d+)_", os.path.basename(pf))
            if not m:
                print(f"Skipping (unrecognised name): {os.path.basename(pf)}")
                continue
            ep_path = os.path.join(args.data_dir, f"{m.group(1)}.npz")
            if not os.path.exists(ep_path):
                print(f"Skipping {os.path.basename(pf)}: episode npz not found")
                continue
            alg = args.algorithm or _detect_algorithm(pf)
            print(f"Processing: {os.path.basename(pf)}  [{alg}]")
            process_episode(
                planned_npz=pf,
                episode_npz=ep_path,
                output_dir=args.output_dir,
                fps=args.fps,
                image_size=args.image_size,
                half_range=args.visible_range,
                algorithm=alg,
            )
    else:
        raise ValueError(
            "Provide either --planned_npz + --episode_npz  "
            "or --planned_dir + --data_dir."
        )


if __name__ == "__main__":
    main()
