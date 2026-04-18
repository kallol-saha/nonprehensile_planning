#!/usr/bin/env python
"""Demo: generate and visualise quasi-static push trajectories.

Creates a few random convex polygons, generates push trajectories for each,
and plots the results.  Saves to ``push_sim_demo.png``.

Usage:
    python scripts/demo_push_sim.py
    python scripts/demo_push_sim.py --num_pieces 4 --steps_min 30 --steps_max 100
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from visplan.push_sim.polygons import random_convex_polygon, regular_polygon
from visplan.push_sim.data_gen import random_push_trajectory, _sample_num_steps


def plot_trajectory(ax, polygon, traj_dict, color="tab:blue", alpha_range=(0.08, 0.85)):
    """Draw a polygon trajectory: faded start → solid end, with path line."""
    poses = traj_dict["poses"]  # (T+1, 3)
    T = len(poses)

    # Only draw polygon every few frames to avoid visual clutter
    draw_every = max(1, T // 12)

    for t in range(0, T, draw_every):
        alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * t / max(T - 1, 1)
        world_verts = polygon.transform(poses[t, 0], poses[t, 1], poses[t, 2])
        patch = MplPolygon(world_verts, closed=True, facecolor=color,
                           edgecolor="black", linewidth=0.4, alpha=alpha)
        ax.add_patch(patch)

    # Always draw the final pose fully opaque
    world_verts = polygon.transform(poses[-1, 0], poses[-1, 1], poses[-1, 2])
    patch = MplPolygon(world_verts, closed=True, facecolor=color,
                       edgecolor="black", linewidth=0.8, alpha=0.95)
    ax.add_patch(patch)

    # Draw centroid path
    ax.plot(poses[:, 0], poses[:, 1], "-", color=color, linewidth=1.2, alpha=0.6)
    ax.plot(poses[0, 0], poses[0, 1], "o", color="white", markersize=5,
            markeredgecolor=color, markeredgewidth=1.5, zorder=5)
    ax.plot(poses[-1, 0], poses[-1, 1], "s", color=color, markersize=6, zorder=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pieces", type=int, default=3)
    parser.add_argument("--steps_min", type=int, default=20)
    parser.add_argument("--steps_max", type=int, default=80)
    parser.add_argument("--trajs_per_piece", type=int, default=3)
    parser.add_argument("--push_dist_min", type=float, default=0.01)
    parser.add_argument("--push_dist_max", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="push_sim_demo.png")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    colors = plt.cm.Set2(np.linspace(0, 1, max(args.num_pieces, 3)))
    push_dist_range = (args.push_dist_min, args.push_dist_max)
    steps_range = (args.steps_min, args.steps_max)

    # Create a mix of polygon shapes
    polygons = []
    for i in range(args.num_pieces):
        if i == 0:
            polygons.append(regular_polygon(5, radius=0.04))
        else:
            nv = rng.randint(4, 8)
            polygons.append(random_convex_polygon(n_verts=nv, scale=0.04, rng=rng))

    fig, axes = plt.subplots(1, args.num_pieces, figsize=(5 * args.num_pieces, 5))
    if args.num_pieces == 1:
        axes = [axes]

    for pi, (polygon, ax) in enumerate(zip(polygons, axes)):
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        for ti in range(args.trajs_per_piece):
            start = np.array([
                rng.uniform(-0.08, 0.08),
                rng.uniform(-0.08, 0.08),
                rng.uniform(-np.pi, np.pi),
            ])
            num_steps = _sample_num_steps(steps_range, rng)
            shade = 0.6 + 0.4 * ti / max(args.trajs_per_piece - 1, 1)

            traj = random_push_trajectory(
                polygon, start, num_steps=num_steps,
                push_distance_range=push_dist_range, rng=rng)
            plot_trajectory(ax, polygon, traj,
                            color=colors[pi] * shade, alpha_range=(0.08, 0.85))

            print(f"  Piece {pi}, traj {ti}: {num_steps} steps")

        ax.set_title(f"Piece {pi} ({polygon.num_vertices} verts)", fontsize=11)
        # Auto-fit view to data
        ax.autoscale_view()
        margin = 0.05
        xl, xr = ax.get_xlim()
        yl, yr = ax.get_ylim()
        ax.set_xlim(xl - margin, xr + margin)
        ax.set_ylim(yl - margin, yr + margin)

    fig.suptitle("Quasi-static push trajectories (variable length)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.save, dpi=150)
    print(f"\nSaved to {args.save}")

    # Print dataset stats
    from visplan.push_sim.data_gen import generate_dataset
    ds = generate_dataset(polygons, num_trajectories_per_piece=10,
                          steps_range=steps_range,
                          push_distance_range=push_dist_range,
                          seed=args.seed)
    lengths = ds["num_steps"]
    print(f"\nDataset: {len(lengths)} trajectories")
    print(f"  step lengths: min={lengths.min()}, max={lengths.max()}, "
          f"mean={lengths.mean():.1f}, median={np.median(lengths):.0f}")
    print(f"  pieces: {len(ds['polygon_verts'])} unique shapes")


if __name__ == "__main__":
    main()
