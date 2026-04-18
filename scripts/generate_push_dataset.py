#!/usr/bin/env python
"""Generate a push trajectory dataset: one trajectory per piece per environment.

Each environment has a fixed number of pieces (random convex polygons).
Each piece gets one independent push trajectory. Saved as one .npz per env.

Usage:
    python scripts/generate_push_dataset.py \
        --num_envs 1000 --pieces_per_env 4 --seed 0 \
        --output_dir assets/push_data
"""

import argparse
import json
import os
import time

import numpy as np

from visplan.push_sim.polygons import random_convex_polygon
from visplan.push_sim.data_gen import random_push_trajectory, _sample_num_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--pieces_per_env", type=int, required=True)
    parser.add_argument("--n_verts_min", type=int, default=4)
    parser.add_argument("--n_verts_max", type=int, default=10)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--steps_min", type=int, default=30)
    parser.add_argument("--steps_max", type=int, default=150)
    parser.add_argument("--push_dist_min", type=float, default=0.01)
    parser.add_argument("--push_dist_max", type=float, default=0.08)
    parser.add_argument("--start_pose_range", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    push_dist_range = (args.push_dist_min, args.push_dist_max)
    steps_range = (args.steps_min, args.steps_max)

    t0 = time.time()
    max_steps_seen = 0
    max_verts_seen = 0

    for ei in range(args.num_envs):
        env_data = {}
        env_data["num_pieces"] = np.int32(args.pieces_per_env)

        for pi in range(args.pieces_per_env):
            nv = rng.randint(args.n_verts_min, args.n_verts_max + 1)
            polygon = random_convex_polygon(n_verts=nv, scale=args.scale, rng=rng)

            x0 = rng.uniform(-args.start_pose_range, args.start_pose_range)
            y0 = rng.uniform(-args.start_pose_range, args.start_pose_range)
            theta0 = rng.uniform(-np.pi, np.pi)
            start_pose = np.array([x0, y0, theta0])

            num_steps = _sample_num_steps(steps_range, rng)

            traj = random_push_trajectory(
                polygon, start_pose, num_steps=num_steps,
                push_distance_range=push_dist_range, rng=rng,
            )

            poses = traj["poses"]  # (T+1, 3)

            env_data[f"polygon_vertices_{pi}"] = polygon.vertices.astype(np.float32)
            env_data[f"num_vertices_{pi}"] = np.int32(polygon.num_vertices)
            env_data[f"start_pose_{pi}"] = poses[0].astype(np.float32)
            env_data[f"goal_pose_{pi}"] = poses[-1].astype(np.float32)
            env_data[f"trajectory_{pi}"] = poses.astype(np.float32)
            env_data[f"actions_{pi}"] = traj["actions"].astype(np.float32)
            env_data[f"num_steps_{pi}"] = np.int32(num_steps)

            max_steps_seen = max(max_steps_seen, num_steps)
            max_verts_seen = max(max_verts_seen, polygon.num_vertices)

        out_path = os.path.join(args.output_dir, f"env_{ei:06d}.npz")
        np.savez_compressed(out_path, **env_data)

        if (ei + 1) % 500 == 0 or ei == 0:
            elapsed = time.time() - t0
            rate = (ei + 1) / elapsed
            eta = (args.num_envs - ei - 1) / rate
            print(f"  [{ei+1}/{args.num_envs}] "
                  f"{(ei+1) * args.pieces_per_env} trajectories, "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    total_trajs = args.num_envs * args.pieces_per_env

    # Save dataset metadata for the dataloader
    meta = {
        "num_envs": args.num_envs,
        "pieces_per_env": args.pieces_per_env,
        "total_trajectories": total_trajs,
        "max_steps": int(max_steps_seen),
        "max_vertices": int(max_verts_seen),
        "steps_range": list(steps_range),
        "push_distance_range": list(push_dist_range),
        "start_pose_range": args.start_pose_range,
        "polygon_scale": args.scale,
        "n_verts_range": [args.n_verts_min, args.n_verts_max],
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {args.num_envs} envs × {args.pieces_per_env} pieces = "
          f"{total_trajs} trajectories in {elapsed:.1f}s")
    print(f"  max trajectory length: {max_steps_seen + 1} poses ({max_steps_seen} steps)")
    print(f"  max polygon vertices:  {max_verts_seen}")
    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
