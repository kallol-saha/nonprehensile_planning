"""Quick smoke test for the VoronoiReassembly ManiSkill environment."""

import gymnasium as gym
import numpy as np
import torch

# The import registers the environment with gymnasium
import visplan.voronoi_env  # noqa: F401


def main():
    print("Creating VoronoiReassembly-v1 environment...")
    env = gym.make(
        "VoronoiReassembly-v1",
        parallel_in_single_scene=False,
        viewer_camera_configs=dict(shader_pack="rt-fast"),
        num_envs=2,
        render_mode="human",
        num_voronoi_points=4,
        side_length=0.2,
        placement_mode="assembled",
        voronoi_seed=1,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"Reset successful. Observation keys: {obs.keys() if isinstance(obs, dict) else type(obs)}")

    # Step a few times with random actions
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    env.render()
    
    speed = 0.5
    for i, (cx, cy) in enumerate(env.centroids):
        direction = np.array([cx, cy, 0.0])
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        vel = direction * speed
        env.set_piece_velocities(torch.tensor(vel).unsqueeze(0), piece_indices=[i])
        env.sim_step(50)


    env.stall(1000)

    env.close()


if __name__ == "__main__":
    main()
