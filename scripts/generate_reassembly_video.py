"""
Generate a reassembly video by:
1. Starting with assembled pieces
2. Randomly applying velocities to random pieces (disassembly)
3. Capturing top-view frames during simulation at a configurable frequency
4. Reversing the frames and saving as an MP4 (reassembly)
"""

import gymnasium as gym
import numpy as np
import torch
import cv2

import visplan.voronoi_env  # noqa: F401


def generate_reassembly_video(
    output_path: str = "reassembly.mp4",
    num_voronoi_points: int = 8,
    voronoi_seed: int = 1,
    num_disassembly_steps: int = 200,
    sim_steps_per_move: int = 30,
    capture_every: int = 3,
    speed_range: tuple = (0.3, 0.7),
    fps: int = 30,
    image_size: int = 512,
    rng_seed: int = 42,
):
    """
    Generate a reassembly video.

    Args:
        output_path: Where to save the MP4.
        num_voronoi_points: Number of Voronoi pieces.
        voronoi_seed: Seed for Voronoi generation.
        num_disassembly_steps: How many random velocity applications.
        sim_steps_per_move: Physics steps to run after each velocity impulse.
        capture_every: Capture a frame every this many physics steps.
        speed_range: (min, max) speed in m/s for random velocities.
        fps: Video frame rate.
        image_size: Resolution of top-view frames.
        rng_seed: RNG seed for reproducibility.
    """
    rng = np.random.RandomState(rng_seed)

    print("Creating environment...")
    env = gym.make(
        "VoronoiReassembly-v1",
        parallel_in_single_scene=False,
        num_envs=1,
        render_mode=None,
        num_voronoi_points=num_voronoi_points,
        side_length=0.2,
        placement_mode="assembled",
        voronoi_seed=voronoi_seed,
    )

    env.reset()

    # Capture the initial assembled frame
    frames = [env.get_top_view(image_size=image_size)]
    print(f"Captured initial frame. Starting disassembly ({num_disassembly_steps} steps, capturing every {capture_every} sim steps)...")

    num_pieces = env.num_pieces

    for step in range(num_disassembly_steps):
        # Pick a random piece
        piece_idx = rng.randint(0, num_pieces)

        # Random direction in XY plane
        angle = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(*speed_range)
        vel = np.array([np.cos(angle) * speed, np.sin(angle) * speed, 0.0])

        # Apply velocity and simulate, capturing frames automatically
        env.set_piece_velocities(
            torch.tensor(vel, dtype=torch.float32).unsqueeze(0),
            piece_indices=[piece_idx],
        )
        step_frames = env.sim_step(
            steps=sim_steps_per_move,
            capture_every=capture_every,
            image_size=image_size,
        )
        frames.extend(step_frames)
        print(f"  Step {step + 1}/{num_disassembly_steps}: moved piece {piece_idx} ({len(step_frames)} frames captured)")

    env.close()

    # Reverse frames for reassembly video
    frames_reversed = frames[::-1]

    # Write MP4 using OpenCV (works headless over SSH)
    h, w = frames_reversed[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames_reversed:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved reassembly video to {output_path} ({len(frames_reversed)} frames, {fps} fps, ~{len(frames_reversed)/fps:.1f}s)")


if __name__ == "__main__":
    generate_reassembly_video()
