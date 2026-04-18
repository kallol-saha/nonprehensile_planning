"""Dataset for geometry-conditioned push trajectory diffusion.

Each sample is one piece from one environment:
  - Input:  polygon vertices (V_max, 2) + mask, start pose (4,), goal pose (4,)
  - Target: trajectory (T_max, 4) + mask

Poses are stored as raw [x, y, θ] in the .npz files. This dataset normalises
them to [x̃, ỹ, cos θ, sin θ] where xy is divided by start_pose_range.
"""

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PushTrajectoryDataset(Dataset):
    """Per-piece push trajectory dataset.

    Args:
        data_dir:   Path containing env_XXXXXX.npz files and meta.json.
        augment:    If True, apply random SE(2) augmentations.
    """

    def __init__(self, data_dir: str, augment: bool = False):
        self.data_dir = data_dir
        self.augment = augment

        # Load metadata
        meta_path = os.path.join(data_dir, "meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.max_traj_len = self.meta["max_steps"] + 1   # poses = steps + 1
        self.max_vertices = self.meta["max_vertices"]
        self.xy_norm = self.meta["start_pose_range"]      # normalise xy to ~[-1, 1]

        self.files = sorted(glob.glob(os.path.join(data_dir, "env_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No env files found in {data_dir}")

        # Build flat index: (file_idx, piece_idx)
        pieces_per_env = self.meta["pieces_per_env"]
        self.index = []
        for fi in range(len(self.files)):
            for pi in range(pieces_per_env):
                self.index.append((fi, pi))

    def __len__(self):
        return len(self.index)

    def _encode_pose(self, pose_xy_theta: np.ndarray) -> np.ndarray:
        """[x, y, θ] → [x̃, ỹ, cos θ, sin θ], float32."""
        x, y, theta = pose_xy_theta[0], pose_xy_theta[1], pose_xy_theta[2]
        return np.array([
            x / self.xy_norm,
            y / self.xy_norm,
            np.cos(theta),
            np.sin(theta),
        ], dtype=np.float32)

    def _encode_trajectory(self, traj: np.ndarray) -> np.ndarray:
        """(T, 3) [x, y, θ] → (T, 4) [x̃, ỹ, cos θ, sin θ], float32."""
        xy = traj[:, :2] / self.xy_norm
        cos_t = np.cos(traj[:, 2:3])
        sin_t = np.sin(traj[:, 2:3])
        return np.concatenate([xy, cos_t, sin_t], axis=-1).astype(np.float32)

    def __getitem__(self, idx):
        fi, pi = self.index[idx]
        d = np.load(self.files[fi])

        # Polygon vertices: (V_i, 2) → pad to (V_max, 2)
        verts = d[f"polygon_vertices_{pi}"]              # (V_i, 2)
        num_verts = int(d[f"num_vertices_{pi}"])
        verts_padded = np.zeros((self.max_vertices, 2), dtype=np.float32)
        verts_padded[:num_verts] = verts
        verts_mask = np.zeros(self.max_vertices, dtype=np.float32)
        verts_mask[:num_verts] = 1.0

        # Start / goal poses
        start_pose = self._encode_pose(d[f"start_pose_{pi}"])   # (4,)
        goal_pose = self._encode_pose(d[f"goal_pose_{pi}"])     # (4,)

        # Trajectory: (T_i+1, 3) → encode → pad to (T_max, 4)
        traj_raw = d[f"trajectory_{pi}"]                         # (T_i+1, 3)
        traj_enc = self._encode_trajectory(traj_raw)             # (T_i+1, 4)
        traj_len = traj_enc.shape[0]
        traj_padded = np.zeros((self.max_traj_len, 4), dtype=np.float32)
        traj_padded[:traj_len] = traj_enc
        traj_mask = np.zeros(self.max_traj_len, dtype=np.float32)
        traj_mask[:traj_len] = 1.0

        d.close()

        sample = {
            "polygon_vertices": torch.from_numpy(verts_padded),  # (V_max, 2)
            "polygon_mask": torch.from_numpy(verts_mask),        # (V_max,)
            "start_pose": torch.from_numpy(start_pose),          # (4,)
            "goal_pose": torch.from_numpy(goal_pose),            # (4,)
            "trajectory": torch.from_numpy(traj_padded),         # (T_max, 4)
            "trajectory_mask": torch.from_numpy(traj_mask),      # (T_max,)
            "traj_len": traj_len,
        }

        if self.augment:
            sample = self._augment(sample)

        return sample

    def _augment(self, sample):
        """Random reflections — exact symmetries of the SE(2) push problem."""
        # Random horizontal flip (negate x, negate sin θ)
        if np.random.random() < 0.5:
            sample["polygon_vertices"][:, 0] *= -1
            sample["start_pose"][0] *= -1
            sample["start_pose"][3] *= -1
            sample["goal_pose"][0] *= -1
            sample["goal_pose"][3] *= -1
            sample["trajectory"][:, 0] *= -1
            sample["trajectory"][:, 3] *= -1

        # Random vertical flip (negate y, negate sin θ)
        if np.random.random() < 0.5:
            sample["polygon_vertices"][:, 1] *= -1
            sample["start_pose"][1] *= -1
            sample["start_pose"][3] *= -1
            sample["goal_pose"][1] *= -1
            sample["goal_pose"][3] *= -1
            sample["trajectory"][:, 1] *= -1
            sample["trajectory"][:, 3] *= -1

        return sample


def make_push_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    val_frac: float = 0.1,
    num_workers: int = 4,
):
    """Create train/val dataloaders with an environment-level split.

    Split is by env file so no piece from a val env leaks into training.
    """
    full = PushTrajectoryDataset(data_dir, augment=False)
    n_files = len(full.files)
    n_val = max(1, int(n_files * val_frac))
    n_train = n_files - n_val

    train_idx = [i for i, (fi, _) in enumerate(full.index) if fi < n_train]
    val_idx = [i for i, (fi, _) in enumerate(full.index) if fi >= n_train]

    train_ds = torch.utils.data.Subset(
        PushTrajectoryDataset(data_dir, augment=True), train_idx)
    val_ds = torch.utils.data.Subset(full, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
