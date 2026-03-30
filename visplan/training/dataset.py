"""Dataset and dataloader for Voronoi reassembly trajectory prediction.

Each sample is a single (piece, episode) pair:
  - Input:  start_image (3, H, W), goal_image (3, H, W), piece_mask (1, H, W)
  - Target: trajectory (T, 4) — SE(2) with θ encoded as (cos θ, sin θ)

Trajectories are normalised:
  - xy divided by visible_range (≈ 0.6 m) so values lie in [-1, 1]
  - θ encoded as (cos θ, sin θ) to avoid discontinuities
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VoronoiReassemblyDataset(Dataset):
    """Per-piece trajectory prediction dataset.

    Args:
        data_dir:      Path containing episode_XXXXXX.npz files.
        visible_range: Half-extent of the overhead camera view (metres).
                       Used to normalise xy positions to [-1, 1].
        augment:       If True, apply random SE(2) augmentation to the whole
                       scene (flips + 90° rotations — cheap, label-preserving).
    """

    def __init__(self, data_dir: str, visible_range: float = 0.6,
                 augment: bool = False):
        self.data_dir = data_dir
        self.visible_range = visible_range
        self.augment = augment

        self.files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No episode files found in {data_dir}")

        # Build flat index: (file_idx, piece_idx)
        self.index = []
        for fi, fpath in enumerate(self.files):
            d = np.load(fpath)
            n = int(d["num_pieces"])
            for pi in range(n):
                self.index.append((fi, pi))
            d.close()

    def __len__(self):
        return len(self.index)

    def _normalise_traj(self, traj: np.ndarray) -> np.ndarray:
        """Convert (T, 3) [x, y, θ] → (T, 4) [x̃, ỹ, cos θ, sin θ]."""
        xy = traj[:, :2] / self.visible_range  # [-1, 1]
        cos_t = np.cos(traj[:, 2:3])
        sin_t = np.sin(traj[:, 2:3])
        return np.concatenate([xy, cos_t, sin_t], axis=-1).astype(np.float32)

    def __getitem__(self, idx):
        fi, pi = self.index[idx]
        d = np.load(self.files[fi])

        # Images: (H, W, 3) uint8 → (3, H, W) float32 in [0, 1]
        start_img = d["start_image"].astype(np.float32) / 255.0
        start_img = np.transpose(start_img, (2, 0, 1))  # (3, H, W)

        goal_img = d["goal_image"].astype(np.float32) / 255.0
        goal_img = np.transpose(goal_img, (2, 0, 1))

        # Piece mask: (H, W) bool → (1, H, W) float32
        mask = d["piece_masks"][pi].astype(np.float32)[None]

        # Trajectory: (T, 3) → normalised (T, 4)
        traj = self._normalise_traj(d["trajectories"][pi])

        # Start and goal pose (normalised, for optional conditioning)
        start_pose = self._normalise_traj(d["start_poses"][pi:pi+1])[0]  # (4,)
        goal_pose = self._normalise_traj(d["goal_poses"][pi:pi+1])[0]    # (4,)

        d.close()

        if self.augment:
            start_img, goal_img, mask, traj, start_pose, goal_pose = (
                self._augment(start_img, goal_img, mask, traj, start_pose, goal_pose)
            )

        return {
            "start_image": torch.from_numpy(start_img),    # (3, H, W)
            "goal_image": torch.from_numpy(goal_img),      # (3, H, W)
            "piece_mask": torch.from_numpy(mask),           # (1, H, W)
            "trajectory": torch.from_numpy(traj),           # (T, 4)
            "start_pose": torch.from_numpy(start_pose),     # (4,)
            "goal_pose": torch.from_numpy(goal_pose),       # (4,)
        }

    def _augment(self, start_img, goal_img, mask, traj, start_pose, goal_pose):
        """Random horizontal flip and 90°-rotation augmentation.

        These are exact symmetries of the overhead-view SE(2) problem:
        flipping/rotating the image and correspondingly transforming the
        trajectory labels produces an equally valid training sample.
        """
        # Random horizontal flip
        if np.random.random() < 0.5:
            start_img = start_img[:, :, ::-1].copy()
            goal_img = goal_img[:, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()
            # Flip x, negate sin θ
            traj[:, 0] *= -1
            traj[:, 3] *= -1
            start_pose[0] *= -1; start_pose[3] *= -1
            goal_pose[0] *= -1; goal_pose[3] *= -1

        # Random vertical flip
        if np.random.random() < 0.5:
            start_img = start_img[:, ::-1, :].copy()
            goal_img = goal_img[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
            # Flip y, negate sin θ
            traj[:, 1] *= -1
            traj[:, 3] *= -1
            start_pose[1] *= -1; start_pose[3] *= -1
            goal_pose[1] *= -1; goal_pose[3] *= -1

        return start_img, goal_img, mask, traj, start_pose, goal_pose


def make_dataloaders(data_dir: str, batch_size: int = 64,
                     val_frac: float = 0.1, num_workers: int = 4,
                     visible_range: float = 0.6, **kwargs):
    """Create train/val dataloaders with a simple episode-level split.

    The split is by *episode file*, not by (episode, piece) sample, so
    no piece from a validation episode leaks into training.
    """
    full = VoronoiReassemblyDataset(data_dir, visible_range=visible_range,
                                    augment=False)
    n_files = len(full.files)
    n_val = max(1, int(n_files * val_frac))
    n_train = n_files - n_val

    # Split index by file
    train_idx = [i for i, (fi, _) in enumerate(full.index) if fi < n_train]
    val_idx = [i for i, (fi, _) in enumerate(full.index) if fi >= n_train]

    train_ds = torch.utils.data.Subset(
        VoronoiReassemblyDataset(data_dir, visible_range=visible_range,
                                 augment=True),
        train_idx,
    )
    val_ds = torch.utils.data.Subset(full, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
