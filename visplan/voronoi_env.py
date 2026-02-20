import os
import tempfile

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from visplan.generation_utils import generate_voronoi_meshes, save_voronoi_assets
from visplan.env_utils import ManiSkillEnvUtils


@register_env("VoronoiReassembly-v1", max_episode_steps=10000)
class VoronoiReassembly(ManiSkillEnvUtils, BaseEnv):

    SUPPORTED_ROBOTS = ["panda"]

    def __init__(
        self,
        *args,
        num_voronoi_points: int = 5,
        side_length: float = 0.2,
        extrusion_height: float = 0.03,
        scale_factor: float = 0.9,
        placement_mode: str = "scattered",
        voronoi_seed: int = 0,
        **kwargs,
    ):
        assert placement_mode in ("scattered", "assembled"), (
            f"placement_mode must be 'scattered' or 'assembled', got '{placement_mode}'"
        )

        self.num_voronoi_points = num_voronoi_points
        self.side_length = side_length
        self.extrusion_height = extrusion_height
        self.scale_factor = scale_factor
        self.placement_mode = placement_mode
        self.voronoi_seed = voronoi_seed

        # Generate assets before super().__init__() because _load_scene needs them
        self._generate_voronoi_assets()

        super().__init__(*args, robot_uids="panda", **kwargs)

    # ------------------------------------------------------------------ #
    #  Asset generation
    # ------------------------------------------------------------------ #

    def _generate_voronoi_assets(self):
        """Pre-generate Voronoi polygon meshes, OBJs, and URDFs."""
        self.meshes, self.centroids = generate_voronoi_meshes(
            num_points=self.num_voronoi_points,
            side_length=self.side_length,
            scale_factor=self.scale_factor,
            extrusion_height=self.extrusion_height,
            seed=self.voronoi_seed,
        )

        # Use a persistent temp directory (lives as long as this env instance)
        self._asset_dir = tempfile.mkdtemp(prefix="voronoi_assets_")
        self.urdf_paths = save_voronoi_assets(
            self.meshes, self._asset_dir, seed=self.voronoi_seed
        )

        self.num_pieces = len(self.meshes)
        self.piece_names = [f"polygon_{i}" for i in range(self.num_pieces)]

    # ------------------------------------------------------------------ #
    #  ManiSkill config hooks
    # ------------------------------------------------------------------ #

    @property
    def _default_sim_config(self):
        return SimConfig(spacing=3)

    @property
    def _default_sensor_configs(self):
        # Camera 1: overhead looking down at the workspace
        cam1_pose = sapien_utils.look_at(
            eye=[0.0, 0.0, 0.6],
            target=[0.0, 0.0, 0.0],
        )
        # Camera 2: angled side view
        cam2_pose = sapien_utils.look_at(
            eye=[0.3, -0.4, 0.4],
            target=[0.0, 0.0, 0.0],
        )
        return [
            CameraConfig("overhead_cam", pose=cam1_pose, width=256, height=256, fov=np.pi / 2),
            CameraConfig("side_cam", pose=cam2_pose, width=256, height=256, fov=np.pi / 2),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.4, -0.6, 0.5],
            target=[0.0, 0.0, 0.02],
        )
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1.2, near=0.01, far=100)

    # ------------------------------------------------------------------ #
    #  Scene loading
    # ------------------------------------------------------------------ #

    def _load_agent(self, options: dict):
        self.robot_base_pose = sapien.Pose(p=[0.615, 0, 0], q=[1, 0, 0, 0])
        super()._load_agent(options, self.robot_base_pose)
        self.robot_links = self.agent.robot.get_links()

    def _load_scene(self, options: dict):
        # Table
        self.scene_builder = TableSceneBuilder(
            env=self, robot_init_qpos_noise=0.0
        )
        self.scene_builder.build()

        # URDF loader
        self.urdf_loader = self.scene.create_urdf_loader()
        self.urdf_loader.load_multiple_collisions_from_file = True

        # Build Voronoi piece actors
        self.objects = {}
        self.object_ids = {}
        self.actors = []

        for i, (name, urdf_path) in enumerate(zip(self.piece_names, self.urdf_paths)):
            actor_builders = self.urdf_loader.parse(urdf_path)["actor_builders"]
            builder = actor_builders[0]
            # Initial pose will be set properly in _initialize_episode
            builder.initial_pose = sapien.Pose(
                p=[0.0, 0.0, self.extrusion_height / 2 + 0.01],
                q=[1, 0, 0, 0],
            )
            actor = builder.build(name=name)
            self.objects[name] = actor
            self.object_ids[name] = actor.per_scene_id[0].item()
            self.actors.append(actor)

    # ------------------------------------------------------------------ #
    #  Episode initialization
    # ------------------------------------------------------------------ #

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)

            if self.placement_mode == "assembled":
                self._place_assembled()
            else:
                self._place_scattered()

    def _place_assembled(self):
        """Place pieces at their original Voronoi tiling positions."""
        for i, name in enumerate(self.piece_names):
            cx, cy = self.centroids[i]
            pose = np.array([
                cx, cy, self.extrusion_height / 2 + 0.001,
                1, 0, 0, 0,  # wxyz identity quaternion
            ])
            self.set_object_pose(name, pose)

    def _place_scattered(self):
        """Place pieces in a grid pattern on the table, spread out."""
        n = self.num_pieces
        cols = int(np.ceil(np.sqrt(n)))
        spacing = 0.12  # generous spacing for small pieces (~0.04-0.08m)

        # Center the grid at (0, 0) on the table
        offset_x = -(cols - 1) * spacing / 2
        offset_y = -(cols - 1) * spacing / 2

        for idx, name in enumerate(self.piece_names):
            row = idx // cols
            col = idx % cols
            x = offset_x + col * spacing
            y = offset_y + row * spacing
            pose = np.array([
                x, y, self.extrusion_height / 2 + 0.001,
                1, 0, 0, 0,
            ])
            self.set_object_pose(name, pose)

    # Pose helpers provided by ManiSkillEnvUtils mixin:
    # get_object_pose(), set_object_pose()

    # ------------------------------------------------------------------ #
    #  Velocity control
    # ------------------------------------------------------------------ #

    def set_piece_velocities(self, velocities: torch.Tensor, piece_indices: list = None):
        """
        Set linear velocities for one or more polygon pieces simultaneously.

        Args:
            velocities: Tensor of shape (M, N, 3) where
                M = number of pieces to control
                N = num_envs
                3 = (vx, vy, vz) in m/s
                Also accepts (N, 3) which is treated as M=1.
            piece_indices: List of M integer indices into self.actors specifying
                which pieces to control. If None, controls pieces 0..M-1.
        """
        if not isinstance(velocities, torch.Tensor):
            velocities = torch.tensor(velocities, dtype=torch.float32, device=self.device)

        if velocities.ndim == 2:
            velocities = velocities.unsqueeze(0)  # (N, 3) -> (1, N, 3)

        M = velocities.shape[0]

        if piece_indices is None:
            piece_indices = list(range(M))

        assert len(piece_indices) == M, (
            f"piece_indices length ({len(piece_indices)}) must match M ({M})"
        )
        assert all(0 <= i < self.num_pieces for i in piece_indices), (
            f"piece_indices must be in [0, {self.num_pieces}), got {piece_indices}"
        )

        for j, idx in enumerate(piece_indices):
            self.actors[idx].set_linear_velocity(velocities[j])

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()

    def sim_step(self, steps: int = 50):
        """Step physics without sending any robot action."""
        zero_action = torch.zeros(self.num_envs, self.action_space.shape[-1], device=self.device)
        for _ in range(steps):
            self.step(zero_action)
            if self.render_mode is not None:
                self.render()

    # ------------------------------------------------------------------ #
    #  Reward / evaluation stubs
    # ------------------------------------------------------------------ #

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

    def compute_dense_reward(self, obs=None, action=None, info=None):
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def compute_normalized_dense_reward(self, obs=None, action=None, info=None):
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
