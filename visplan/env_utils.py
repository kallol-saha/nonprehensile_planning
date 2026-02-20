import json
import os
import copy
import sapien
import torch
import numpy as np
from omegaconf import OmegaConf
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.utils import randomization
# from plan_scene_builder import TableSceneBuilder
from mani_skill.utils.building import articulations, actors
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from sapien.physx import PhysxRigidBodyComponent

from visplan.robot_controller import RobotController

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from tqdm import tqdm
from visplan.utils import (
    load_grasps, 
    to_numpy_pose, 
    to_sapien_pose, 
    to_torch_pose,
    flip_quat_about_z,
    load_acronym_object_and_grasps,
    point_in_cuboid
)

from visplan.submodules.robo_utils.robo_utils.visualization.plotting import plot_pcd, plot_pcd_with_highlighted_segment
from visplan.submodules.robo_utils.robo_utils.visualization.point_cloud_structures import make_gripper_visualization
from visplan.submodules.robo_utils.robo_utils.conversion_utils import (
    pose_to_transformation, 
    invert_transformation, 
    transform_pcd, 
    transformation_to_pose,
    move_pose_along_local_z,
    furthest_point_sample
)

from visplan.generation_utils import (
    sample_object_pose_on_table,
    sample_object_pose_on_table_multi_object,
    sample_object_pose_on_shelf,
    sample_point_in_fixed_rectangle,
    sample_point_in_fixed_rectangle_uniformly,
    compute_ray_box_intersection
)

CUROBO_ASSETS_PATH = "visplan/submodules/curobo/src/curobo/content/assets/"
POST_GRASP_LIFT = 0.15
# GRASP_DEPTH = 0.2
GRASP_DEPTH = 0.25
GRASP_RETRACTION_DISTANCE = 0.15
EE_LINK_CENTER_TO_GRIPPER_TIP = 0.13

CUROBO_SUCCESSES = 0

OPEN = 1
CLOSED = -1


def load_checkpoint_for_eval(checkpoint_path: str, model: torch.nn.Module) -> torch.nn.Module:
    """Load from checkpoint."""
    print(f"=> trying checkpoint '{checkpoint_path}'")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    model_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )
    # Load weights flexibly
    msn, unxpct = model.load_state_dict(model_dict["weight"], strict=False)
    if msn:
        print(f"Missing keys (not found in checkpoint): {len(msn)}")
        print(msn)
    if unxpct:
        print(f"Unexpected keys (ignored): {len(unxpct)}")
        print(unxpct)
    if not msn and not unxpct:
        print("All keys matched successfully!")

    print(f"=> loaded successfully '{checkpoint_path}' (step {model_dict.get('iter', 0)})")
    del model_dict
    torch.cuda.empty_cache()

    return model


class ManiSkillEnvUtils:
    """
    Mixin class containing utility methods for ManiSkill environments.
    
    NOTE: This mixin requires the class to also inherit from BaseEnv.
    It accesses BaseEnv attributes like self.scene, self.device, etc.
    """
    
    def set_gripper_friction(self, friction_coefficient=2.0):

        # Set high friction for the finger links
        self.finger_entities = [link._objs[0].entity for link in self.robot_links[-4:]]
        self.finger_components = [entity.find_component_by_type(PhysxRigidBodyComponent) for entity in self.finger_entities]

        for component in self.finger_components:
            if component is not None:
                for shape in component.collision_shapes:
                    shape.physical_material.set_static_friction(friction_coefficient)
    
    def stall(self, num_steps=None):

        print("Left click to stop")

        while True:
            if self.render_mode is not None:
                self.render()
            if num_steps is not None:
                num_steps -= 1
            if num_steps == 0:
                break
    
    def capture_video_frame(self):
        if len(self.scene.human_render_cameras.items()) == 0:
            return
        render_img = self.scene.get_human_render_camera_images()['render_camera']
        render_img = render_img.cpu().detach().numpy()
        self.video_frames = np.concatenate([self.video_frames, render_img], axis=0)
    
    def capture_image(self):
        if len(self.scene.human_render_cameras.items()) == 0:
            return
        # render_img = self.scene.get_human_render_camera_images()['render_camera']
        render_img = self.render_rgb_array()
        render_img = render_img.cpu().detach().numpy()[0, ..., :3]
        return render_img
    
    def get_point_cloud(self, object_name=None):

        pcd_obs = self.get_obs()['pointcloud']

        pcd = pcd_obs['xyzw'][0, :, :3].to(self.device)     # tensor(N, 3)
        rgb = pcd_obs['rgb'][0].to(self.device)         # In RGB uint8 format (0-255) --> tensor(N, 3)
        seg = pcd_obs['segmentation'][0].to(self.device)         # tensor(N, 1) int16

        mask = (pcd[:, 0] > self.pcd_bounds[0]) & \
               (pcd[:, 0] < self.pcd_bounds[3]) & \
               (pcd[:, 1] > self.pcd_bounds[1]) & \
               (pcd[:, 1] < self.pcd_bounds[4]) & \
               (pcd[:, 2] > self.pcd_bounds[2]) & \
               (pcd[:, 2] < self.pcd_bounds[5])

        pcd = pcd[mask]
        rgb = rgb[mask]
        seg = seg[mask]

        # 12, 14 --> right finger, left finger from from the robot's perspective
        # 16 --> Table
        # Objects start from 18
        if object_name is not None:
            # Filter to only the specified object and shelf
            object_id = self.object_ids[object_name]
            object_ids = [object_id]
        else:
            # Use all objects
            object_ids = [self.object_ids[object_name] for object_name in self.object_names]
        
        scene_ids = object_ids + [self.table_id]

        # Remove exactly half of the table points randomly (keep other half):
        table_mask = (seg == self.table_id)[..., 0]
        # true_idx = torch.where(table_mask)[0]
        # num_true = true_idx.numel()
        # # randomly select quarter of True indices to KEEP
        # perm = torch.randperm(num_true, device=table_mask.device)
        # to_keep = true_idx[perm[: num_true // 4]]

        # # build a keep mask: keep all non-table points, and only the selected half of table points
        # keep_mask = (~table_mask).clone()
        # keep_mask[to_keep] = True

        keep_mask = ~table_mask     # Keep all non-table points

        pcd = pcd[keep_mask]
        rgb = rgb[keep_mask]
        seg = seg[keep_mask]

        robot_mask = torch.zeros_like(seg, dtype=torch.bool, device=self.device)
        for id in range(16):
            robot_mask = robot_mask | (seg == id)

        # Invert robot mask, mask out robot points
        non_robot_mask = ~robot_mask[..., 0]
        pcd = pcd[non_robot_mask]
        rgb = rgb[non_robot_mask]
        seg = seg[non_robot_mask]
        
        # If object_name is specified, filter to only that object and shelf
        # if object_name is not None:
        #     object_shelf_mask = (seg == object_id) | (seg == shelf_id)
        #     pcd = pcd[object_shelf_mask.squeeze()]
        #     rgb = rgb[object_shelf_mask.squeeze()]
        #     seg = seg[object_shelf_mask.squeeze()]
        
        # Create scene mask: 1 for points belonging to objects/table, 0 otherwise (required for collision avoidance)
        scene_mask = torch.zeros_like(seg, dtype=torch.bool, device=self.device)
        for id in scene_ids:
            scene_mask = scene_mask | (seg == id)

        # Filter out zero points
        non_zero_mask = (pcd[:, 0] != 0) & (pcd[:, 1] != 0) & (pcd[:, 2] != 0)
        pcd = pcd[non_zero_mask]
        rgb = rgb[non_zero_mask]
        seg = seg[non_zero_mask]

        return pcd, rgb, seg, scene_mask
    
    def add_pointcloud_noise(self, points):
        sigma = np.random.uniform(0.002, 0.006)
        noise = torch.randn_like(points) * sigma
        noisy_points = points + noise

        # Dropout
        drop_prob = np.random.uniform(0.01, 0.05)
        keep_mask = torch.rand(len(points)) > drop_prob
        noisy_points = noisy_points[keep_mask]

        # Outliers
        n_outliers = np.random.randint(50, 200)
        outliers = (torch.rand((n_outliers, 3), device=points.device) - 0.5) * 2.0
        noisy_points = torch.cat([noisy_points, outliers], dim=0)

        return noisy_points
    
    def get_observed_pcd(self, object_name):

        scene_pcd, scene_rgb, scene_seg, scene_mask = self.get_point_cloud(object_name)
        noisy_pcd = self.add_pointcloud_noise(scene_pcd)
        noisy_pcd = furthest_point_sample(noisy_pcd, num_points=4096)
        noisy_pcd[:, 0] = noisy_pcd[:, 0] + 0.615    # TODO: This is a hack to move the point cloud to the robot base frame.

        return noisy_pcd
       
    def get_sampled_point_cloud(
        self, 
        num_points_per_actor: int = 10000,
        env_idx: int = 0,
        actors_to_ignore: list = ['table-workspace']
        ):

        sampled_points = []

        for actor in self.actors:
            if actor.name in actors_to_ignore:
                continue
            mesh = actor.get_collision_meshes()[env_idx]            # Initialize with env 0
            points, _ = trimesh.sample.sample_surface(mesh, num_points_per_actor)
            points = np.array(points)
            sampled_points.append(points)
            
        return np.concatenate(sampled_points, axis=0)

    def get_sampled_point_cloud2(
        self, 
        num_points: int = 4096,
        env_idx: int = 0,
        actors_to_ignore: list = ['table-workspace']
        ):

        # Count actors to sample from
        actors_to_sample = [actor for actor in self.actors if actor.name not in actors_to_ignore]
        num_actors = len(actors_to_sample)
        
        if num_actors == 0:
            return np.zeros((num_points, 3))
        
        # Divide points equally among actors
        points_per_actor = num_points // num_actors
        remainder = num_points % num_actors
        
        sampled_points = []
        for i, actor in enumerate(actors_to_sample):
            # Distribute remainder points to first 'remainder' actors
            num_points_this_actor = points_per_actor + (1 if i < remainder else 0)
            mesh = actor.get_collision_meshes()[env_idx]
            points, _ = trimesh.sample.sample_surface(mesh, num_points_this_actor)
            points = np.array(points)
            sampled_points.append(points)
            
        return np.concatenate(sampled_points, axis=0)

    def get_sampled_actor_point_cloud(self, actor, env_idx: int = 0, num_points: int = 2048):
        mesh = actor.get_collision_meshes()[env_idx]
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        points = np.array(points)
        return points
    
    def get_object_pose(self, object_name=None):
        """
        Returns:
            object_pose: torch.Tensor (num_envs, 7)
        """
        if isinstance(object_name, str):
            p = self.objects[object_name].pose.p
            q = self.objects[object_name].pose.q
            return torch.cat([p, q], dim=-1)
        else:
            p = object_name.pose.p
            q = object_name.pose.q
            return torch.cat([p, q], dim=-1)
    
    def set_object_pose(self, object_name, pose: torch.Tensor):
        """
        object_name: str or Articulation Object
        Pose is a torch tensor (num_envs, 7) or (n, 7) where n <= num_envs
        If n < num_envs, repeats the last element until num_envs is reached.
        If pose is of shape (7,), it is broadcasted to (num_envs, 7)
        """

        if not isinstance(pose, torch.Tensor):
            pose = torch.tensor(pose, dtype=torch.float32, device=self.device)
        
        if pose.ndim == 1:
            pose = pose.unsqueeze(0)
        
        if pose.shape[0] < self.num_envs:
            # Repeat the last element until num_envs
            last_element = pose[-1:]  # (1, 7)
            n_repeats = self.num_envs - pose.shape[0]
            repeated = last_element.repeat(n_repeats, 1)
            pose = torch.cat([pose, repeated], dim=0)
        elif pose.shape[0] > self.num_envs:
            # Truncate if longer
            pose = pose[:self.num_envs]

        pose = pose.to(self.device).to(torch.float32)

        # TODO: Should this function be parallelized, since it is only used in a single object context.
        if isinstance(object_name, str):
            self.objects[object_name].set_pose(Pose.create_from_pq(p=pose[..., :3], q=pose[..., 3:]))
        else:
            object_name.set_pose(Pose.create_from_pq(p=pose[..., :3], q=pose[..., 3:]))

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
    
    @property
    def object_poses_tensor(self):
        object_poses = torch.zeros((self.num_envs, self.num_objects, 7), device=self.device, dtype=torch.float32)
        for i, object_name in enumerate(self.object_names):
            object_poses[:, i, :] = self.get_object_pose(object_name)
        return object_poses
    
    @property
    def object_centric_state(self):
        """
        Returns object-centric state including poses and dimensions.
        Returns:
            object_state: torch.Tensor (num_envs, num_objects, 10)
                - First 7 values: pose (x, y, z, qw, qx, qy, qz)
                - Last 3 values: dimensions (dim_x, dim_y, dim_z)
        """
        object_state = torch.zeros((self.num_envs, self.num_objects, 10), device=self.device, dtype=torch.float32)
        for i, object_name in enumerate(self.object_names):
            # Get pose (7 values)
            object_state[:, i, :7] = self.get_object_pose(object_name)
            
            # Get dimensions from bounds (3 values)
            if hasattr(self, 'bounds') and object_name in self.bounds:
                bounds = self.bounds[object_name]
                dim_x = float(bounds[0][1] - bounds[0][0])
                dim_y = float(bounds[1][1] - bounds[1][0])
                dim_z = float(bounds[2][1] - bounds[2][0])
                object_state[:, i, 7] = dim_x
                object_state[:, i, 8] = dim_y
                object_state[:, i, 9] = dim_z
        return object_state
    
    def set_object_poses_tensor(self, object_poses: torch.Tensor):
        """
        object_poses: torch.Tensor (num_envs, num_objects, 7)
        If object_poses is of shape (num_objects, 7), then it is broadcasted to (num_envs, num_objects, 7)
        """
        if object_poses.ndim == 2:
            object_poses = object_poses.unsqueeze(0).repeat(self.num_envs, 1, 1)
        elif object_poses.ndim == 3 and object_poses.shape[0] != self.num_envs:
            object_poses = object_poses.repeat(self.num_envs, 1, 1)

        object_poses = object_poses.to(self.device).to(torch.float32)

        assert object_poses.shape == (self.num_envs, self.num_objects, 7)
        
        for i, object_name in enumerate(self.object_names):
            self.set_object_pose(object_name, object_poses[:, i, :])
    
    def control_robot(self, qpos: torch.Tensor):
        """
        qpos: torch.Tensor (..., 9) or (num_envs, 9)
        """
        # Scale 0 to 0.04 to -1 to 1
        gripper_state = (qpos[..., -1] - 0.02) / 0.02
        action = torch.cat([qpos[..., :-2], gripper_state.unsqueeze(-1)], dim=-1)
        # Convert to numpy for step() if needed
        obs, reward, done, info, _ = self.step(action)
        return obs, reward, done, info
    
    def get_robot_qpos(self):
        qpos = self.agent.robot.get_qpos()
        return qpos

    def get_joint_state(self):
        qpos = self.get_robot_qpos()
        joint_state = qpos[..., :7]
        return joint_state
    
    def get_gripper_state(self):
        qpos = self.get_robot_qpos()
        gripper_state = qpos[..., -2:]
        return gripper_state

    def get_gripper_pose(self):
        gripper_pose = self.agent.robot.get_links()[-5].pose.raw_pose
        return gripper_pose
    
    def set_robot_qpos(self, qpos):
        """
        qpos => torch tensor (num_envs, 9) including gripper widths 
        """
        self.agent.robot.set_qpos(qpos)
    
    def set_joint_state(self, joint_state):
        """
        joint_state => torch tensor (num_envs, 7) or (1, 7) or (n, 7) where n < num_envs
        If n < num_envs, repeats the last element until num_envs is reached.
        """
        assert isinstance(joint_state, torch.Tensor)
        
        current_gripper_state = self.get_gripper_state()
        # Ensure shapes match for concatenation
        if joint_state.ndim == 1:
            joint_state = joint_state.unsqueeze(0)
        
        if joint_state.shape[0] < self.num_envs:
            # Repeat the last element until num_envs
            last_element = joint_state[-1:]  # (1, 7)
            n_repeats = self.num_envs - joint_state.shape[0]
            repeated = last_element.repeat(n_repeats, 1)
            joint_state = torch.cat([joint_state, repeated], dim=0)
        elif joint_state.shape[0] > self.num_envs:
            # Truncate if longer
            joint_state = joint_state[:self.num_envs]

        if current_gripper_state.ndim == 1:
            current_gripper_state = current_gripper_state.unsqueeze(0).repeat(self.num_envs, 1)
        elif current_gripper_state.ndim == 2 and current_gripper_state.shape[0] != self.num_envs:
            current_gripper_state = current_gripper_state.repeat(self.num_envs, 1)
        
        qpos = torch.cat([joint_state.to(current_gripper_state.device), current_gripper_state], dim=-1)
        self.set_robot_qpos(qpos)

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
    
    def wait(self, steps: int = 50):

        current_robot_qpos = self.get_robot_qpos()

        i = 0
        while i < steps:
            _ = self.control_robot(current_robot_qpos)
            if self.render_mode is not None:
                self.render()
            i += 1
    
    def wait_for_stability(self, max_steps: int = 200):
        """
        Waits for the objects to be stable, by checking if the object poses are changing.
        Stops waiting after max_steps iterations to avoid hanging forever.
        """

        current_robot_qpos = self.get_robot_qpos()
        
        prev_object_poses = self.object_poses_tensor.clone()

        
        tolerance = 8e-3
        curr_error = torch.inf
        steps = 0

        while curr_error > tolerance and steps < max_steps:
            _ = self.control_robot(current_robot_qpos)

            curr_object_poses = self.object_poses_tensor.clone()

            error = torch.abs(curr_object_poses - prev_object_poses)
            curr_error = torch.max(error)

            prev_object_poses = curr_object_poses.clone()
            steps += 1
    
    def synchronize_object_poses(self, env_idx: int):
        """
        Synchronize the object poses to the given environment index.
        """
        object_poses = self.object_poses_tensor
        object_poses_to_match = object_poses[env_idx]
        self.set_object_poses_tensor(object_poses_to_match)
    
    
    def wait_for_synchronized_stability(self):
        """
        Wait for stability but make sure all the object poses are synchronized.
        """
        
        initial_object_poses = self.object_poses_tensor.clone()    
        self.wait_for_stability()
        final_object_poses = self.object_poses_tensor.clone()

        error = torch.abs(final_object_poses - initial_object_poses)
        error_per_env = error.amax(dim=(1, 2))
        env_with_min_error = torch.argmin(error_per_env)
        object_poses_to_match = final_object_poses[env_with_min_error]

        self.set_object_poses_tensor(object_poses_to_match)
    

        
    