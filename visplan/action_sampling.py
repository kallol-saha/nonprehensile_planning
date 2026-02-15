import json
import os
import copy
import time
import sapien
from sympy.core import I
import torch
import shutil
import numpy as np
from typing import Tuple
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
from visplan.action_primitives import ActionPrimitives
from visplan.env_utils import ManiSkillEnvUtils
from visplan.graphs.directed_acyclic import DAG

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
    flip_pose_quat_about_z,
    load_acronym_object_and_grasps,
    point_in_cuboid,
    generate_placement_quaternions
)

from visplan.submodules.robo_utils.robo_utils.visualization.plotting import (
    plot_pcd, 
    plot_pcd_with_highlighted_segment
)
from visplan.submodules.robo_utils.robo_utils.visualization.point_cloud_structures import make_gripper_visualization
from visplan.submodules.robo_utils.robo_utils.conversion_utils import (
    pose_to_transformation, 
    invert_transformation, 
    transform_pcd, 
    transformation_to_pose,
    move_pose_along_local_z,
    move_pose_along_local_x,
    move_pose_along_local_y,
    furthest_point_sample
)
from models.flowmatch_actor.modeling.policy.denoise_actor_3d_packing import DenoiseActor
from models.flowmatch_actor.utils.common_utils import count_parameters

from visplan.generation_utils import (
    create_shelf, 
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

class ActionSampler:
    """
    Action sampling for robot manipulation using cuRobo motion planning.
    Inherits from ActionPrimitives to access all action sampling functionality.
    """
    
    def __init__(self, env):

        self.env = env
        if hasattr(env, 'placement_model'):
            self.placement_model = env.placement_model

    # ------------------------ GRASP SAMPLING ------------------------ #

    def get_object_grasp_poses(self, object_name, env_idx: int = 0, grasp_poses=None):
        
        if grasp_poses is None:
            grasp_poses = self.env.grasps[object_name]

        object_pose = self.env.get_object_pose(object_name)[env_idx].cpu().detach().numpy()

        # Using conversion utils:
        grasp_transformation = pose_to_transformation(grasp_poses, format='wxyz')
        object_transformation = pose_to_transformation(object_pose, format='wxyz')
        grasp_transformation = object_transformation @ grasp_transformation
        transformed_grasp_poses = transformation_to_pose(grasp_transformation, format='wxyz')

        transformed_grasp_poses = torch.tensor(transformed_grasp_poses, dtype=torch.float32, device="cuda:0")

        return transformed_grasp_poses

    def sample_grasp_poses(self):

        grasp_poses = []
        grasp_costs = []
        grasp_object_ids = []
        
        for object_index in range(len(self.env.object_names)):
            # Get the transformed grasp pose tensors for the specified object:
            object_name = self.env.object_names[object_index]
            
            # Skip objects that are already in the shelf
            if self.env.check_object_inside_shelf(object_name, env_idx=0):
                continue
            
            obj_grasp_poses = self.get_object_grasp_poses(
                object_name, 
                env_idx=0,
                grasp_poses=self.env.grasps[object_name]
            )
            # self.visualize_grasp_poses(grasp_poses)
            grasp_poses.append(torch.tensor(obj_grasp_poses, device="cuda:0", dtype=torch.float32))
            grasp_costs.append(torch.tensor(self.env.grasp_costs[object_name], device="cuda:0", dtype=torch.float32))
            grasp_object_ids.append(torch.full((obj_grasp_poses.shape[0],), object_index, device="cuda:0", dtype=torch.int64))

        if len(grasp_poses) == 0:
            return torch.zeros((0, 7), device="cuda:0", dtype=torch.float32)
        
        grasp_poses = torch.cat(grasp_poses, dim=0)
        grasp_costs = torch.cat(grasp_costs, dim=0)
        grasp_object_ids = torch.cat(grasp_object_ids, dim=0)

        print(f"Doing IK for grasps")
        curobo_grasp_poses = grasp_poses.clone()
        curobo_grasp_poses[..., 0] = curobo_grasp_poses[..., 0] + 0.615
        _, success = self.env.action_primitives.inverse_kinematics(curobo_grasp_poses)

        feasible_indices = torch.where(success[:, 0])[0]

        if len(feasible_indices) == 0:
            return torch.zeros((0, 7), device="cuda:0", dtype=torch.float32)

        grasp_poses = grasp_poses[feasible_indices]
        grasp_costs = grasp_costs[feasible_indices].unsqueeze(-1)
        grasp_object_ids = grasp_object_ids[feasible_indices].unsqueeze(-1)

        sampled_actions = torch.cat([grasp_poses, grasp_costs, grasp_object_ids], dim=-1)

        # (N, 9) tensor, with (x, y, z, qw, qx, qy, qz, grasp_cost, grasp_object_id)
        return sampled_actions
    
    # ------------------------ PLACEMENT SAMPLING ------------------------ #

    def sample_points_in_shelf(
        self,
        # Depth and width tolerance are based on the Franka Gripper dimensions. 
        # The gripper cylinder diameter is 0.12, the hand width is 0.2, and finger to hand center is 0.105 (here I use 0.08)
        depth_tolerance=0.08,
        width_tolerance=0.2,
        extension_distance=0.,
        height_tolerance=0.08,
        n = 100,
        ):

        center_points = sample_point_in_fixed_rectangle(
            fixed_rect_center=(self.env.shelf_position[0], self.env.shelf_position[1]),
            fixed_rect_dimensions=(self.env.shelf_depth - depth_tolerance, self.env.shelf_width - width_tolerance),
            fixed_rect_angle=self.env.shelf_rotation, # + np.pi/2,
            num_points=n
        )
        cz = self.env.shelf_distance_from_floor + np.random.uniform(height_tolerance, self.env.shelf_height - height_tolerance, size=n)
        cz = np.clip(
            cz, 
            self.env.shelf_distance_from_floor + height_tolerance, 
            self.env.shelf_distance_from_floor + self.env.shelf_height - height_tolerance
            )[:, np.newaxis]

        points = np.concatenate([center_points, cz], axis=-1)

        return points

    def sample_points_outside_shelf(
        self,
        extension_distance=0.2,
        height_tolerance=0.08,
        width_tolerance=0.1,
        n = 100,
        ):

        direction_vec = np.array([np.cos(self.env.shelf_rotation), np.sin(self.env.shelf_rotation)])
        new_shelf_center = self.env.shelf_position[:2] - (((extension_distance / 2) + self.env.shelf_depth) / 2) * direction_vec
        
        center_points = sample_point_in_fixed_rectangle(
            fixed_rect_center=(new_shelf_center[0], new_shelf_center[1]),
            fixed_rect_dimensions=(extension_distance, self.env.shelf_width - width_tolerance),
            fixed_rect_angle=self.env.shelf_rotation, # + np.pi/2,
            num_points=n
        )
        cz = self.env.shelf_distance_from_floor + np.random.uniform(height_tolerance, self.env.shelf_height - height_tolerance, size=n)
        cz = np.clip(
            cz, 
            self.env.shelf_distance_from_floor + height_tolerance, 
            self.env.shelf_distance_from_floor + self.env.shelf_height - height_tolerance
            )[:, np.newaxis]

        points = np.concatenate([center_points, cz], axis=-1)

        return points

    def sample_gripper_poses_in_shelf(
        self, 
        n = 1, 
        object_name = None,
        depth_tolerance=0.05,
        width_tolerance=0.1,
        height_tolerance=0.08
        ):

        points = self.sample_points_in_shelf(n = n)
        points = torch.tensor(points, device="cuda:0", dtype=torch.float32)

        rotation_to_orient_with_shelf = R.from_euler('yz', [np.pi/2, self.env.shelf_rotation])
        base_quaternion = rotation_to_orient_with_shelf.as_quat().squeeze()[[3,0,1,2]]
        base_quaternion = flip_quat_about_z(base_quaternion)  # Otherwise the object is upside down.
        
        # Apply random rotations: X rotation (-25 to +25 deg), then Z rotation (-5 to +5 deg)
        quaternions = []
        for i in range(n):
            # Sample random angles
            x_angle_deg = np.random.uniform(-25, 25)
            z_angle_deg = np.random.uniform(-5, 5)
            
            # Convert base quaternion to Rotation object
            base_rot = R.from_quat(base_quaternion[[1, 2, 3, 0]])  # Convert from wxyz to xyzw
            
            # Apply X rotation, then Z rotation
            x_rot = R.from_euler('x', x_angle_deg, degrees=True)
            z_rot = R.from_euler('z', z_angle_deg, degrees=True)
            combined_rot = base_rot * x_rot * z_rot  # Apply X first, then Z
            
            # Convert back to wxyz format
            quat_xyzw = combined_rot.as_quat()  # Returns [x, y, z, w]
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # Convert to [w, x, y, z]
            quaternions.append(quat_wxyz)
        
        quaternions = np.array(quaternions)  # (n, 4)
        quaternions = torch.tensor(quaternions, device="cuda:0", dtype=torch.float32)

        shelf_poses = torch.zeros((n, 7), device="cuda:0", dtype=torch.float32)
        shelf_poses[:, :3] = points
        shelf_poses[:, 3:] = quaternions

        curobo_shelf_poses = move_pose_along_local_z(shelf_poses, -EE_LINK_CENTER_TO_GRIPPER_TIP)

        return curobo_shelf_poses

        # curobo_shelf_poses = torch.tensor(shelf_poses, device="cuda:0", dtype=torch.float32)
        # curobo_shelf_poses = shelf_poses.clone()
        # curobo_shelf_poses[..., 0] = curobo_shelf_poses[..., 0] + 0.615

        # if object_name is not None:
        #     self.action_primitives.attach_object_to_robot(object_name)

        # shelf_joints, ik_successes = self.action_primitives.inverse_kinematics(curobo_shelf_poses)

        # if object_name is not None:
        #     self.action_primitives.detach_object_from_robot(object_name)

        # shelf_poses = shelf_poses[torch.where(ik_successes[:, 0])[0]]

        # return shelf_poses, shelf_joints
    
    def sample_gripper_poses_outside_shelf(
        self, 
        n = 1, 
        object_name = None,
        extension_distance=0.12,
        height_tolerance=0.08,
        width_tolerance=0.1,
        ):

        points = self.sample_points_outside_shelf(
            extension_distance=extension_distance,
            height_tolerance=height_tolerance,
            width_tolerance=width_tolerance,
            n = n
        )
        points = torch.tensor(points, device="cuda:0", dtype=torch.float32)

        # Get quaternions from generate_placement_quaternions
        quaternions = generate_placement_quaternions(num_quaternions=n, target_xy_angle=self.env.shelf_rotation)  # (n, 4) in wxyz format
        quaternions = torch.tensor(quaternions, device="cuda:0", dtype=torch.float32)

        shelf_poses = torch.zeros((n, 7), device="cuda:0", dtype=torch.float32)
        shelf_poses[:, :3] = points
        shelf_poses[:, 3:] = quaternions

        curobo_shelf_poses = move_pose_along_local_z(shelf_poses, -EE_LINK_CENTER_TO_GRIPPER_TIP)

        return curobo_shelf_poses    

    def sample_placement_poses(self, object_index: int, batch_size: int = 1):

        # TODO: Environment should already be reset to the desired state before calling this function.
        # TODO: So environment 0 should be the one from which everything is loaded.

        object_name = self.env.object_names[object_index]
        num_points = 4096   # TODO: This is hardcoded
        shelf_pcd = self.env.get_sampled_actor_point_cloud(self.env.shelf, num_points = num_points // 2)
        object_pcd = self.env.get_sampled_actor_point_cloud(self.env.objects[object_name], env_idx = 0, num_points = num_points // 2)

        input_pcd = np.concatenate([shelf_pcd, object_pcd], axis=0)
        input_pcd = torch.tensor(input_pcd, device="cuda:0", dtype=torch.float32)
        input_pcd = input_pcd.unsqueeze(0).repeat(batch_size, 1, 1)
        input_pcd[..., 0] = input_pcd[..., 0] + 0.615

        gripper_pose = self.env.action_primitives.fk(self.env.get_joint_state()[0].to("cuda:0"))
        gripper_pose = gripper_pose.unsqueeze(0).repeat(batch_size, 1, 1)
        zeros = torch.zeros((batch_size, 1, 1), device=gripper_pose.device, dtype=gripper_pose.dtype)   # For the gripper being closed
        gripper_pose = torch.cat([gripper_pose, zeros], dim=-1)

        max_batch_size = 4      # Reduce according to available CUDA memory
        all_placement_poses = []
        
        with torch.no_grad():
            for i in range(0, batch_size, max_batch_size):
                end_idx = min(i + max_batch_size, batch_size)
                chunk_pcd = input_pcd[i:end_idx]
                chunk_proprioception = gripper_pose[i:end_idx]
                
                placement_poses_chunk = self.placement_model(
                    gt_trajectory = None,
                    pcd = chunk_pcd,
                    proprioception = chunk_proprioception,
                    run_inference = True
                )
                
                placement_poses_chunk = placement_poses_chunk.squeeze(0)
                all_placement_poses.append(placement_poses_chunk)
        
        placement_poses = torch.cat(all_placement_poses, dim=0)
        placement_poses[..., 0] = placement_poses[..., 0] - 0.615
        placement_poses = placement_poses[:, 0, :7]     # Extract only the pose part, and squeeze the horizon dimension

        # Add object_index at the end
        object_indices = torch.full((placement_poses.shape[0], 1), float(object_index), device=placement_poses.device, dtype=placement_poses.dtype)
        placement_poses = torch.cat([placement_poses, object_indices], dim=-1)

        # (N, 8) tensor, with (x, y, z, qw, qx, qy, qz, object_index)
        return placement_poses
    
    # ------------------------ PUSH SAMPLING ------------------------ #

    def sample_normals_from_actor(self, object_actor, env_idx: int = 0, n: int = 10000):
        
        object_mesh = object_actor.get_collision_meshes()[env_idx]
        
        # Compute object center for filtering interior-facing normals
        object_center = object_mesh.vertices.mean(axis=0)
        
        # Maximum vertical component for 30 degrees from horizontal: sin(30°) ≈ 0.5
        max_vertical_component = np.sin(np.deg2rad(30))
        
        # Sample points and filter - keep sampling until we have enough valid points
        valid_points = []
        valid_normals = []
        max_attempts = 10
        samples_per_attempt = 10000
        
        for attempt in range(max_attempts):
            # Sample points and face indices from mesh surface
            points, face_indices = trimesh.sample.sample_surface(object_mesh, samples_per_attempt)
            
            # Get normals for sampled points (from face normals)
            normals = object_mesh.face_normals[face_indices]
            
            # Normalize normals
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            
            # Filter 1: Normals within 30 degrees of horizontal (|normal_z| < sin(30°))
            horizontal_mask = np.abs(normals[:, 2]) < max_vertical_component
                        
            # Filter 2: Normals pointing outward (away from object center)
            # Vector from object center to surface point
            to_surface = points - object_center
            to_surface = to_surface / np.linalg.norm(to_surface, axis=1, keepdims=True)
            # Normal should point in roughly the same direction as to_surface (dot product > 0)
            outward_mask = np.sum(normals * to_surface, axis=1) > 0
            
            # Combine filters
            valid_mask = horizontal_mask & outward_mask
            
            if np.any(valid_mask):
                valid_points.append(points[valid_mask])
                valid_normals.append(normals[valid_mask])
            
            # Check if we have enough points
            total_valid = sum(len(p) for p in valid_points)
            if total_valid >= n:
                break
        
        # Concatenate all valid points
        if len(valid_points) == 0:
            print(f"Warning: No valid push points found")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        
        points = np.concatenate(valid_points, axis=0)
        normals = np.concatenate(valid_normals, axis=0)
        
        return points, normals
    
    def sample_push_actions(
        self,
        object_name: str,
        n: int = 1,
        min_push_distance: float = 0.08,
        max_push_distance: float = 0.15,
        max_offset_angle: float = np.pi / 12,  # 15 degrees max offset
        env_idx: int = 0
        ):
        """
        Sample push actions for an object.
        
        Args:
            object_name: Name of the object to push.
            n: Number of push samples to generate.
            standoff_distance: Distance to stand off from object surface before pushing.
            min_push_distance: Minimum push distance.
            max_push_distance: Maximum push distance.
            bottom_filter_threshold: Filter points within this distance from bottom.
            max_offset_angle: Maximum angular offset from surface normal (radians).
            env_idx: Environment index to get mesh from.
        
        Returns:
            push_starts: Push start poses in sapien frame. Shape: (n_valid, 7)
            push_directions: Push direction axis indices (0=X, 1=Y, 2=Z). Shape: (n_valid,)
            push_distances: Push distances in meters. Shape: (n_valid,)
        
        Note:
            - Only samples normals within 30 degrees of horizontal plane.
            - Filters out interior-facing normals (pointing toward object center).
            - Filters out points at the bottom of the object.
        """
        # Get object mesh (in world frame for current pose)
        object_actor = self.env.objects[object_name]
        # Sample points and normals from the object actor
        points, normals = self.sample_normals_from_actor(object_actor, env_idx=env_idx, n=n)
        
        if len(points) < n:
            print(f"Warning: Only {len(points)} valid push points found, requested {n}")
            # Re-writing n to the actual number of points sampled.
            n = len(points)

        if n == 0:
            # Return empty tensors
            return (
                torch.zeros((0, 7), device="cuda:0", dtype=torch.float32),
                torch.zeros((0,), device="cuda:0", dtype=torch.long),
                torch.zeros((0,), device="cuda:0", dtype=torch.float32)
            )
        
        # Randomly sample n points from valid points
        # Sample n/2 points, then create both Y and Z push for each point to get n total
        n_points = n // 2
        indices = np.random.choice(len(points), size=n_points, replace=False)
        sampled_points = points[indices]
        sampled_normals = normals[indices]
        
        # Compute push start positions: move outward along surface normal
        push_start_positions_half = sampled_points #+ standoff_distance * sampled_normals
        
        # The base push direction is INTO the object = -normal (in world frame)
        base_push_dir = -sampled_normals  # (n_points, 3)
        
        # Apply random angular offset to push direction
        push_world_dir_half = np.zeros_like(base_push_dir)
        for i in range(n_points):
            # Sample random offset angle (0 to max_offset_angle)
            offset_angle = np.random.uniform(0, max_offset_angle)
            
            # Sample random rotation axis perpendicular to base push direction
            # Find any perpendicular vector
            if np.abs(base_push_dir[i, 0]) < 0.9:
                perp = np.cross(base_push_dir[i], np.array([1, 0, 0]))
            else:
                perp = np.cross(base_push_dir[i], np.array([0, 1, 0]))
            perp = perp / np.linalg.norm(perp)
            
            # Random angle around the push direction to determine offset direction
            azimuth = np.random.uniform(0, 2 * np.pi)
            rotation_axis = R.from_rotvec(azimuth * base_push_dir[i]).apply(perp)
            
            # Apply the offset rotation
            offset_rotation = R.from_rotvec(offset_angle * rotation_axis)
            push_world_dir_half[i] = offset_rotation.apply(base_push_dir[i])
        
        # Re-filter to ensure offset directions are still within 30 degrees of horizontal
        max_vertical_component = np.sin(np.deg2rad(30))
        valid_offset_mask = np.abs(push_world_dir_half[:, 2]) < max_vertical_component
        if not np.all(valid_offset_mask):
            # For invalid ones, fall back to base direction
            push_world_dir_half[~valid_offset_mask] = base_push_dir[~valid_offset_mask]
        
        # Normalize push directions
        push_world_dir_half = push_world_dir_half / np.linalg.norm(push_world_dir_half, axis=1, keepdims=True)
        
        # Duplicate: each point gets both a Y push and a Z push
        # First n_points are Y pushes, next n_points are Z pushes
        push_start_positions = np.concatenate([push_start_positions_half, push_start_positions_half], axis=0)
        push_world_dir = np.concatenate([push_world_dir_half, push_world_dir_half], axis=0)
        push_axes = np.concatenate([np.ones(n_points, dtype=np.int64), 2 * np.ones(n_points, dtype=np.int64)])
        push_signs = np.ones(n_points * 2, dtype=np.int64)  # Will be updated for Y-axis pushes
        
        # Update n to actual count
        n = n_points * 2
        
        # Build gripper orientations based on push axis
        # The chosen axis should align with push_world_dir
        quaternions = np.zeros((n, 4))
        world_up = np.array([0, 0, 1])
        world_x = np.array([1, 0, 0])
        world_y = np.array([0, 1, 0])
        
        for i in range(n):
            push_dir = push_world_dir[i]
            axis = push_axes[i]
            
            if axis == 1:  # Push along gripper Y (+Y or -Y)
                # If push_dir is closer to world +Y, push along gripper -Y (gripper_y = -push_dir)
                # If push_dir is closer to world -Y, push along gripper +Y (gripper_y = push_dir)
                if np.dot(push_dir, world_y) > 0:
                    gripper_y = -push_dir  # Push along gripper -Y
                    push_signs[i] = -1
                else:
                    gripper_y = push_dir   # Push along gripper +Y
                    push_signs[i] = 1
                # Choose gripper Z perpendicular, prefer world up, 
                # NOTE: This makes sure that the z axis is always pointing forward w.r.t world frame
                gripper_z = np.cross(world_up, gripper_y)
                if np.linalg.norm(gripper_z) < 0.05:
                    # NOTE: This makes sure that if the push is too vertical, then the z-axis points dowan w.r.t world
                    gripper_z = np.cross(world_x, gripper_y)
                # Both the above gripper_z assignments makes sure that the manipulator is in a realistic position
                gripper_z = gripper_z / np.linalg.norm(gripper_z)
                gripper_x = np.cross(gripper_y, gripper_z)
                gripper_x = gripper_x / np.linalg.norm(gripper_x)
                
            else:  # axis == 2, Push along gripper Z (+Z)
                gripper_z = push_dir
                # Choose gripper Y perpendicular based on push direction relative to world Y
                if np.dot(push_dir, world_y) > 0:
                    # Push closer to +Y: use gripper_z cross world_up
                    gripper_y = np.cross(gripper_z, world_up)
                else:
                    # Push closer to -Y: use world_up cross gripper_z
                    gripper_y = np.cross(world_up, gripper_z)
                if np.linalg.norm(gripper_y) < 0.1:
                    gripper_y = np.cross(world_x, gripper_z)
                gripper_y = gripper_y / np.linalg.norm(gripper_y)
                # Compute gripper_x to complete right-handed coordinate system: x = y × z
                gripper_x = np.cross(gripper_y, gripper_z)
                gripper_x = gripper_x / np.linalg.norm(gripper_x)
            
            # Build rotation matrix
            rot_matrix = np.column_stack([gripper_x, gripper_y, gripper_z])
            r = R.from_matrix(rot_matrix)
            q = r.as_quat()  # xyzw
            quaternions[i] = [q[3], q[0], q[1], q[2]]  # convert to wxyz
        
        # Build push_start poses (n, 7): [x, y, z, qw, qx, qy, qz]
        push_starts_np = np.concatenate([push_start_positions, quaternions], axis=1)
        
        # Move back along the push axis to account for gripper tip offset
        # Apply offset along the axis we're pushing with
        for i in range(n):
            axis = push_axes[i]
            if axis == 1:   # Y pushes
                # TODO: Hardcoding these parameters
                push_starts_np[i:i+1] = move_pose_along_local_z(push_starts_np[i:i+1], -0.03, format='wxyz')
                push_starts_np[i:i+1] = move_pose_along_local_y(push_starts_np[i:i+1], -push_signs[i] * 0.13, format='wxyz')
            else:  # axis == 2, Z pushes
                push_starts_np[i:i+1] = move_pose_along_local_z(push_starts_np[i:i+1], -0.14, format='wxyz')
        
        # Convert to torch tensors
        push_starts = torch.tensor(push_starts_np, device="cuda:0", dtype=torch.float32)
        # push_directions is (n,) with axis_index
        push_directions = torch.tensor(
            push_axes, 
            device="cuda:0", dtype=torch.long
        )
        
        # Sample push distances and multiply by push_signs
        push_distances = torch.tensor(
            np.random.uniform(min_push_distance, max_push_distance, size=n) * push_signs,
            device="cuda:0", dtype=torch.float32
        )

        # For debug visualization:
        push_ends = push_starts.clone()
        for i in range(n):
            push_start_np = push_starts[i].cpu().numpy()
            push_distance = push_distances[i].item()
            if push_axes[i] == 1:  # Y-axis push
                push_end_np = move_pose_along_local_y(push_start_np.reshape(1, -1), push_distance, format='wxyz')
            else:  # axis == 2, Z-axis push
                push_end_np = move_pose_along_local_z(push_start_np.reshape(1, -1), push_distance, format='wxyz')
            push_ends[i] = to_torch_pose(push_end_np.squeeze())
        
        # Filter by IK feasibility
        curobo_push_starts = push_starts.clone()
        curobo_push_starts[..., 0] = curobo_push_starts[..., 0] + 0.615
        
        push_joints, ik_successes = self.env.action_primitives.inverse_kinematics(curobo_push_starts)
        valid_indices = torch.where(ik_successes[:, 0])[0]
        
        push_starts = push_starts[valid_indices]
        push_directions = push_directions[valid_indices]
        push_distances = push_distances[valid_indices]
        
        return push_starts, push_directions, push_distances

    
    #################### UNUSED FUNCTIONS ##################################