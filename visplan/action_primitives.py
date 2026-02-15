"""
We do motion planning using Curobo.
"""

import os

# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import Obstacle
from curobo.wrap.reacher.motion_gen import PoseCostMetric

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import trimesh
from trimesh.creation import icosphere, axis
from visplan.submodules.robo_utils.robo_utils.conversion_utils import (
    pose_to_transformation, 
    move_pose_along_local_z, 
    move_pose_along_local_x,
    move_pose_along_local_y,
    transformation_to_pose, 
    invert_transformation,
    quaternion_to_matrix
)
from visplan.utils import to_torch_pose
import copy
from tqdm import tqdm

EE_LINK_CENTER_TO_GRIPPER_TIP = 0.13

# Constants for motion planning
CUROBO_ASSETS_PATH = "visplan/submodules/curobo/src/curobo/content/assets/"       # Have to save here because cuRobo looks for mesh obstacles here
POST_GRASP_LIFT = 0.15
GRASP_DEPTH = 0.12
OPEN = 1
CLOSED = -1

from visplan.motionplanner import MotionPlanner
# from visplan.shelf_packing_env import ShelfPackingMultiObject

class ActionPrimitives(MotionPlanner):
    """
    Action primitives for robot manipulation using cuRobo motion planning.
    Inherits from MotionPlanner to access all motion planning functionality.
    """
    
    def __init__(
        self,
        env,           # TODO: Make this more general, to be able to take in any environment.
        movement_threshold: float = 0.01  # threshold for detecting unwanted object movement (meters)
    ):
        """
        Initialize ActionPrimitives with motion planning capabilities.
        
        Args:
            env: ShelfPackingMultiObject environment.
            mesh_file_dict: Dictionary mapping object names to mesh file paths.
            initial_object_poses: Dictionary mapping object names to initial poses.
            num_envs: Number of parallel environments to support.
            movement_threshold: Threshold for detecting unwanted object movement (meters).
        """

        self.env = env
        self.num_envs = env.num_envs
        self.movement_threshold = movement_threshold
        
        planner_initial_object_poses = {}
        individual_mesh_files = {}

        # TODO: How does actor.name deal with repeated actors? Need to debug this case
        for actor in self.env.actors:
            mesh = actor.get_collision_meshes()[0]            # Initialize with env 0
            # Store each mesh individually with actor name
            mesh_name = f"{actor.name}"
            mesh_path = os.path.join(CUROBO_ASSETS_PATH, f"{mesh_name}.obj")
            mesh.export(mesh_path)
            individual_mesh_files[mesh_name] = f"{mesh_name}.obj"
            initial_object_pose = self.env.get_object_pose(actor)[0].cpu().detach().numpy()
            planner_initial_object_poses[mesh_name] = initial_object_pose

        # Initialize parent MotionPlanner class
        super().__init__(
            mesh_file_dict=individual_mesh_files,
            initial_object_poses=planner_initial_object_poses,
        )

        # Useful for action primitives:

        self.lift_constraint = PoseCostMetric(
            hold_vec_weight = torch.tensor([1, 1, 1, 1, 1, 0], device="cuda:0"),
            project_to_goal_frame=False
        )
        self.lift_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.lift_constraint,          
        )

        self.along_x_constraint = PoseCostMetric(
            hold_vec_weight = torch.tensor([1, 1, 1, 0, 1, 1], device="cuda:0"),
            project_to_goal_frame=True
        )
        self.along_x_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.along_x_constraint,          
        )
        
        self.along_y_constraint = PoseCostMetric(
            hold_vec_weight = torch.tensor([1, 1, 1, 1, 0, 1], device="cuda:0"),
            project_to_goal_frame=True
        )
        self.along_y_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.along_y_constraint,          
        )

        self.along_z_constraint = PoseCostMetric(
            hold_vec_weight = torch.tensor([1, 1, 1, 1, 1, 0], device="cuda:0"),
            project_to_goal_frame=True
        )
        self.along_z_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.along_z_constraint,          
        )

        self.placement_constraint = PoseCostMetric(
            reach_vec_weight = torch.tensor([1, 0, 0, 1, 1, 1], device="cuda:0"),
            project_to_goal_frame=True
        )
        self.placement_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.placement_constraint,          
        )
    
    
    # =============================== COLLISION WORLD FUNCTIONS =============================== #
    
    def compute_pose_distance(self, pose1: torch.Tensor, pose2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between two poses using element-wise absolute difference.
        
        Args:
            pose1: First pose tensor of shape (..., 7) [x, y, z, qw, qx, qy, qz]
            pose2: Second pose tensor of shape (..., 7) [x, y, z, qw, qx, qy, qz]
            
        Returns:
            Distance tensor of shape (...,) - max element-wise absolute difference
        """
        # Position difference (first 3 elements)
        pos_diff = torch.abs(pose1[..., :3] - pose2[..., :3])
        
        # Quaternion difference (last 4 elements)
        # Handle quaternion sign ambiguity: q and -q represent the same rotation
        q1 = pose1[..., 3:]
        q2 = pose2[..., 3:]
        q_diff1 = torch.abs(q1 - q2)
        q_diff2 = torch.abs(q1 - (-q2))  # Check negative quaternion
        q_diff = torch.minimum(q_diff1, q_diff2)
        
        # Concatenate position and quaternion differences
        diff = torch.cat([pos_diff, q_diff], dim=-1)  # (..., 7)
        
        # Take max across all 7 elements
        pose_distance = diff.max(dim=-1)[0]
        
        return pose_distance
    
    def update_collision_world(self, env_idx: int = 0):
        """
        Update the motion planner with current mesh data from all movable actors.
        """

        if not hasattr(self.env, 'movable_actors'):
            print("movable_actors list not found. Please create it first.")
            return

        for actor in self.env.movable_actors:

            obj_pose = self.env.get_object_pose(actor)[env_idx]          
            curobo_pose = obj_pose.clone().cpu().detach().numpy()

            world_to_object_transform = pose_to_transformation(curobo_pose, format='wxyz')
            initial_to_object_transform = pose_to_transformation(self.initial_object_poses[actor.name], format='wxyz')
            object_transform = world_to_object_transform @ invert_transformation(initial_to_object_transform)

            obj_pose = to_torch_pose(transformation_to_pose(object_transform, format='wxyz'))
            curobo_pose = obj_pose.clone()
            curobo_pose[..., 0] = curobo_pose[..., 0] + 0.615

            pose = Pose(
                position=curobo_pose[..., :3],
                quaternion=curobo_pose[..., 3:]
            )

            # IMPORTANT: Use update_obstacle_pose to update GPU collision tensors (_cube_tensor_list)
            # update_obstacle_pose_in_world_model only updates CPU reference, not collision tensors
            self.motion_gen.world_coll_checker.update_obstacle_pose(
                name=actor.name,
                w_obj_pose=pose,
                update_cpu_reference=True  # Also update CPU reference for consistency
            )
        
    def visualize_planner_world(self, pose=None, env_idx: int = 0):
        qpos = self.env.get_robot_qpos()
        self.update_collision_world(env_idx)
        current_joint_state = qpos[env_idx, :7].to("cuda:0").to(torch.float32)
        # Check if planner is initialized
        # if self.planner is None:
        #     print("Planner not initialized. Call initialize_planner() first.")
        #     return
        self.visualize_world_and_robot(current_joint_state, pose)
    
    def attach_object_to_robot(self, object_name):
        """
        Attach an object to the robot's end effector.
        
        Args:
            object_name: Name of the object to attach
        """
            
        # Get current robot state
        robot_qpos = self.env.get_robot_qpos().to("cuda:0")

        current_joint_state = robot_qpos[0, :7]
        
        # Attach the object
        success = self.attach_objects_to_robot(
            current_joint_state=current_joint_state,
            object_names=[object_name]
        )        
        self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=object_name)
                
    def detach_object_from_robot(self, object_name):
        """
        Detach an object from the robot's end effector.
        
        Args:
            object_name: Name of the object to detach (if None, detaches all)
        """
        
        success = self.detach_objects_from_robot()
        self.motion_gen.world_coll_checker.enable_obstacle(enable=True, name=object_name)
    
    # =============================== ACTION PRIMITIVES =============================== #

    def OpenGripper(
        self
        ):
        # Get current joint state (num_envs, 7)
        current_joint_state = self.env.get_joint_state().to("cuda:0")
        
        # Create single-step trajectory (num_envs, 1, 7)
        trajectories = current_joint_state.unsqueeze(1).cpu().detach().numpy()
        
        # Add OPEN gripper state (num_envs, 1, 1)
        gripper_state = np.full((self.num_envs, 1, 1), OPEN)
        
        # Concatenate to get (num_envs, 1, 8)
        trajectories = np.concatenate([trajectories, gripper_state], axis=-1)
        
        # Execute trajectory
        self.env.robot_controller.execute_trajectory(trajectories)
        
        # All should succeed since it's just gripper control
        success = np.ones(self.num_envs, dtype=bool)
        
        return trajectories, success

    def CloseGripper(
        self
        ):
        # Get current joint state (num_envs, 7)
        current_joint_state = self.env.get_joint_state().to("cuda:0")
        
        # Create single-step trajectory (num_envs, 1, 7)
        trajectories = current_joint_state.unsqueeze(1).cpu().detach().numpy()
        
        # Add CLOSED gripper state (num_envs, 1, 1)
        gripper_state = np.full((self.num_envs, 1, 1), CLOSED)
        
        # Concatenate to get (num_envs, 1, 8)
        trajectories = np.concatenate([trajectories, gripper_state], axis=-1)
        
        # Execute trajectory
        self.env.robot_controller.execute_trajectory(trajectories)
        
        # All should succeed since it's just gripper control
        success = np.ones(self.num_envs, dtype=bool)
        
        return trajectories, success
    
    def Move(
        self,
        goal_poses: torch.Tensor       # in sapien frame, shape: (batch_size, 7)
        ):
        # Get batch size from input
        batch_size = goal_poses.shape[0]
        
        # Assert that batch_size is not greater than num_envs
        assert batch_size <= self.num_envs, f"batch_size ({batch_size}) cannot be greater than num_envs ({self.num_envs})"

        initial_state = self.env.get_state().clone()
        initial_object_poses = self.env.object_poses_tensor.clone()  # (num_envs, num_objects, 7)

        self.update_collision_world()

        curobo_goal_poses = goal_poses.clone()
        curobo_goal_poses[..., 0] = curobo_goal_poses[..., 0] + 0.615

        # Get joint state torch tensor (num_envs, 7) - only use first batch_size
        current_joint_state = self.env.get_joint_state().to("cuda:0")[:batch_size]
        # Get gripper state numpy array (num_envs, 1) - only use first batch_size
        current_gripper_state = self.env.robot_controller.gripper_state_batch[:batch_size].copy()   # This is a numpy array, and should remain a numpy array, because trajectory execution happens with numpy
        
        trajectories, success = self.plan_to_goal_poses(
            current_joints=current_joint_state,
            goal_poses=curobo_goal_poses,
            disable_collision_links=[]
        )

        trajectories = trajectories.cpu().detach().numpy()
        success = success.cpu().detach().numpy()
        current_gripper_state = np.repeat(current_gripper_state[:, np.newaxis, :], trajectories.shape[1], axis=1)

        trajectories = np.concatenate(
            [trajectories, current_gripper_state],
            axis=-1
        )

        # If nothing was successful:
        if np.all(success == False):
            return trajectories, success

        self.env.robot_controller.execute_trajectory(trajectories)

        # Check if any objects have moved beyond threshold (checking full pose: position + rotation)
        current_object_poses = self.env.object_poses_tensor  # (num_envs, num_objects, 7)
        
        # Compute pose distances for all objects (position + rotation)
        pose_distances = self.compute_pose_distance(initial_object_poses, current_object_poses)  # (num_envs, num_objects)
        
        # Check if any object has moved beyond threshold - only check first batch_size
        max_movement = pose_distances[:batch_size].max(dim=1)[0]  # (batch_size,) - max movement per environment
        no_unwanted_movement = max_movement < self.movement_threshold  # (batch_size,)
        no_unwanted_movement = no_unwanted_movement.cpu().numpy()
        
        # Combine planning success with movement check
        success = success & no_unwanted_movement

        # Reset failed environments to initial state (only reset first batch_size)
        current_state = self.env.get_state().clone()
        failures = torch.zeros(self.num_envs, dtype=torch.bool, device="cuda:0")
        failures[:batch_size] = torch.tensor(~success, device="cuda:0", dtype=torch.bool)
        current_state[failures] = initial_state[failures]

        self.env.set_state(current_state)
        if self.env.render_mode is not None:
            self.env.render()

        return trajectories, success

    def Pick(
        self,
        grasp_parameters: torch.Tensor,       # in sapien frame, shape: (batch_size, 9), with (x, y, z, qw, qx, qy, qz, grasp_cost, grasp_object_id)
        ) -> tuple[np.ndarray, np.ndarray]:
        # Get batch size from input
        batch_size = grasp_parameters.shape[0]
        grasp_poses = grasp_parameters[..., :7].clone()
        object_indices = [int(idx.item()) for idx in grasp_parameters[..., -1]]

        # Assert that batch_size is not greater than num_envs
        assert batch_size <= self.num_envs, f"batch_size ({batch_size}) cannot be greater than num_envs ({self.num_envs})"
        assert len(object_indices) == batch_size, f"object_indices length ({len(object_indices)}) must match batch_size ({batch_size})"

        lift_distances=torch.tensor([POST_GRASP_LIFT for _ in range(batch_size)], device="cuda:0", dtype=torch.float32)

        initial_state = self.env.get_state().clone()
        initial_object_poses = self.env.object_poses_tensor.clone()  # (num_envs, num_objects, 7)
        rewards_before_execution = self.env.compute_dense_reward()[:batch_size]
        
        self.update_collision_world()

        curobo_grasp_poses = grasp_poses.clone()
        curobo_grasp_poses[..., 0] = curobo_grasp_poses[..., 0] + 0.615

        curobo_pre_grasp_poses = curobo_grasp_poses.clone()
        curobo_pre_grasp_poses = move_pose_along_local_z(curobo_pre_grasp_poses, -GRASP_DEPTH)
        curobo_pre_grasp_poses = torch.tensor(curobo_pre_grasp_poses, dtype=curobo_grasp_poses.dtype, device=curobo_grasp_poses.device)

        curobo_lift_poses = curobo_grasp_poses.clone()
        curobo_lift_poses[..., 2] = curobo_lift_poses[..., 2] + lift_distances

        # Get joint state torch tensor (num_envs, 7) - only use first batch_size
        current_joint_state = self.env.get_joint_state().to("cuda:0")[:batch_size]

        pre_grasp_trajectories, pre_grasp_success = self.plan_to_goal_poses(
            current_joints=current_joint_state,
            goal_poses=curobo_pre_grasp_poses,
            disable_collision_links=[]
        )
        
        grasp_trajectories, grasp_success = self.plan_to_goal_poses(
            current_joints=pre_grasp_trajectories[:, -1],
            goal_poses=curobo_grasp_poses,
            plan_config=self.along_z_plan_config,
            disable_collision_links=[]
        )

        lift_trajectories, lift_success = self.plan_to_goal_poses(
            current_joints=grasp_trajectories[:, -1],
            goal_poses=curobo_lift_poses,
            plan_config=self.lift_plan_config,
            disable_collision_links=[]
        )

        # Convert to numpy
        pre_grasp_trajectories = pre_grasp_trajectories.cpu().detach().numpy()
        grasp_trajectories = grasp_trajectories.cpu().detach().numpy()
        lift_trajectories = lift_trajectories.cpu().detach().numpy()
        pre_grasp_success = pre_grasp_success.cpu().detach().numpy()
        grasp_success = grasp_success.cpu().detach().numpy()
        lift_success = lift_success.cpu().detach().numpy()
        
        # Combine success: pre_grasp, grasp, and lift must all succeed
        success = pre_grasp_success & grasp_success & lift_success

        # Form the gripper state (using batch_size)
        pre_grasp_gripper = np.full((batch_size, pre_grasp_trajectories.shape[1]), OPEN)
        grasp_gripper = np.full((batch_size, grasp_trajectories.shape[1]), OPEN)
        grasp_gripper[:, -1] = CLOSED  # Last element is where the grasp should happen
        lift_gripper = np.full((batch_size, lift_trajectories.shape[1]), CLOSED)
        
        # Stack trajectories and gripper states
        trajectories = np.concatenate([pre_grasp_trajectories, grasp_trajectories, lift_trajectories], axis=1)  # (batch_size, pre_grasp_len + grasp_len + lift_len, 7)
        gripper_states = np.concatenate([pre_grasp_gripper, grasp_gripper, lift_gripper], axis=1)  # (batch_size, pre_grasp_len + grasp_len + lift_len)
        gripper_states = gripper_states[:, :, np.newaxis]  # (batch_size, total_len, 1)

        # Concatenate trajectories with gripper states
        trajectories = np.concatenate([trajectories, gripper_states], axis=-1)  # (batch_size, total_len, 8)

        # If nothing was successful:
        if np.all(success == False):
            return trajectories, success

        self.env.robot_controller.execute_trajectory(trajectories)

        # Check if objects are grasped for each environment (only check first batch_size)
        # Object is grasped if its z position is at least lift_distance/2 above initial z position
        current_object_poses = self.env.object_poses_tensor  # (num_envs, num_objects, 7)
        
        is_grasped = np.zeros(batch_size, dtype=bool)
        for env_idx in range(batch_size):
            object_index = object_indices[env_idx]
            initial_z = initial_object_poses[env_idx, object_index, 2].item()  # Initial z position
            current_z = current_object_poses[env_idx, object_index, 2].item()  # Current z position
            lift_threshold = initial_z + (lift_distances[env_idx].item() / 2)
            is_grasped[env_idx] = current_z >= lift_threshold

        success = success & is_grasped #& no_unwanted_movement

        # Only reset failed environments in the batch
        current_state = self.env.get_state().clone()[:batch_size]
        rewards_after_execution = self.env.compute_dense_reward()[:batch_size]

        rewards = rewards_after_execution - rewards_before_execution    # Volume of objects transferred to the shelf
        
        if success.sum() == 0:
            # Everything failed, return empty lists
            return [], [], [], []

        successful_states = current_state[success]
        successful_trajectories = trajectories[success]
        successful_actions = grasp_parameters[success]
        rewards = rewards[success]

        return successful_states, successful_trajectories, successful_actions, rewards
        
    def Place(
        self,
        place_parameters: torch.Tensor,       # in sapien frame, shape: (batch_size, 8), with (x, y, z, qw, qx, qy, qz, object_index)
        ):
        # Get batch size from input
        batch_size = place_parameters.shape[0]
        
        # Extract place_poses and object_index from the tensor
        place_poses = place_parameters[..., :7].clone()
        object_indices = [int(round(idx.item())) for idx in place_parameters[..., -1]]
        
        # Assert that batch_size is not greater than num_envs
        assert batch_size <= self.num_envs, f"batch_size ({batch_size}) cannot be greater than num_envs ({self.num_envs})"
        assert len(object_indices) == batch_size, f"object_indices length ({len(object_indices)}) must match batch_size ({batch_size})"
        assert all(idx == object_indices[0] for idx in object_indices), "All placements in a batch must be for the same object"

        initial_state = self.env.get_state().clone()
        initial_object_poses = self.env.object_poses_tensor.clone()  # (num_envs, num_objects, 7)
        rewards_before_execution = self.env.compute_dense_reward()[:batch_size]
        
        object_index = object_indices[0]  # All should be the same
        object_name = self.env.object_names[object_index]
        
        self.update_collision_world()
        self.attach_object_to_robot(object_name)

        # Find objects already in shelf and disable them during planning
        _, object_indices_in_shelf = self.env.number_of_objects_in_shelf(env_idx=0)
        objects_in_shelf = [self.env.object_names[i] for i in object_indices_in_shelf]

        curobo_place_poses = place_poses.clone()
        curobo_place_poses[..., 0] = curobo_place_poses[..., 0] + 0.615

        # Get joint state torch tensor (num_envs, 7) - only use first batch_size
        current_joint_state = self.env.get_joint_state().to("cuda:0")[:batch_size]
        
        print(f"Objects in shelf: {objects_in_shelf}")
        place_trajectories, place_success = self.plan_to_goal_poses(
            current_joints=current_joint_state,
            goal_poses=curobo_place_poses,
            plan_config=self.placement_plan_config,
            objects_to_ignore=objects_in_shelf,
            disable_collision_links=[]
        )
        self.detach_object_from_robot(object_name)

        place_trajectories = place_trajectories.cpu().detach().numpy()
        place_success = place_success.cpu().detach().numpy()

        # Form the gripper state (all CLOSED for forward place) - using batch_size
        place_gripper = np.full((batch_size, place_trajectories.shape[1]), CLOSED)
        place_gripper = place_gripper[:, :, np.newaxis]  # (batch_size, traj_len, 1)

        # Concatenate forward trajectories with gripper states
        place_trajectories_forward = np.concatenate([place_trajectories, place_gripper], axis=-1)  # (batch_size, traj_len, 8)

        # Reverse placement trajectories to go back (with OPEN gripper)
        place_trajectories_reversed = np.flip(place_trajectories, axis=1)  # Reverse along time axis
        place_gripper_open = np.full((batch_size, place_trajectories_reversed.shape[1]), OPEN)
        place_gripper_open = place_gripper_open[:, :, np.newaxis]  # (batch_size, traj_len, 1)
        place_trajectories_reversed = np.concatenate([place_trajectories_reversed, place_gripper_open], axis=-1)  # (batch_size, traj_len, 8)

        # Stack forward and reversed trajectories
        place_trajectories = np.concatenate([place_trajectories_forward, place_trajectories_reversed], axis=1)

        # If nothing was successful:
        if np.all(place_success == False):
            return [], [], [], []

        self.env.robot_controller.execute_trajectory(place_trajectories)
        
        is_placed = np.zeros(batch_size, dtype=bool)
        for env_idx in range(batch_size):
            is_placed[env_idx] = self.env.check_object_inside_shelf(object_name, env_idx=env_idx)

        place_success = place_success & is_placed

        current_state = self.env.get_state().clone()[:batch_size]
        rewards_after_execution = self.env.compute_dense_reward()[:batch_size]
        rewards = rewards_after_execution - rewards_before_execution    # Volume of objects transferred to the shelf

        if place_success.sum() == 0:
            # Everything failed, return empty lists
            return [], [], [], []

        successful_states = current_state[place_success]
        successful_trajectories = place_trajectories[place_success]
        successful_actions = place_parameters[place_success]
        rewards = rewards[place_success]

        return successful_states, successful_trajectories, successful_actions, rewards

    def Push(
        self,
        push_start: torch.Tensor,
        push_direction: torch.Tensor,
        push_distance: torch.Tensor,
        ):
        """
        Push an object from a start pose in a specified direction.
        
        Args:
            push_start: Push start poses in sapien frame. Shape: (batch_size, 7)
            push_direction: Push direction specification. Shape: (batch_size, 2)
                Each row is [axis_index, sign] where:
                - axis_index ∈ {0:X, 1:Y, 2:Z}
            push_distance: Push distance in meters. Shape: (batch_size,), this is negative for negative direction of the axis
            object_name: The object to be pushed. Will be disabled from collision world.
        
        Returns:
            trajectories: Executed trajectory. Shape: (batch_size, T, 8)
            success: Success flags for each environment. Shape: (batch_size,)
        """
        # Get batch size from input
        batch_size = push_start.shape[0]
        
        # Assertions for input sizes
        assert batch_size <= self.num_envs, f"batch_size ({batch_size}) cannot be greater than num_envs ({self.num_envs})"
        assert push_start.shape == (batch_size, 7), f"push_start must be (batch_size={batch_size}, 7), got {push_start.shape}"
        assert push_direction.shape == (batch_size,), f"push_direction must be (batch_size={batch_size},), got {push_direction.shape}"
        assert push_distance.shape == (batch_size,), f"push_distance must be (batch_size={batch_size},), got {push_distance.shape}"
    
        initial_state = self.env.get_state().clone()
        
        self.update_collision_world()

        # Compute push end poses in parallel
        # Extract position and quaternion from push_start
        push_start_np = push_start.cpu().detach().numpy()
        pos = push_start_np[:, :3]  # (batch_size, 3)
        quat = push_start_np[:, 3:7]  # (batch_size, 4) in wxyz format
        
        # Get rotation matrices: (batch_size, 3, 3)
        R = quaternion_to_matrix(quat, format='wxyz')
        
        # Gather the appropriate axis for each environment based on push_direction
        # push_direction: (batch_size,) with values 0, 1, or 2 for X, Y, Z
        axis_indices = push_direction.cpu().numpy().astype(int)  # (batch_size,)
        push_distance_np = push_distance.cpu().numpy()  # (batch_size,)
        
        # For each env, select the appropriate column of R
        # R[:, :, axis_indices[i]] gives the local axis direction
        env_indices = np.arange(batch_size)
        local_axes = R[env_indices, :, axis_indices]  # (batch_size, 3)
        
        # Compute new positions: pos + push_distance * local_axis
        pos_new = pos + (push_distance_np[:, np.newaxis] * local_axes)
        
        # Construct push_end poses (orientation stays the same)
        push_end_np = push_start_np.copy()
        push_end_np[:, :3] = pos_new
        push_end = torch.tensor(push_end_np, dtype=push_start.dtype, device=push_start.device)

        # Convert to cuRobo frame
        curobo_push_start = push_start.clone()
        curobo_push_start[..., 0] = curobo_push_start[..., 0] + 0.615        
        curobo_push_end = push_end.clone()
        curobo_push_end[..., 0] = curobo_push_end[..., 0] + 0.615

        # Get joint state torch tensor (num_envs, 7) - only use first batch_size
        current_joint_state = self.env.get_joint_state().to("cuda:0")[:batch_size]
        
        # ========== Step 1: Plan to push_start without disabling anything ==========
        approach_trajectories, approach_success = self.plan_to_goal_poses(
            current_joints=current_joint_state,
            goal_poses=curobo_push_start,
            disable_collision_links=[]
        )
        
        # ========== Step 2: Plan push movement for each axis separately ==========
        # Start joints for push are the end of approach trajectories
        push_start_joints = approach_trajectories[:, -1, :]  # (batch_size, 7)
        
        # Initialize push trajectories and success
        traj_len = self.motion_gen.trajopt_solver.traj_tsteps
        push_trajectories = push_start_joints.unsqueeze(1).repeat(1, traj_len, 1)  # Default: stay at start
        push_success = torch.zeros(batch_size, dtype=torch.bool, device="cuda:0")
        
        # Map axis index to plan config
        axis_plan_configs = {
            0: self.along_x_plan_config,
            1: self.along_y_plan_config,
            2: self.along_z_plan_config,
        }
        
        # Collision links to disable for push movement (panda hand to attached object)
        push_disable_links = [
            "panda_link6", "panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger", "attached_object"
            ]
        
        # Group environments by axis and plan for each axis that has environments
        for axis_idx in [0, 1, 2]:
            # Find environments using this axis
            axis_mask = (axis_indices == axis_idx)
            if not np.any(axis_mask):
                continue  # Skip if no environments use this axis
            
            env_indices_for_axis = torch.tensor(np.where(axis_mask)[0], device="cuda:0", dtype=torch.long)
            
            # Get the start joints and goal poses for these environments
            axis_start_joints = push_start_joints[env_indices_for_axis]
            axis_goal_poses = curobo_push_end[env_indices_for_axis]
            
            # Plan for this axis
            axis_trajectories, axis_success = self.plan_to_goal_poses(
                current_joints=axis_start_joints,
                goal_poses=axis_goal_poses,
                disable_collision_links=push_disable_links,
                plan_config=axis_plan_configs[axis_idx]
            )
            
            # Store results back into full arrays
            push_trajectories[env_indices_for_axis] = axis_trajectories
            push_success[env_indices_for_axis] = axis_success
        
        # Combine approach and push success
        success = approach_success & push_success
        
        # ========== Combine trajectories and add gripper states ==========
        approach_trajectories = approach_trajectories.cpu().detach().numpy()
        push_trajectories = push_trajectories.cpu().detach().numpy()
        success = success.cpu().detach().numpy()
        
        # Gripper stays closed throughout - using batch_size
        approach_gripper = np.full((batch_size, approach_trajectories.shape[1], 1), CLOSED)
        push_gripper = np.full((batch_size, push_trajectories.shape[1], 1), CLOSED)
        
        # Concatenate gripper states
        approach_trajectories = np.concatenate([approach_trajectories, approach_gripper], axis=-1)
        push_trajectories = np.concatenate([push_trajectories, push_gripper], axis=-1)
        
        # Reverse push trajectories to go back
        push_trajectories_reversed = np.flip(push_trajectories, axis=1)  # Reverse along time axis
        
        # Stack approach, push (forward), and push (backward) trajectories
        trajectories = np.concatenate([approach_trajectories, push_trajectories, push_trajectories_reversed], axis=1)
        
        # If nothing was successful, return early
        if np.all(success == False):
            return trajectories, success
        
        # Execute trajectory
        self.env.robot_controller.execute_trajectory(trajectories)
        
        # Reset failed environments to initial state (only reset first batch_size)
        current_state = self.env.get_state().clone()
        failures = torch.zeros(self.num_envs, dtype=torch.bool, device="cuda:0")
        failures[:batch_size] = torch.tensor(~success, device="cuda:0", dtype=torch.bool)
        current_state[failures] = initial_state[failures]
        
        self.env.set_state(current_state)
        if self.env.render_mode is not None:
            self.env.render()
        
        return trajectories, success