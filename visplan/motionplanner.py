"""
We do motion planning using Curobo.
"""

import os

# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig, Mesh
from curobo.geom.sphere_fit import SphereFitType
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
    transformation_to_pose, 
    invert_transformation
)
from visplan.utils import to_torch_pose
import copy
from tqdm import tqdm

EE_LINK_CENTER_TO_GRIPPER_TIP = 0.13

# Constants for motion planning
CUROBO_ASSETS_PATH = "visplan/submodules/curobo/src/curobo/content/assets/"       # Have to save here because cuRobo looks for mesh obstacles here
POST_GRASP_LIFT = 0.15
GRASP_DEPTH = 0.25

class MotionPlanner:

    def __init__(
        self, 
        mesh_file_dict: dict = None, 
        initial_object_poses: dict = None,
        ):

        self.mesh_file_dict = mesh_file_dict
        self.initial_object_poses = initial_object_poses

        self.reset_planner()

        self.links = [
            "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_link8",
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
            "attached_object",
        ]
        
    def reset_planner(self):

        print("Building CuRobo World")
        setup_curobo_logger("error")
        tensor_args = TensorDeviceType()
        robot_file = "franka.yml"

        # Create world configuration with individual obstacles
        self.world_config = WorldConfig()

        for mesh_name, mesh_path in self.mesh_file_dict.items():
            obstacle = Mesh(
                file_path=mesh_path,
                name=mesh_name,
                pose=[0.615, 0, 0, 1, 0, 0, 0]
            )
            self.world_config.add_obstacle(obstacle)

        self.back_wall = Cuboid(
            name = "back_wall",
            pose = [-0.4, 0., 0.5, 1, 0, 0, 0],
            dims = [0.2, 1.4, 1.0]
        )
        self.world_config.add_obstacle(self.back_wall)

        self.front_wall = Cuboid(
            name = "front_wall",
            pose = [0.85, 0., 0.5, 1, 0, 0, 0],
            dims = [0.2, 1.4, 1.0]
        )
        self.world_config.add_obstacle(self.front_wall)

        # This is from robot perspective
        self.right_wall = Cuboid(
            name = "right_wall",
            pose = [0.5, -(0.52 + 0.2/2), 0.5, 1, 0, 0, 0],
            dims = [1., 0.2, 1.0]
        )
        self.world_config.add_obstacle(self.right_wall)
        
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            self.world_config,
            tensor_args,
            interpolation_dt=0.01,
            # trajopt_dt=0.15,
            # velocity_scale=0.1,
            use_cuda_graph=False,
            # finetune_dt_scale=2.5,
            interpolation_steps=1000,   # was 10000
        )        

        self.motion_gen = MotionGen(motion_gen_config)
        # Warmup with goalset planning since we use plan_grasp which requires goalset
        # Set n_goalset to the maximum number of grasps you expect to use (Basically the maximum number of poses that can be passed to plan_goalset)
        
        # self.motion_gen.warmup(enable_graph=True, n_goalset=100)
        self.motion_gen.warmup(n_goalset=200)

    def sapien_pose_to_curobo_pose(self, sapien_pose):

        """
        sapien-pose --> (N, 7) or (7,) numpy array.
        """
        sapien_pose = to_torch_pose(sapien_pose)
        curobo_pose = copy.deepcopy(sapien_pose)
        curobo_pose[..., 0] += 0.615       # TODO: Manually adding 0.615 to grasp pose, to transform it to robot base frame, this needs to be automatic
        curobo_pose = move_pose_along_local_z(curobo_pose, -EE_LINK_CENTER_TO_GRIPPER_TIP)
        curobo_pose = to_torch_pose(curobo_pose, device="cuda:0")

        return curobo_pose

    def curobo_pose_to_sapien_pose(self, curobo_pose: torch.Tensor):
        """
        curobo-pose --> (N, 7) or (7,) torch tensor.
        """
        sapien_pose = copy.deepcopy(curobo_pose.squeeze().cpu().numpy())
        sapien_pose[..., 0] -= 0.615       # TODO: Manually adding 0.615 to grasp pose, to transform it to robot base frame, this needs to be automatic
        sapien_pose = move_pose_along_local_z(sapien_pose, EE_LINK_CENTER_TO_GRIPPER_TIP)
        return sapien_pose
       
    def set_collision_world_components(
        self,  
        enable: bool,
        objects: list[str] = [], 
        collision_links: list[str] = []
        ):
        """
        Enable or disable collision checking for world obstacles and robot links.
        
        Args:
            enable: If True, enable collision checking; if False, disable it.
            objects: List of obstacle names to enable/disable.
            collision_links: List of robot link names to enable/disable collision for.
        """
        if len(collision_links) > 0:
            self.motion_gen.toggle_link_collision(collision_links, enable)
            # Enable all other links that are not in the collision_links list
            other_links = [link for link in self.links if link not in collision_links]
            if len(other_links) > 0:
                self.motion_gen.toggle_link_collision(other_links, True)

        for object_name in objects:
            self.motion_gen.world_coll_checker.enable_obstacle(enable=enable, name=object_name)
        
    def clear_gpu_memory(self):
        """
        Clear GPU memory cache. Call this after planning operations to free up GPU memory.
        This helps prevent out-of-memory errors when doing multiple planning operations.
        """
        import torch
        import gc
        
        # Clear PyTorch's CUDA cache
        torch.cuda.empty_cache()
        
        # Optionally reset cuRobo's internal buffers (less aggressive)
        # self.motion_gen.reset(reset_seed=False)
        
        # Force Python garbage collection
        gc.collect()
        
        # Clear cache again after garbage collection
        torch.cuda.empty_cache()

        # Reset cuRobo's internal state (add this)
        self.motion_gen.ik_solver.solver.reset()

    
    def plan_to_joint_state(self, current_joint_state: torch.Tensor, goal_joint_state: torch.Tensor, holding_object: bool = False):
        """
        Plan to a joint state.
        """
        
        start_joint = JointState.from_position(current_joint_state.view(1, -1))
        goal_joint = JointState.from_position(goal_joint_state.view(1, -1))
        
        plan_config = MotionGenPlanConfig(
            max_attempts=100,
        )

        if not holding_object:
            disable_collision_links = ["attached_object"]
            self.motion_gen.toggle_link_collision(disable_collision_links, False)

        result = self.motion_gen.plan_single_js(start_joint, goal_joint, plan_config)
        self.clear_gpu_memory()

        success = result.success.item()
        if success:
            return result.get_interpolated_plan().position, success
        else:
            return None, success
    
    # ==================================================== #

    # ================ PARALLEL PLANNING FUNCTIONS MULTI OBJECT ================ #

    def inverse_kinematics(
        self, 
        input_grasp_poses: torch.Tensor, 
        objects_to_ignore: list[str] = [], 
        disable_collision_links: list[str] = ["attached_object"]
        ):
        """
        Parallel IK solver for multiple grasp poses.
        
        Note: Maximum batch size is 1000. Input batch size should not exceed this limit.
        
        Args:
            input_grasp_poses: Grasp poses to reach ---> (N, 7) torch tensor, (x, y, z, qw, qx, qy, qz)
                where N <= max batch size that fits on the GPU (for RTX 4090, it is 1000)
            objects_to_ignore: List of object names to disable for collision checking during planning
            disable_collision_links: List of collision links to disable for collision checking during planning. 
            By default, it is ["attached_object"]
            # Collision link names:
            # [
            #     "panda_link0",
            #     "panda_link1",
            #     "panda_link2",
            #     "panda_link3",
            #     "panda_link4",
            #     "panda_link5",
            #     "panda_link6",
            #     "panda_link7",
            #     "panda_hand",
            #     "panda_leftfinger",
            #     "panda_rightfinger",
            #     "attached_object",
            # ]

        Returns:
            Valid grasp poses and joint states.
        """

        # Disable collision checking for relevant world components
        self.set_collision_world_components(
            enable=False, 
            objects=objects_to_ignore, 
            collision_links=disable_collision_links
        )
        
        # Predefine the output joint states tensor:
        ik_solutions = torch.zeros((input_grasp_poses.shape[0], 7), device=input_grasp_poses.device, dtype=torch.float32)
        
        # Convert to Pose
        ik_poses = Pose(position=input_grasp_poses[..., :3], quaternion=input_grasp_poses[..., 3:])
        
        # Solve IK for the batch
        ik_result = self.motion_gen.ik_solver.solve_batch(ik_poses)

        # Get successful indices and solutions, then store them in the output tensor:
        success = ik_result.success.to(input_grasp_poses.device)    # (batch_size,) boolean tensor
        valid_indices = torch.where(success)[0]
        if len(valid_indices) > 0:
            ik_solutions[valid_indices] = ik_result.solution[valid_indices, 0]
        else:
            print(f"No valid IK solutions found for the poses")

        # Re-enable collision checking for relevant world components
        self.set_collision_world_components(
            enable=True, 
            objects=objects_to_ignore, 
            collision_links=disable_collision_links
        )

        self.clear_gpu_memory()
            
        return ik_solutions, success
    
    def plan_to_goal_poses(
        self, 
        current_joints: torch.Tensor, 
        goal_poses: torch.Tensor, 
        objects_to_ignore: list[str] = [], 
        disable_collision_links: list[str] = ["attached_object"],
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(max_attempts=100)
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Plan a grasp using CuRobo.

        Note: Maximum batch size is 50. Input batch size should not exceed this limit.

        Args:
            current_joints: Current joint state of the robot ---> (N, 7) torch tensor where N <= 50
            goal_poses: Goal poses of the robot ---> (N, 7) torch tensor where N <= 50
            holding_object: Whether the robot is holding an object
            return_missing: If True, returns full-size trajectories (N, T, 7) with missing indices
                          filled with repeated current_joints. If False, returns only successful trajectories.
            objects_to_ignore: List of object names to disable for collision checking during planning

        Returns:
            trajectories: (N_success, T, 7) if return_missing=False, or (N, T, 7) if return_missing=True
            successful_indices: (N_success,) tensor of successful indices
        """
        
        traj_len = self.motion_gen.trajopt_solver.traj_tsteps
        assert current_joints.shape[0] == goal_poses.shape[0], "Current joint states and goal poses must have the same batch size"
        assert current_joints.shape[1] == 7 and goal_poses.shape[1] == 7, "Current joint states and goal poses must have 7 dimensions"
        assert current_joints.ndim == 2 and goal_poses.ndim == 2, "Current joint states and goal poses must be 2D tensors"
        
        # Disable collision checking for relevant world components
        self.set_collision_world_components(
            enable=False, 
            objects=objects_to_ignore, 
            collision_links=disable_collision_links
        )
        
        # Convert to JointState and Pose
        start_states = JointState.from_position(current_joints)
        curobo_goal_poses = Pose(position=goal_poses[..., :3], quaternion=goal_poses[..., 3:])

        # Initialize the solution, by default as just staying at the current joint state:
        trajectories = current_joints.clone()
        trajectories = trajectories.unsqueeze(1).repeat(1, traj_len, 1)  # (N, traj_len, 7)
        
        # NOTE: For cuRobo motion planning, 
        # use interpolated_plan if you need fine-grained waypoints, or optimized_plan if coarse is sufficient
        # Plan for the batch if environments:
        plan_result = self.motion_gen.plan_batch(
            start_state=start_states, 
            goal_pose=curobo_goal_poses, 
            plan_config=plan_config
        )

        # Get successful indices and solutions, then store them in the output tensor:
        success = plan_result.success.to(current_joints.device)    # (batch_size,) boolean tensor
        successful_indices = torch.where(success)[0]

        if len(successful_indices) == 0:
            return trajectories, success

        trajectories[successful_indices] = plan_result.optimized_plan.position[successful_indices].clone()

        # Re-enable collision checking for relevant world components
        self.set_collision_world_components(
            enable=True, 
            objects=objects_to_ignore, 
            collision_links=disable_collision_links
        )

        self.clear_gpu_memory()

        return trajectories, success

    # ==================================================== #

    def check_pose_collision(self, pose: torch.Tensor):
        """
        Check if a pose is collision free.
        """
        
        pose = Pose(pose[:3], quaternion=pose[3:])
        result = self.motion_gen.ik_solver.solve_single(pose)
        return result.js_solution.position[0], result.success.item()
    
    def attach_objects_to_robot(self, current_joint_state: torch.Tensor, object_names: list):
        """
        Attach objects to the robot.
        """
        joint_state = JointState.from_position(current_joint_state.view(1, -1))
        success = self.motion_gen.attach_objects_to_robot(
            joint_state, 
            object_names,
            sphere_fit_type = SphereFitType.VOXEL_SURFACE,
            surface_sphere_radius=0.015)

        return success
    
    def detach_objects_from_robot(self):
        """
        Detach objects from the robot.
        """
        success = self.motion_gen.detach_object_from_robot()
        return success
    
    def visualize_world_and_robot(self, q: torch.Tensor = None, pose: torch.Tensor = None):

        """
        Args:
            q: Joint configuration tensor. Shape (7,) or (B, 7).
            pose: Pose tensor. Shape (7,) or (B, 7).
            env_idx: Environment index.
        """

        # Optional: visualize an arbitrary pose as a frame in the scene
        T = None
        if pose is not None:
            if isinstance(pose, torch.Tensor):
                pose_np = pose.detach().cpu().numpy()
            else:
                pose_np = np.asarray(pose, dtype=np.float32)
            # pose_np expected as (x, y, z, qw, qx, qy, qz)
            T = pose_to_transformation(pose_np, format='wxyz')
        
        world = self.world_config
        scene = WorldConfig.get_scene_graph(world)

        # Add a small axis triad to visualize the provided pose
        if T is not None:
            frame = axis(origin_size=0.02, axis_length=0.12, transform=T)
            scene.add_geometry(frame)

        # robot spheres from FK
        if q is None:
            q = torch.zeros(7, dtype=torch.float32, device="cuda:0")
        if q.dim() == 1:
            q = q.view(1, -1)
        kin = self.motion_gen.compute_kinematics(JointState.from_position(q))
        spheres = kin.robot_spheres.squeeze(0).cpu().numpy()  # [n,4] x,y,z,r

        for x, y, z, r in spheres:
            s = icosphere(subdivisions=2, radius=float(r))
            s.apply_translation([float(x), float(y), float(z)])
            s.visual.face_colors = [200, 50, 50, 120]
            scene.add_geometry(s)

        scene.show()
    
    def fk(self, q: torch.Tensor, link_name: str = None):
        """
        Forward kinematics using CuRobo via MotionGen's kinematics.

        Args:
            q: Joint configuration tensor. Shape (7,) or (B, 7).
            link_name: Optional link name. If None, returns end-effector pose.

        Returns:
            CudaRobotModelState with fields like ee_pos [B, 3], ee_rot [B, 4] (wxyz),
            link_pos, link_rot, etc.
        """
        if q.dim() == 1:
            q = q.view(1, -1)
        ee_pose = torch.zeros_like(q)
        fk_result = self.motion_gen.ik_solver.kinematics.get_state(q, link_name=link_name)
        ee_pose[:, :3] = fk_result.ee_position
        ee_pose[:, 3:] = fk_result.ee_quaternion
        if q.shape[0] == 1:
            ee_pose = ee_pose.squeeze(0)
        return ee_pose
