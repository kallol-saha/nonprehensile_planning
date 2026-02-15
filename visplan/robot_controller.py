import mplib
import numpy as np
import sapien
import torch
import trimesh

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose

from visplan.utils import build_panda_gripper_grasp_pose_visual

import sapien.physx as physx
OPEN = 1
CLOSED = -1


class RobotController:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose)

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.gripper_state_batch = np.full((self.env.num_envs, 1), self.gripper_state)
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_panda_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            # self.set_grasp_pose_visual(self.base_env.agent.tcp.pose)
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

    
    def set_grasp_pose_visual(self, pose: sapien.Pose):
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
            current_action = self.env.get_robot_qpos()
            _ = self.env.control_robot(current_action)
            # self.env.capture_video_frame()

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def follow_path_from_position(self, position_path, refine_steps: int = 0):
        """Follow path from position for the first environment (unbatched)."""
        n_step = position_path.shape[0]
        for i in range(n_step + refine_steps):
            qpos = position_path[min(i, n_step - 1)]
            action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def follow_path_from_position_parallel(self, position_path, refine_steps: int = 0):
        """
        Follow path from position for all environments (batched).
        
        Args:
            position_path: Path to follow - shape (num_envs, n_step, 7)
            refine_steps: Additional steps to execute after the path is complete
        
        Returns:
            obs, reward, terminated, truncated, info from the last step
        """
        n_step = position_path.shape[1]
        
        for i in range(n_step + refine_steps):
            step_idx = min(i, n_step - 1)
            qpos = position_path[:, step_idx, :]  # Shape: (num_envs, 7)
            
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state_batch])  # Shape: (num_envs, 8)
            else:
                zeros = np.zeros_like(qpos)  # Shape: (num_envs, 7)
                action = np.hstack([qpos, zeros, self.gripper_state_batch])  # Shape: (num_envs, 15)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def execute_trajectory(self, trajectories, gripper_steps: int = 6, refine_steps: int = 0):
        """
        Follow trajectory with per-timestep gripper states for all environments.
        When any environment has a gripper state change, holds the robot's current
        position while actuating the gripper.
        
        Args:
            trajectories: (batch_size, T, 8) numpy array where:
                - First 7 columns: joint positions
                - Last column: gripper state (OPEN=1, CLOSED=-1)
                - batch_size can be <= num_envs
            gripper_steps: Number of steps to run when gripper state changes
            refine_steps: Additional steps to hold final position
        
        Returns:
            obs, reward, terminated, truncated, info from the last step
        """
        assert self.control_mode == "pd_joint_pos", "Only pd_joint_pos control mode supported"
        
        batch_size, n_step, _ = trajectories.shape
        num_envs = self.env.num_envs
        
        # Assert that batch_size is not greater than num_envs
        assert batch_size <= num_envs, f"trajectories batch_size ({batch_size}) cannot be greater than num_envs ({num_envs})"
        
        joint_positions = trajectories[:, :, :7]  # (batch_size, T, 7)
        gripper_states = trajectories[:, :, 7]    # (batch_size, T)
        
        # Track current gripper state for all environments
        current_gripper = self.gripper_state_batch.flatten().copy()  # (num_envs,)
        
        # Get current robot state for all environments (for environments not in batch)
        current_qpos = self.robot.get_qpos()[:, :-2].cpu().numpy()  # (num_envs, 7)
        
        for i in range(n_step + refine_steps):
            step_idx = min(i, n_step - 1)
            target_qpos = joint_positions[:, step_idx, :]  # (batch_size, 7)
            new_gripper = gripper_states[:, step_idx]  # (batch_size,)
            
            # Prepare action for all environments
            # For environments in batch: use trajectory
            # For environments not in batch: use current state (no movement)
            action_qpos = current_qpos.copy()  # (num_envs, 7)
            action_qpos[:batch_size] = target_qpos  # Update first batch_size environments
            
            action_gripper = current_gripper.copy()  # (num_envs,)
            action_gripper[:batch_size] = new_gripper  # Update first batch_size environments
            
            # Execute trajectory step
            action = np.hstack([action_qpos, action_gripper[:, np.newaxis]])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            if self.vis:
                self.base_env.render_human()
            
            # Check if any environment in batch has a gripper state change
            batch_gripper_changed = (new_gripper != current_gripper[:batch_size]).any()
            if batch_gripper_changed:
                current_gripper[:batch_size] = new_gripper.copy()
                # Get robot's CURRENT position and hold it while actuating gripper
                hold_qpos = self.robot.get_qpos()[:, :-2].cpu().numpy()  # (num_envs, 7)
                for _ in range(gripper_steps):
                    action = np.hstack([hold_qpos, current_gripper[:, np.newaxis]])
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    self.elapsed_steps += 1
                    if self.print_env_info:
                        print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
                    if self.vis:
                        self.base_env.render_human()
        
        # Update internal gripper state
        self.gripper_state_batch = current_gripper[:, np.newaxis].copy()
        
        return obs, reward, terminated, truncated, info

    def follow_path_from_position_until_gripper_collision(self, position_path, refine_steps: int = 0):
        """Follow path from position until gripper collision for the first environment (unbatched)."""
        n_step = position_path.shape[0]
        for i in range(n_step + refine_steps):
            qpos = position_path[min(i, n_step - 1)]
            action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            gripper_collision = self.env.check_gripper_collision()
            if gripper_collision:
                return_path_until = min(i, n_step - 1) - 1
                return position_path[:return_path_until]
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return position_path

    def follow_path_from_position_until_gripper_collision_parallel(self, position_path, refine_steps: int = 0):
        """
        Follow path from position until gripper collision for all environments (batched).
        
        For each environment, advances through the path until collision is detected. Once an
        environment collides, it keeps repeating the same action until all environments have
        collided or all steps are executed.
        
        Args:
            position_path: Path to follow - shape (num_envs, n_step, 7)
            refine_steps: Additional steps to execute after the path is complete
        
        Returns:
            stop_indices: Array of shape (num_envs,) containing the index where each
                         environment stopped in the path
        """
        num_envs = position_path.shape[0]
        n_step = position_path.shape[1]
        
        # Track which environments have collided (torch tensor)
        has_collided = torch.zeros(num_envs, dtype=torch.bool, device=self.env.device)
        # Track current step index for each environment (torch tensor)
        current_step = torch.zeros(num_envs, dtype=torch.long, device=self.env.device)
        # Track where each environment stopped (torch tensor)
        stop_indices = torch.full((num_envs,), n_step + refine_steps - 1, dtype=torch.long, device=self.env.device)
        
        # Initialize with first position
        qpos = position_path[:, 0, :]  # Shape: (num_envs, 7)
        
        for i in range(n_step + refine_steps):
            # For environments that haven't collided, advance to next step
            # For environments that have collided, keep using the same position
            step_idx = torch.clamp(current_step, max=n_step - 1)
            step_idx_np = step_idx.cpu().numpy()
            qpos = position_path[np.arange(num_envs), step_idx_np, :]  # Shape: (num_envs, 7)
            
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state_batch])  # Shape: (num_envs, 8)
            else:
                zeros = np.zeros_like(qpos)  # Shape: (num_envs, 7)
                action = np.hstack([qpos, zeros, self.gripper_state_batch])  # Shape: (num_envs, 15)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            
            # Check collisions for all environments
            gripper_collisions = self.env.check_gripper_collision()  # Shape: (num_envs,) torch bool
            
            # Record stop index for environments that just collided (before updating has_collided)
            newly_collided = gripper_collisions & ~has_collided
            collision_step = min(i, n_step - 1)
            stop_indices[newly_collided] = max(collision_step - 1, 0)  # NOTE: -1 because we want to stop before the collision.
            # Keep current_step at the collision point for newly collided environments
            current_step[newly_collided] = collision_step
            
            # Update collision status (once True, stays True)
            has_collided = has_collided | gripper_collisions
            
            # Advance step index only for environments that haven't collided
            current_step[~has_collided] += 1
            
            # Stop early if all environments have collided
            if has_collided.all():
                break
            
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        
        return stop_indices.cpu().numpy()

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        if self.grasp_pose_visual is not None:
            self.set_grasp_pose_visual(pose)
        pose = sapien.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        """
        This is a reactive collision avoidance approach - it doesn't actively plan around obstacles, 
        but rather stops planning when it hits one. The screw motion planner assumes a straight-line 
        path in configuration space and aborts if that path collides with anything.
        
        This is different from more sophisticated planners (like RRT) that actively explore 
        around obstacles to find collision-free paths.
        """
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.grasp_pose_visual is not None:
            self.set_grasp_pose_visual(pose)
        pose = sapien.Pose(p=pose.p , q=pose.q)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def control_gripper(self, gripper_states, num_steps=6):
        """
        Control gripper for all environments.
        
        Args:
            gripper_states: Gripper state(s) - can be:
                - Scalar: broadcast to all environments
                - 1D array of shape (num_envs,): one state per environment
                - 2D array of shape (num_envs, 1): one state per environment
            num_steps: Number of steps to execute the gripper control
        
        Returns:
            obs, reward, terminated, truncated, info from the last step
        """
        qpos = self.robot.get_qpos()[:, :-2].cpu().numpy()  # Shape: (num_envs, 7)
        num_envs = qpos.shape[0]
        
        # Convert gripper_states to numpy array and expand dimensions if necessary
        gripper_states = np.asarray(gripper_states)
        if gripper_states.ndim == 0:
            gripper_states = np.full((num_envs, 1), gripper_states.item())
        if gripper_states.ndim == 1:
            # 1D array: ensure it matches num_envs
            if gripper_states.shape[0] != num_envs:
                raise ValueError(f"gripper_states shape {gripper_states.shape} does not match num_envs {num_envs}")
            gripper_state_batch = gripper_states[:, np.newaxis]  # Shape: (num_envs, 1)
        elif gripper_states.ndim == 2:
            # 2D array: ensure it matches num_envs
            if gripper_states.shape[0] != num_envs:
                raise ValueError(f"gripper_states shape {gripper_states.shape} does not match num_envs {num_envs}")
            if gripper_states.shape[1] == 1:
                gripper_state_batch = gripper_states  # Shape: (num_envs, 1)
            else:
                raise ValueError(f"gripper_states must have shape (num_envs,) or (num_envs, 1), got {gripper_states.shape}")
        else:
            raise ValueError(f"gripper_states must be scalar, 1D, or 2D array, got {gripper_states.ndim}D")
        
        for i in range(num_steps):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, gripper_state_batch])  # Shape: (num_envs, 8)
            else:
                zeros = np.zeros_like(qpos)  # Shape: (num_envs, 7)
                action = np.hstack([qpos, zeros, gripper_state_batch])  # Shape: (num_envs, 15)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        self.gripper_state_batch = gripper_state_batch.copy()
        return obs, reward, terminated, truncated, info

    def open_gripper(self):
        """Open gripper for the first environment (unbatched)."""
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state = CLOSED):
        """Close gripper for the first environment (unbatched)."""
        self.gripper_state = gripper_state
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        self.use_point_cloud = True
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass
