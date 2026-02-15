import torch
import numpy as np
import copy
import cv2
import os
import graphviz as gv

from visplan.shelf_packing_env import ShelfPackingMultiObject as Simulator
from visplan.utils import to_torch_pose
from visplan.submodules.robo_utils.robo_utils.conversion_utils import move_pose_along_local_z

POST_GRASP_LIFT = 0.15

class Node:

    def __init__(
        self, 
        id: int,
        state: torch.Tensor, # 1D state vector
        simulator: Simulator,
        parent=None,
        prev_action=None,
        prev_trajectory=None,
        grasped_object_index=None
        ):

        # --------------------------------------------- #
        
        self.id = id
        self.state = state
        self.parent = parent
        self.prev_action = prev_action
        self.simulator = simulator
        self.prev_trajectory = prev_trajectory
        self.grasped_object_index = grasped_object_index

        if self.prev_action is None:
            self.available_actions = ["grasp"]
        elif self.prev_action == "grasp":
            self.available_actions = ["place"]
        elif self.prev_action == "place" or self.prev_action == "push":
            self.available_actions = ["grasp", "push"]

        # --------------------------------------------- #

        self.children: list = []
        self.sampled: bool = False    # Whether this node has sampled children or not.
        self.terminal: bool = False
        self.visit_count: int = 0
        self.total_value = 0.0

class BestFirstSearch:

    def __init__(self, simulator: Simulator):

        self.simulator = simulator
        root_state = self.simulator.get_state().clone()[0]

        self.root_node = Node(
            id = 0,
            state = root_state,
            simulator = simulator,
        )

    def sample_actions_to_expand_node(self, node: Node):

        available_actions = node.available_actions

        # Reset state:
        reset_state = node.state.unsqueeze(0).expand(self.simulator.num_envs, -1)
        self.simulator.set_state(reset_state)
        if self.simulator.render_mode is not None:
            self.simulator.render()

        if "grasp" in available_actions:

            grasp_poses, _, grasp_object_ids = self.simulator.get_current_grasp_poses_for_all_objects()
            num_grasps = min(self.simulator.num_envs, len(grasp_poses))

            grasp_poses = grasp_poses[:num_grasps]
            grasp_object_ids = grasp_object_ids[:num_grasps]

            trajectories, success = self.simulator.action_primitives.Pick(
                grasp_poses=grasp_poses,
                object_indices=grasp_object_ids.tolist(),
                lift_distances=torch.tensor([POST_GRASP_LIFT for _ in range(num_grasps)], device="cuda:0", dtype=torch.float32)
            )

            if len(success_indices) == 0:
                return
            
            current_state = self.simulator.get_state().clone()
            success_indices = np.where(success)[0]

            success_trajectories = trajectories[success_indices]
            success_states = reset_state[torch.tensor(success_indices, device="cuda:0", dtype=torch.long)]

            for i in range(len(success_indices)):
                new_node = Node(
                    id = node.id + 1,
                    state = success_states[i].clone(),
                    simulator = self.simulator,
                    parent = node,
                    prev_action = "grasp",
                    prev_trajectory = success_trajectories[i].copy()
                )
                node.children.append(new_node)

        if "place" in available_actions:

            shelf_poses, shelf_joints = self.simulator.sample_gripper_poses_in_shelf(
                n=100,
                object_name=self.simulator.object_names[object_index],
                depth_tolerance=0.05,
                width_tolerance=0.05,
                height_tolerance=0.05
            )

            trajectories, success = self.simulator.action_primitives.Place(
                place_poses=shelf_poses,
                object_index=object_index
            )


            
    # def grasp()
    
    
    def plan(self):

        pass