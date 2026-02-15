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
ACTION_DIM = 9

class Node:

    def __init__(
        self, 
        id: int,
        simulator: Simulator,
        state: torch.Tensor, 
        parent=None,
        prev_action: torch.Tensor = None,
        prev_trajectory: torch.Tensor = None,
        reward: float = 0.0,
        primitive: str = None,
        C: float = 2.0,
        ):

        # --------------------------------------------- #
        
        self.id = id
        self.state = state
        self.parent = parent
        self.parent_id = parent.id if parent is not None else None
        self.simulator = simulator
        self.batch_size = self.simulator.num_envs
        self.prev_action = prev_action
        self.prev_trajectory = prev_trajectory
        self.reward = reward
        self.Q = reward
        self.C = C

        # --------------------------------------------- #

        self.children: list = []
        self.children_ids = []
        self.visit_count: int = 0
        self.all_grasp_parameters = None
        self.grasps_tried = 0

        # --------------------------------------------- #
        self.actions = []
        self.trajectories = []
        self.children = []
        
        if primitive is None:
            self.primitive = self.assign_primitive()
        else:
            self.primitive = primitive


    def assign_primitive(self):

        if self.object_in_hand is None:
            return self.simulator.action_primitives.Pick
        else:
            return self.simulator.action_primitives.Place
    
    @property
    def object_in_hand(self):

        # Derive object_in_hand from parent's primitive
      if self.parent is None:
        return None
      elif self.parent.primitive == self.simulator.action_primitives.Pick:
        object_index = int(self.prev_action[-1].item())
        object_name = self.simulator.object_names[object_index]
        return object_name
      elif self.parent.primitive == self.simulator.action_primitives.Place:
        return None
    
    def action_sampler(self):

        self.simulator.set_state(self.state.unsqueeze(0).repeat(self.simulator.num_envs, 1))
        if self.simulator.render_mode is not None:
            self.simulator.render()

        if self.primitive == self.simulator.action_primitives.Pick:
            if self.all_grasp_parameters is None:
                self.all_grasp_parameters = self.simulator.action_sampler.sample_grasp_poses()
            if self.grasps_tried >= self.all_grasp_parameters.shape[0]:
                # All grasp poses have been sampled, so there are no more children to sample.
                return []
            grasp_parameters = self.all_grasp_parameters[self.grasps_tried : self.grasps_tried + self.batch_size]
            self.grasps_tried += self.batch_size
            self.grasps_tried = min(self.grasps_tried, self.all_grasp_parameters.shape[0])
            return grasp_parameters.clone()

        elif self.primitive == self.simulator.action_primitives.Place:
            # Get object_index from object_in_hand
            object_name = self.object_in_hand
            object_index = self.simulator.object_names.index(object_name)
            return self.simulator.action_sampler.sample_placement_poses(object_index, batch_size = 16)    
    
    @property
    def UCB(self):

        n_parent = self.parent.visit_count if self.parent is not None else 0
        n_edge = self.visit_count

        ucb = self.Q + self.C * np.sqrt( np.log(n_parent+1) / (n_edge+1) )

        return ucb

    def capture_image(self):
        
        self.simulator.set_state(self.state.unsqueeze(0).repeat(self.simulator.num_envs, 1))
        if self.simulator.render_mode is not None:
            self.simulator.render()
        image = self.simulator.capture_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def save(self, nodes_dir: str):
        """
        Save node to npz file.
        
        Args:
            nodes_dir: Directory to save the node file
        """
        file_path = os.path.join(nodes_dir, f"node_{self.id}.npz")
        
        # Convert primitive to string
        if self.primitive == self.simulator.action_primitives.Pick:
            primitive_str = "Pick"
        elif self.primitive == self.simulator.action_primitives.Place:
            primitive_str = "Place"
        
        # Update children_ids from actual children list to ensure consistency
        self.children_ids = [child.id for child in self.children]
        
        # Update parent_id from parent if it exists
        if self.parent is not None:
            self.parent_id = self.parent.id
        
        # Prepare data dictionary
        data = {
            'id': self.id,
            'state': self.state.cpu().detach().numpy() if isinstance(self.state, torch.Tensor) else self.state,
            'parent_id': self.parent_id if self.parent_id is not None else -1,
            'reward': self.reward,
            'Q': self.Q,
            'C': self.C,
            'visit_count': self.visit_count,
            'grasps_tried': self.grasps_tried,
            'primitive': primitive_str,
            'children_ids': np.array(self.children_ids) if len(self.children_ids) > 0 else np.array([], dtype=np.int64),
        }
        
        # Save prev_action if it exists
        if self.prev_action is not None:
            data['prev_action'] = self.prev_action.cpu().numpy() if isinstance(self.prev_action, torch.Tensor) else self.prev_action
        
        # Save prev_trajectory if it exists
        if self.prev_trajectory is not None:
            data['prev_trajectory'] = self.prev_trajectory.cpu().numpy() if isinstance(self.prev_trajectory, torch.Tensor) else self.prev_trajectory
        
        # Save all_grasp_parameters if it exists
        if self.all_grasp_parameters is not None:
            data['all_grasp_parameters'] = self.all_grasp_parameters.cpu().numpy() if isinstance(self.all_grasp_parameters, torch.Tensor) else self.all_grasp_parameters
        
        np.savez_compressed(file_path, **data)
    
    @classmethod
    def load(cls, file_path: str, simulator: Simulator, parent=None):
        """
        Load node from npz file.
        
        Args:
            file_path: Path to the npz file
            simulator: Simulator instance
            parent: Parent node (if available)
            
        Returns:
            Node: Loaded node instance
        """
        data = np.load(file_path, allow_pickle=True)
        
        # Extract data
        node_id = int(data['id'])
        state = torch.tensor(data['state'], device='cuda:0')
        parent_id = int(data['parent_id']) if data['parent_id'] != -1 else None
        reward = float(data['reward'])
        Q = float(data['Q'])
        C = float(data['C'])
        visit_count = int(data['visit_count'])
        grasps_tried = int(data['grasps_tried'])
        primitive_str = str(data['primitive'])
        children_ids = data['children_ids'].tolist() if len(data['children_ids']) > 0 else []
        
        # Convert primitive string back to function
        if primitive_str == "Pick":
            primitive = simulator.action_primitives.Pick
        elif primitive_str == "Place":
            primitive = simulator.action_primitives.Place
        
        # Load optional fields
        prev_action = None
        if 'prev_action' in data:
            prev_action = torch.tensor(data['prev_action'], device='cuda:0')
        
        prev_trajectory = None
        if 'prev_trajectory' in data:
            prev_trajectory = data['prev_trajectory']
        
        all_grasp_parameters = None
        if 'all_grasp_parameters' in data:
            all_grasp_parameters = torch.tensor(data['all_grasp_parameters'], device='cuda:0')
        
        # Create node
        node = cls(
            id=node_id,
            simulator=simulator,
            state=state,
            parent=parent,
            prev_action=prev_action,
            prev_trajectory=prev_trajectory,
            reward=reward,
            primitive=primitive,
            C=C
        )
        
        # Set additional attributes
        node.Q = Q
        node.visit_count = visit_count
        node.grasps_tried = grasps_tried
        node.all_grasp_parameters = all_grasp_parameters
        node.children_ids = children_ids
        node.parent_id = parent_id
        
        return node


class MCTS:

    def __init__(
        self, 
        simulator: Simulator, 
        C: float = 2.0,
        alpha: float = 0.95,
        k: float = 1.0,
        max_depth: int = 10, 
        nodes_dir: str = "assets/nodes",
        ):
        
        self.simulator = simulator
        self.nodes: list[Node] = []
        self.nodes_dir = nodes_dir
        self.loaded_root_node = None

        self.C = C
        self.alpha = alpha
        self.k = k
        self.max_depth = max_depth
        
        # Create nodes directory if it doesn't exist
        os.makedirs(self.nodes_dir, exist_ok=True)
        
        # Load existing nodes if directory has node files
        self.loaded_root_node = self.load_nodes()

    def widening_condition(self, node):

        return len(node.children) < self.k * (node.visit_count ** self.alpha)
    
    def save_nodes(self):
        """Save all nodes to disk."""
        for node in self.nodes:
            node.save(self.nodes_dir)
    
    def load_nodes(self):
        """Load all nodes from disk if they exist.
        
        Returns:
            root_node: The root node (node with id=0 or parent_id=None), or None if no nodes loaded
        """
        if not os.path.exists(self.nodes_dir):
            return None
        
        # Find all node files
        node_files = [f for f in os.listdir(self.nodes_dir) if f.startswith('node_') and f.endswith('.npz')]
        
        if len(node_files) == 0:
            return None
        
        print(f"Loading {len(node_files)} nodes from {self.nodes_dir}")
        
        # Load all nodes first (without parent relationships)
        node_dict = {}
        for node_file in node_files:
            file_path = os.path.join(self.nodes_dir, node_file)
            node = Node.load(file_path, self.simulator, parent=None)
            node_dict[node.id] = node
            self.nodes.append(node)
        
        # Restore parent-child relationships
        for node in self.nodes:
            # Set parent
            if node.parent_id is not None and node.parent_id in node_dict:
                node.parent = node_dict[node.parent_id]
            
            # Set children
            node.children = [node_dict[child_id] for child_id in node.children_ids if child_id in node_dict]
        
        # Find root node (node with id=0)
        root_node = None
        for node in self.nodes:
            if node.id == 0:
                root_node = node
                break
        
        print(f"Loaded {len(self.nodes)} nodes with restored relationships")
        return root_node

    def mcts_iteration(self, root_node):
        """
        Perform one MCTS iteration: selection and expansion.
        
        Args:
            root_node: Starting node for this iteration
            max_depth: Maximum depth to explore
            
        Returns:
            current_node: Final node reached in this iteration
            depth: Final depth reached
        """

        current_node = root_node
        depth = 0

        while depth < self.max_depth:

            current_node.visit_count += 1

            # --------------- Progressive widening --------------- #

            if self.widening_condition(current_node):

                actions = current_node.action_sampler()
                primitive = current_node.primitive
                
                # If there are no actions to sample, exit widening condition and continue to selection
                # Otherwise, sample actions and create new nodes.
                if len(actions) > 0:   
                    new_states, trajectories, executable_actions, rewards = primitive(actions)
                   
                    if len(executable_actions) == 0:
                        break   # Everything failed, this is terminal

                    for i in range(len(executable_actions)):
                        new_node = Node(
                            id=len(self.nodes),
                            simulator=self.simulator,
                            state=new_states[i].clone(),
                            parent=current_node,
                            prev_action=executable_actions[i].clone(),
                            prev_trajectory=trajectories[i].copy(),
                            reward=rewards[i].item(),
                            C=current_node.C
                        )
                        current_node.children.append(new_node)
                        current_node.children_ids.append(new_node.id)
                        self.nodes.append(new_node)
                        new_node.trajectories.append(trajectories[i].copy())
                        new_node.actions.append(executable_actions[i].clone())

            # --------------- Selection --------------- #

            # Check if there are children to select from
            if len(current_node.children) == 0:
                break  # No children available, terminal node
            
            # Randomly shuffle children indices before selection
            child_indices = list(range(len(current_node.children)))
            np.random.shuffle(child_indices)
            
            # Find the child with maximum reward from shuffled list
            shuffled_rewards = [current_node.children[i].UCB for i in child_indices]
            selected_shuffled_index = np.argmax(shuffled_rewards)
            selected_child_index = child_indices[selected_shuffled_index]
            selected_child = current_node.children[selected_child_index]
            current_node = selected_child

            depth += 1

        current_node.visit_count += 1
        
        return current_node

    def plan(
        self, 
        root_node: Node,
        iterations: int = 10, 
        ):

        # If we have a loaded root node, use it instead of the provided one
        if self.loaded_root_node is not None:
            root_node = self.loaded_root_node
            print(f"Using loaded root node (id={root_node.id}) instead of provided root node")
        elif root_node not in self.nodes:
            self.nodes.append(root_node)

        root_node.C = self.C
        
        for i in range(iterations):
        
            print(f"MCTS Iteration {i+1}")

            current_node = self.mcts_iteration(root_node)

            # --------------- Backpropagation --------------- #

            # Value of leaf node is always zero, we only care about accumulated progress, not the progress to go.
            G = 0
            while current_node is not None:
                
                # Update edge Q-value if this node has a parent (i.e., there's an edge)
                if current_node.parent is not None:

                    # This gives us the return starting from current_node
                    G = current_node.reward + G
                    
                    # Update edge Q-value using incremental mean update
                    current_node.Q = current_node.Q + (G - current_node.Q) / current_node.visit_count
                
                current_node = current_node.parent
            
            # Save all nodes after each iteration
            self.save_nodes()
    
    def _find_all_paths(self, node, current_path=None, all_paths=None):
        """
        Recursively find all paths from node to leaves.
        
        Args:
            node: Current node
            current_path: Current path being built
            all_paths: List to store all complete paths
            
        Returns:
            List of all paths (each path is a list of nodes)
        """
        if all_paths is None:
            all_paths = []
        if current_path is None:
            current_path = []
        
        current_path = current_path + [node]
        
        if len(node.children) == 0:
            # Leaf node - add this path
            all_paths.append(current_path.copy())
        else:
            # Continue to all children
            for child in node.children:
                self._find_all_paths(child, current_path, all_paths)
        
        return all_paths
    
    def _score_path(self, path):
        """
        Score a path. Uses sum of Q values along the path.
        
        Args:
            path: List of nodes in the path
            
        Returns:
            score: Total score for the path
        """
        return sum(node.Q for node in path)
    
    def greedy_rollout(self, root_node=None, top_k=3):
        """
        Find top K best plans and execute them by stacking trajectories.
        Executes trajectories without updating any node statistics.
        
        Args:
            root_node: Starting node for rollout. If None, uses loaded_root_node.
            top_k: Number of top plans to execute (default: 3)
            
        Returns:
            paths: List of top K paths (each path is a list of nodes)
        """
        if root_node is None:
            if self.loaded_root_node is None:
                raise ValueError("No root node available. Provide root_node or load nodes first.")
            root_node = self.loaded_root_node
        
        # Set initial state
        self.simulator.set_state(root_node.state.unsqueeze(0).repeat(self.simulator.num_envs, 1))
        if self.simulator.render_mode is not None:
            self.simulator.render()
        
        # Find all paths from root to leaves
        all_paths = self._find_all_paths(root_node)
        
        if len(all_paths) == 0:
            print("No paths found in tree")
            return []
        
        # Score all paths and select top K
        scored_paths = [(self._score_path(path), path) for path in all_paths]
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        top_paths = [path for _, path in scored_paths[:top_k]]
        
        print(f"Found {len(all_paths)} total paths, executing top {len(top_paths)} plans")
        
        # Execute each top path sequentially
        for path_idx, path in enumerate(top_paths):
            print(f"\nExecuting plan {path_idx + 1} (score: {self._score_path(path):.4f})")
            
            # Collect all trajectories along this path
            path_trajectories = []
            for i in range(1, len(path)):  # Skip root node (no trajectory to root)
                child = path[i]
                if child.prev_trajectory is not None:
                    trajectory = child.prev_trajectory
                    # Ensure trajectory is numpy array
                    if isinstance(trajectory, torch.Tensor):
                        trajectory = trajectory.cpu().numpy()
                    # Ensure shape is (traj_len, 8)
                    if trajectory.ndim == 3:
                        trajectory = trajectory[0]  # Take first batch if needed
                    path_trajectories.append(trajectory)
            
            # Concatenate all trajectories in this path to form complete plan
            if len(path_trajectories) > 0:
                # Stack trajectories sequentially for this path: (total_len, 8)
                complete_trajectory = np.concatenate(path_trajectories, axis=0)
                
                # Add batch dimension: (1, total_len, 8)
                complete_trajectory = complete_trajectory[np.newaxis, ...]
                
                # Execute the complete trajectory for this plan
                self.simulator.set_state(root_node.state.unsqueeze(0).repeat(self.simulator.num_envs, 1))
                if self.simulator.render_mode is not None:
                    self.simulator.render()
                self.simulator.robot_controller.execute_trajectory(complete_trajectory)
                if self.simulator.render_mode is not None:
                    self.simulator.render()
                
                print(f"  Completed plan {path_idx + 1}: {len(path)} nodes, {complete_trajectory.shape[1]} trajectory steps")
        
        return top_paths

