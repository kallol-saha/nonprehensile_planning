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
        object_state: torch.Tensor, 
        simulator: Simulator,
        available_object_indices: list[int] = None,
        parent=None, 
        prev_action=None,
        value=0.,
        visit_count=0,
        C = 2.0,
        ):

        # --------------------------------------------- #
        
        self.id = id
        self.object_state = object_state
        self.available_object_indices = available_object_indices
        self.parent = parent
        self.simulator = simulator
        self.prev_action = prev_action

        # --------------------------------------------- #

        self.children: list = []
        self.sampled: bool = False    # Whether this node has sampled children or not.
        self.terminal: bool = False
        self.visit_count: int = visit_count
        self.total_value = value
        self.C = C

    @property
    def uct_score(self):

        V = self.total_value
        n = self.visit_count
        
        if self.parent is None:
            return V / n

        N = self.parent.visit_count
        if N == 0:
            return np.inf

        if n == 0:
            return np.inf

        uct_score = V / n + self.C * np.sqrt(np.log(N) / n)
        # uct_score = V / n + self.C * np.sqrt(np.log(N))
        return uct_score

    @property
    def value(self):
        return self.total_value / self.visit_count

    def capture_image(self):
        
        self.simulator.set_object_poses_tensor(self.object_state)
        self.simulator.stall(1)
        image = self.simulator.capture_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class MCTS:

    def __init__(
        self, 
        simulator: Simulator, 
        start_object_state: torch.Tensor,
        C=2.0
        ):
        
        self.simulator = simulator

        self.start_object_state = start_object_state

        # Associated object index is -1 if we are not currently grasping an object, or have placed an object.
        # Basically this is only true when we are in the root node (or some weird motion planning waypoint)
        self.root_node = Node(
            id=0,
            object_state=start_object_state,
            simulator=simulator,
            available_object_indices=list(range(len(self.simulator.object_names))),
            C=C
        )
        self.root_node.visit_count = 1
        self.root_node.total_value = 0.
        self.num_nodes = 1

    # NOTE: Can be replaced with a multimodal policy (action sampler)
    def sample_actions_to_expand_node(self, node: Node):

        # I assume that we will be storing a batch of grasps with object indices, and a batch of placements with grasp indices.
        # So if we just placed an object, we should sample from the grasps other than the object we placed
        
        # Variables that I have:
        # - final_grasps: (N, 7) torch tensor => These have corresponding pre_grasps (N,7)
        # - final_grasp_object_indices: (N,) torch tensor
        # - final_placement_poses: (M, 7) torch tensor
        # - final_placement_grasp_indices: (M,) torch tensor => The index of final_grasps

        if node.sampled:
            return

        node.sampled = True

        if len(node.available_object_indices) == 0:
            return

        actions = self.simulator.sample_placement_actions(object_indices=node.available_object_indices)
        obj_indices = actions[:, 0].long()

        new_object_states = copy.deepcopy(node.object_state)
        new_object_states[torch.arange(self.simulator.num_envs), obj_indices, :3] = actions[:, 1:4]

        self.simulator.set_object_poses_tensor(new_object_states)

        self.simulator.wait_for_stability()

        new_object_states = self.simulator.object_poses_tensor

        for i in range(actions.shape[0]):
            # Extract object index from action and remove it from available indices
            obj_index = obj_indices[i].item()
            new_available_indices = [idx for idx in node.available_object_indices if idx != obj_index]
            
            child_node = Node(
                id=self.num_nodes,
                object_state=new_object_states[i].unsqueeze(0).repeat(self.simulator.num_envs, 1, 1),
                simulator=self.simulator,
                available_object_indices=new_available_indices,
                parent=node,
                prev_action=actions[i],
                C=node.C
            )
            self.num_nodes += 1
            # Automatically visit the node once it is created, so it helps node selection.
            child_node.visit_count = 1
            child_node.total_value = self.compute_value(child_node)

            node.children.append(child_node)
        
        

    # NOTE: Can be replaced with a Value function
    def compute_value(self, node: Node):

        self.simulator.set_object_poses_tensor(node.object_state)
        N, _ = self.simulator.number_of_objects_in_shelf()
        
        disturbance = torch.mean(torch.abs(node.object_state - node.parent.object_state))

        value = N + 1 / (1 + disturbance.item())
        
        return value
    
    def rollout(self, node: Node, max_rollout_steps: int = 10):

        # Get the current node.
        # Randomly select a child (next placement or next grasp)
        
        current_node = node
        i = 0
        while not self.is_terminal_node(current_node) and i < max_rollout_steps:
            current_node = self.select_random_child(current_node)
            i+=1

        return self.compute_value(current_node)

    def is_terminal_node(self, node: Node):

        if not node.sampled:
            self.sample_actions_to_expand_node(node)

        # If we have sampled this node's children:
        if len(node.children) == 0:
            return True
        else:
            return False
            
    def is_leaf_node(self, node: Node):

        # NOTE: All terminal nodes are leaf nodes, but all leaf nodes are not terminal nodes.

        if self.is_terminal_node(node):
            return True

        if not node.sampled:
            self.sample_actions_to_expand_node(node)
        
        # Atleast one of the children of this node have been visited only during expansion.
        for child in node.children:
            if child.visit_count == 1:
                return True
        
        return False
            
    def select_best_child(self, node: Node):
        
        # If the node is not sampled, sample actions to expand it, so we have all possible children.
        if not node.sampled:
            self.sample_actions_to_expand_node(node)
        
        uct_scores = np.array([child.uct_score for child in node.children])
        best_child_index = np.argmax(uct_scores)
        best_child = node.children[best_child_index]

        return best_child
    
    def select_random_child(self, node: Node):
        
        if not node.sampled:
            self.sample_actions_to_expand_node(node)
        
        random_child_index = np.random.randint(len(node.children))
        random_child = node.children[random_child_index]
        
        return random_child
    
    def select_random_unexpanded_child(self, node: Node):

        if not node.sampled:
            self.sample_actions_to_expand_node(node)

        unexplored_children = []
        for child in node.children:
            if child.visit_count == 1:
                unexplored_children.append(child)
        
        assert len(unexplored_children) > 0, "No unexplored children found, this should not be possible"

        random_child_index = np.random.randint(len(unexplored_children))
        random_child = unexplored_children[random_child_index]
        
        return random_child
    
    def plan(self, iterations: int = 100, graph_dir: str = "assets/graphs"):

        # First build a tree of depth 1, so we have all possible children for the root node.
        self.sample_actions_to_expand_node(self.root_node)
        
        for i in range(iterations):
        
            print(f"MCTS Iteration {i}")
            current_node = self.root_node

            # 1. SELECTION
            # Traverse the tree until we reach a leaf node or a terminal node.
            while not (self.is_leaf_node(current_node) or self.is_terminal_node(current_node)):
                current_node = self.select_best_child(current_node)

            # 2. EXPANSION
            # If the node is terminal, compute the value, skip rollout, and backpropagate the value.
            if self.is_terminal_node(current_node):
                value = self.compute_value(current_node)

            elif self.is_leaf_node(current_node):

                # Double-making sure that the node has children (it should have, because we just selected it)
                self.sample_actions_to_expand_node(current_node)

                # 3. ROLLOUT
                current_node = self.select_random_unexpanded_child(current_node)
                value = self.rollout(current_node, max_rollout_steps=0) # TODO: Variant of MCTS without rollout, just selection.

            else:
                raise ValueError("Current node is not a leaf node or a terminal node, this should not be possible")

            # 4. BACKPROPAGATION
            while current_node is not None:
                current_node.visit_count += 1
                current_node.total_value += value
                current_node = current_node.parent

        
        create_graph(graph_dir, self.root_node)
        # TODO: Return the best plan (Need to forward compute the trajectories maximizing the values of children nodes and return them concatenated)

            
# ----------------------------------------------------------------- #

def create_graph(graph_dir: str, root_node: Node):

    node_images_dir = os.path.join(graph_dir, "node_images")
    os.makedirs(node_images_dir, exist_ok=True)

    graph = gv.Digraph()
    graph = add_nodes_recursively(root_node, graph, node_images_dir)
    graph.render(
        os.path.join(graph_dir, "graph"), view=False
    )

    sampled_graph = gv.Digraph()
    sampled_graph = add_nodes_sampled(root_node, sampled_graph, node_images_dir)
    sampled_graph.render(
        os.path.join(graph_dir, "sampled_graph"), view=False
    )

    selected_graph = gv.Digraph()
    selected_graph = add_nodes_selected(root_node, selected_graph, node_images_dir)
    selected_graph.render(
        os.path.join(graph_dir, "selected_graph"), view=False
    )

    # TODO: Save node info, and save the best plan, so we can replay it at any time.

def add_nodes_recursively(
    node: Node, graph: gv.Digraph, node_images_dir
):
    # Get rendered image of node:
    node_image = node.capture_image()
    image_file_path = os.path.join(node_images_dir, str(node.id) + ".png")
    cv2.imwrite(image_file_path, node_image)

    # Add node to the graph
    U = "inf" if node.uct_score == np.inf else np.round(node.uct_score, 2)
    graph.node(
        str(node.id),
        label=f"N={node.visit_count}, V={np.round(node.total_value, 2)} \n id={node.id} \n UCT={U}",
        image=os.path.join(os.getcwd(), image_file_path),
        penwidth="2",
        imagescale="true",
        width="1.5",
        height="1.5",
        shape="square",
        labelloc="b",
    )

    # Add edge to the graph:
    if node.parent is not None:
        # Assign color to the edge:
        if node.sampled:
            color = "blue"
        else:
            color = "black"

        graph.edge(
            str(node.parent.id),
            str(node.id),
            color=color,
            # label=f"deviation: {node.deviation:9.5f}",
            penwidth="10",
        )

    # Loop through the children and recurse the function:
    for child in node.children:
        graph = add_nodes_recursively(
            child, graph, node_images_dir
        )

    return graph

def add_nodes_sampled(
    node: Node, graph: gv.Digraph, node_images_dir
):
    # Get rendered image of node:
    node_image = node.capture_image()
    image_file_path = os.path.join(node_images_dir, str(node.id) + ".png")
    cv2.imwrite(image_file_path, node_image)

    # Add node to the graph
    U = "inf" if node.uct_score == np.inf else np.round(node.uct_score, 2)
    graph.node(
        str(node.id),
        label=f"N={node.visit_count}, V={np.round(node.total_value, 2)} \n id={node.id} \n UCT={U}",
        image=os.path.join(os.getcwd(), image_file_path),
        penwidth="2",
        imagescale="true",
        width="1.5",
        height="1.5",
        shape="square",
        labelloc="b",
    )

    # Add edge to the graph:
    if node.parent is not None:
        # Assign color to the edge:
        if node.sampled:
            color = "blue"
        else:
            color = "black"

        graph.edge(
            str(node.parent.id),
            str(node.id),
            color=color,
            # label=f"deviation: {node.deviation:9.5f}",
            penwidth="10",
        )

    # Loop through the children and recurse the function:
    for child in node.children:
        if child.sampled:
            graph = add_nodes_sampled(
                child, graph, node_images_dir
            )

    return graph


def add_nodes_selected(
    node: Node, graph: gv.Digraph, node_images_dir
):
    # Get rendered image of node:
    node_image = node.capture_image()
    image_file_path = os.path.join(node_images_dir, str(node.id) + ".png")
    cv2.imwrite(image_file_path, node_image)

    # Add node to the graph
    U = "inf" if node.uct_score == np.inf else np.round(node.uct_score, 2)
    graph.node(
        str(node.id),
        label=f"N={node.visit_count}, V={np.round(node.total_value, 2)} \n id={node.id} \n UCT={U}",
        image=os.path.join(os.getcwd(), image_file_path),
        penwidth="2",
        imagescale="true",
        width="1.5",
        height="1.5",
        shape="square",
        labelloc="b",
    )

    # Add edge to the graph:
    if node.parent is not None:
        # Assign color to the edge:
        if node.visit_count > 1:
            color = "blue"
        else:
            color = "black"

        graph.edge(
            str(node.parent.id),
            str(node.id),
            color=color,
            # label=f"deviation: {node.deviation:9.5f}",
            penwidth="10",
        )

    # Loop through the children and recurse the function:
    for child in node.children:
        if child.visit_count > 1:
            graph = add_nodes_selected(
                child, graph, node_images_dir
            )

    return graph

