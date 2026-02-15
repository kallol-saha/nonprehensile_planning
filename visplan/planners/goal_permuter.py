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
NODE_COUNT = 0

class Node:

    def __init__(
        self, 
        id: int,
        object_state: torch.Tensor, 
        simulator: Simulator,
        available_object_indices: list[int],
        parent=None, 
        value=0.,
        C = 2.0,
        ):

        # --------------------------------------------- #
        
        self.id = id
        self.object_state = object_state
        self.available_object_indices = available_object_indices
        self.parent = parent
        self.simulator = simulator

        # --------------------------------------------- #

        self.children: list = []
        self.sampled: bool = False    # Whether this node has sampled children or not.
        self.terminal: bool = False
        self.total_value = value
        self.C = C

    def capture_image(self):
        
        self.simulator.set_object_poses_tensor(self.object_state.unsqueeze(0).repeat(self.simulator.num_envs, 1, 1))
        self.simulator.stall(1)
        image = self.simulator.capture_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
        
        
def recursive_node_addition(node: Node, goal_object_state: torch.Tensor):

    global NODE_COUNT

    for i in range(len(node.available_object_indices)):

        obj_index = node.available_object_indices[i]
        new_available_indices = [idx for idx in node.available_object_indices if idx != obj_index]

        new_object_state = copy.deepcopy(node.object_state)
        new_object_state[obj_index] = goal_object_state[obj_index]

        child_node = Node(
            id=NODE_COUNT,
            object_state=new_object_state,
            simulator=node.simulator,
            available_object_indices=new_available_indices,
            parent=node,
            value=0.,
            C=2.0,
        )
        NODE_COUNT += 1
        node.children.append(child_node)

        recursive_node_addition(child_node, goal_object_state)

def form_graph(simulator: Simulator, start_object_state: torch.Tensor, goal_object_state: torch.Tensor):

    global NODE_COUNT

    root_node = Node(
        id=NODE_COUNT,
        object_state=start_object_state,
        simulator=simulator,
        available_object_indices=list(range(len(simulator.object_names))),
        parent=None,
        value=0.,
        C=2.0,
    )
    NODE_COUNT += 1

    recursive_node_addition(root_node, goal_object_state)

    return root_node

            
# ----------------------------------------------------------------- #

def create_graph(graph_dir: str, root_node: Node):

    node_images_dir = os.path.join(graph_dir, "node_images")
    os.makedirs(node_images_dir, exist_ok=True)

    graph = gv.Digraph()
    graph = add_nodes_recursively(root_node, graph, node_images_dir)
    graph.render(
        os.path.join(graph_dir, "graph"), view=False
    )

    # sampled_graph = gv.Digraph()
    # sampled_graph = add_nodes_sampled(root_node, sampled_graph, node_images_dir)
    # sampled_graph.render(
    #     os.path.join(graph_dir, "sampled_graph"), view=False
    # )

    # selected_graph = gv.Digraph()
    # selected_graph = add_nodes_selected(root_node, selected_graph, node_images_dir)
    # selected_graph.render(
    #     os.path.join(graph_dir, "selected_graph"), view=False
    # )

    # TODO: Save node info, and save the best plan, so we can replay it at any time.

def add_nodes_recursively(
    node: Node, graph: gv.Digraph, node_images_dir
):
    # Get rendered image of node:
    node_image = node.capture_image()
    image_file_path = os.path.join(node_images_dir, str(node.id) + ".png")
    cv2.imwrite(image_file_path, node_image)

    # Add node to the graph
    graph.node(
        str(node.id),
        label=f"id={node.id}",
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

