import torch
import numpy as np
import copy
import cv2
import os
import graphviz as gv
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors

from visplan.utils import to_torch_pose
from visplan.submodules.robo_utils.robo_utils.conversion_utils import move_pose_along_local_z

POST_GRASP_LIFT = 0.15


def create_graph(graph_dir: str, root_node):
    node_images_dir = os.path.join(graph_dir, "node_images")
    os.makedirs(node_images_dir, exist_ok=True)

    cmap = mpl_cm.get_cmap("viridis")
    graph = gv.Digraph()
    graph = add_nodes_recursively(root_node, graph, node_images_dir, cmap)

    graph.render(
        os.path.join(graph_dir, "graph"), view=False
    )


def add_nodes_recursively(
    node, graph: gv.Digraph, node_images_dir, cmap
):
    # Get rendered image of node:
    node_image = node.capture_image()
    image_file_path = os.path.join(node_images_dir, str(node.id) + ".png")
    cv2.imwrite(image_file_path, node_image)

    # Add node to the graph
    graph.node(
        str(node.id),
        label=f"id={node.id}\nN={node.visit_count}, R={np.round(node.reward, 2)}, Q={np.round(node.Q, 2)}",
        image=os.path.join(os.getcwd(), image_file_path),
        penwidth="2",
        imagescale="true",
        width="1.5",
        height="1.5",
        shape="square",
        labelloc="b",
    )

    # Add edge to the graph (color by this node's rollout_value, assumed in [0, 1])
    if node.parent is not None:
        t = np.clip(getattr(node, "rollout_value", 0.0), 0.0, 1.0)
        color = mpl_colors.to_hex(cmap(t))
        graph.edge(
            str(node.parent.id),
            str(node.id),
            color=color,
            penwidth="10",
        )

    for child in node.children:
        graph = add_nodes_recursively(child, graph, node_images_dir, cmap)

    return graph

def add_nodes_sampled(
    node, graph: gv.Digraph, node_images_dir
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

