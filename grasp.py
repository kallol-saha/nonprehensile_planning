import argparse

import hydra
import numpy as np

from vtamp.pybullet_env.scene_gen.generate_scene import Scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the planning algorithm")
    parser.add_argument(
        "--config",
        type=str,
        default="table_bussing_sim.yaml",
        help="config for the planner",
    )
    args = parser.parse_args()

    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(args.config)

    # Create simulator
    env = Scene(
        cfg,
        seed=0,
        gui=cfg.gui,
        robot=True,
    )

    # Saves pcd and pcd_seg to be used by Contact-GraspNet
    # pcd, pcd_segs, rgb = env.get_observation()
    # np.save("initial_pcd.npy", pcd)
    # np.save("initial_pcd_seg.npy", pcd_segs)

    contact_graspnet_results = np.load(
        "assets/grasps/contact_graspnet_results.npz", allow_pickle=True
    )
    grasp_poses = {}
    rel_grasp_poses = {}
    for object_id in [2, 3, 4]:
        scores = contact_graspnet_results["scores"].item()[object_id]
        idx = np.argmax(scores)
        grasp = contact_graspnet_results["pred_grasps_cam"].item()[object_id][idx]

        view_matrix = np.asarray(env.camera_list[0].view_matrix).reshape(
            [4, 4], order="F"
        )
        T = np.array(
            [
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )
        grasp_W = np.linalg.inv(view_matrix) @ T @ grasp
        # env.draw_frame(grasp_W)
        # print()

        grasp_pose_W = env.transformation_to_pose(grasp_W)
        grasp_poses[object_id] = grasp_pose_W

        # Calculate relative grasp pose
        rel_grasp_pose = env.transformation_to_pose(
            np.linalg.inv(env.pose_to_transformation(env.get_object_pose(object_id)))
            @ grasp_W
        )
        rel_grasp_poses[object_id] = rel_grasp_pose

    # print(rel_grasp_poses)

    env.move_object_with_grasp(env.initial_state, "cup", np.eye(4))
