import os
from copy import deepcopy

import numpy as np

from vtamp.pybullet_env.scene_gen.generate_scene import Scene
from vtamp.suggester_utils import invert_transformation
from vtamp.table_bussing_states import sample_initial_poses
from vtamp.utils.pcd_utils import transform_object_pcd


class KeyboardDemo:
    def __init__(self, cfg, sim: Scene):
        self.sim = sim

        # Prepare data folder:
        # self.folder_path = os.path.join(cfg.data_folder, cfg.task_name)
        # os.makedirs(self.folder_path, exist_ok=True)
        # os.makedirs(os.path.join(self.folder_path, "train"), exist_ok=True)
        # os.makedirs(os.path.join(self.folder_path, "test"), exist_ok=True)

        # self.train_demos = len(os.listdir(os.path.join(self.folder_path, "train")))
        # self.test_demos = len(os.listdir(os.path.join(self.folder_path, "test")))

        # self.T_ground_to_table = np.load(cfg.extrinsics_file)["T"]
        # self.T_table_to_ground = invert_transformation(self.T_ground_to_table)

    def save_data(self, prev_state, state, moved_obj, mode="train"):
        self.sim.set_state(prev_state)

        prev_pcd, prev_pcd_seg, _ = self.sim.get_observation()
        initial_pose = self.sim.get_object_pose(moved_obj)
        initial_transform = self.sim.pose_to_transformation(initial_pose)

        self.sim.set_state(state)

        final_pose = self.sim.get_object_pose(moved_obj)
        final_transform = self.sim.pose_to_transformation(final_pose)

        # transform = np.matmul(invert_transformation(initial_transform), final_transform)
        transform = final_transform @ invert_transformation(initial_transform)
        transform = self.T_table_to_ground @ transform @ self.T_ground_to_table

        # pcd, pcd_seg, _ = self.sim.get_observation()

        # Make the fixed object segmentation IDs as 0
        # pcd_seg = np.where(np.isin(pcd_seg, self.sim.fixed_obj_ids), 0, pcd_seg)
        prev_pcd_seg = np.where(
            np.isin(prev_pcd_seg, self.sim.fixed_obj_ids), 0, prev_pcd_seg
        )

        placement_pcd = transform_object_pcd(
            prev_pcd, prev_pcd_seg, transform, moved_obj
        )
        obj_mask = np.where(prev_pcd_seg == moved_obj, 0, 1)

        # # Take all objects except moved_obj from prev_pcd:
        # prev_pcd_anchor = prev_pcd[prev_pcd_seg != moved_obj]
        # # Take action object from pcd:
        # pcd_action = pcd[pcd_seg == moved_obj]

        # # Concatenate the two point clouds
        # placement_pcd = np.concatenate((prev_pcd_anchor, pcd_action), axis=0)
        # placement_pcd_seg = np.concatenate((prev_pcd_seg[prev_pcd_seg != moved_obj], pcd_seg[pcd_seg == moved_obj]), axis=0)
        # obj_mask = np.where(placement_pcd_seg == moved_obj, 0, 1)

        # plot_pcd(pcd, obj_mask)

        if mode == "train":
            np.savez(
                os.path.join(
                    self.folder_path,
                    str(mode),
                    (str(self.train_demos) + "_teleport_obj_points.npz"),
                ),
                clouds=placement_pcd,
                masks=prev_pcd_seg,
                classes=obj_mask,
                previous_clouds=prev_pcd,
                moved_obj=moved_obj,
            )

            self.train_demos += 1

        elif mode == "test":
            np.savez(
                os.path.join(
                    self.folder_path,
                    str(mode),
                    (str(self.test_demos) + "_teleport_obj_points.npz"),
                ),
                clouds=placement_pcd,
                masks=prev_pcd_seg,
                classes=obj_mask,
                previous_clouds=prev_pcd,
                moved_obj=moved_obj,
            )

            self.test_demos += 1

    def collect(self):
        _ = input("Press Enter to Start collecting train demos")

        # sample_initial_poses(self.sim)      # This is table bussing specific
        print("\nReady to go!")
        prev_state = self.sim.get_state()

        while True:
            moved_obj = self.sim.control_objects()
            # if moved_obj:
            #     state = self.sim.get_state()
            #     save = input(
            #         "Do you want to save this transition? (y/n). Type 'R' to reset "
            #     )
            #     if save == "R":
            #         # sample_initial_poses(self.sim)        # This is table bussing specific
            #         state = self.sim.get_state()
            #         print("\nReady to go again!")
            #     # if save == "y" or save == "Y":
            #     #     self.save_data(prev_state, state, moved_obj, mode="train")
            #     prev_state = deepcopy(state)

        # self.sim.reset()
        # _ = input("Press Enter to Start collecting test demos")

        # prev_state = self.sim.get_state()

        # while True:
        #     moved_obj = self.sim.control_objects()
        #     if moved_obj:
        #         state = self.sim.get_state()
        #         save = input("Do you want to save this transition? (y/n) If done with train collection, type 'DONE' ")
        #         if save == "DONE":      # If done, the last transition is not saved
        #             break
        #         if save == "y" or save == "Y":
        #             self.save_data(prev_state, state, moved_obj, mode = "test")
        #         prev_state = deepcopy(state)
