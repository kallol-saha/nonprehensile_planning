from collections import defaultdict
import pickle
import re
import os
import open3d as o3d
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


FOLDER_PATH = "assets/data/p2p_eval"
os.makedirs(FOLDER_PATH + "/train", exist_ok=True)
os.makedirs(FOLDER_PATH + "/test", exist_ok=True)
os.makedirs(FOLDER_PATH + "/initial_states", exist_ok=True)


def plot_pcd(pcd, pcd_seg=None, frame=False, colormap="tab10"):
    if type(pcd) == torch.Tensor:
        pcd = pcd.cpu().detach().numpy()
    if pcd_seg is not None and type(pcd_seg) == torch.Tensor:
        pcd_seg = pcd_seg.cpu().detach().numpy()

    pts_vis = o3d.geometry.PointCloud()
    pts_vis.points = o3d.utility.Vector3dVector(pcd)

    if pcd_seg is not None:
        seg_ids = np.unique(pcd_seg)
        n = len(seg_ids)
        cmap = plt.get_cmap(colormap)
        id_to_color = {uid: cmap(i / n)[:3] for i, uid in enumerate(seg_ids)}

        # # for blocks
        # id_to_color = {
        #     0: COLORS["gray"],
        #     1: COLORS["brown"],
        #     2: COLORS["red"],
        #     3: COLORS["green"],
        #     4: COLORS["blue"],
        # }

        colors = np.array([id_to_color[seg_id] for seg_id in pcd_seg])
        pts_vis.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pts_vis]

    if frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        geometries.append(frame)

    o3d.visualization.draw_geometries(geometries)


def depth_to_pcd(depth, projection_matrix, view_matrix, seg_img=None):

    tran_pix_world = np.linalg.inv(
        np.matmul(projection_matrix, view_matrix)
    )  # Pixel to 3D transformation

    height = depth.shape[1]
    width = depth.shape[0]

    # create a mesh grid with pixel coordinates, by converting 0 to width and 0 to height to -1 to 1
    y, x = np.mgrid[-1 : 1 : 2 / height, -1 : 1 : 2 / width]
    y *= -1.0  # y is reversed in pixel coordinates

    # Reshape to single dimension arrays
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)

    # Homogenize:
    pixels = np.stack([x, y, z, np.ones_like(z)], axis=1)
    # filter out "infinite" depths:
    # fin_depths = np.where(z < 0.99)
    # pixels = pixels[fin_depths]

    # filter out depths where seg id is 0
    if seg_img is not None:
        seg_img = np.array(seg_img)
        fin_depths = seg_img.reshape(-1) != 0
        pixels = pixels[fin_depths]
        pcd_seg = seg_img.reshape(-1)[fin_depths]
    else:
        pcd_seg = None

        fin_depths = np.arange(pixels.shape[0])

    # Depth z is between 0 to 1, so convert it to -1 to 1.
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # if seg_img is not None:
    #     seg_img = np.array(seg_img)
    #     pcd_seg = seg_img.reshape(-1)[fin_depths]  # filter out "infinite" depths
    # else:
    #     pcd_seg = None

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3:4]  # Homogenize in 3D
    points = points[:, :3]  # Remove last axis ones

    return points, pcd_seg, fin_depths


class DataConverter:
    def __init__(self):
        self.num_demos = 0

    def save_transition(self, prev_pcd, placement_pcd, pcd_seg, obj_mask, moved_obj, mode="train"):
        np.savez(
            os.path.join(
                FOLDER_PATH,
                str(mode),
                (str(self.num_demos) + "_teleport_obj_points.npz"),
            ),
            clouds=placement_pcd,
            masks=pcd_seg,
            classes=obj_mask,
            previous_clouds=prev_pcd,
            moved_obj=moved_obj,
        )
        self.num_demos += 1

    def save_initial_state(self, pcd, pcd_seg):
        np.savez(
            os.path.join(
                FOLDER_PATH,
                "initial_states",
                (str(self.num_demos) + "_teleport_obj_points.npz"),
            ),
            clouds=pcd,
            masks=pcd_seg,
        )
        self.num_demos += 1

    def convert_data(self, filename, mode="train"):

        # Open and read the pickle file
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            
            time_to_pcds = defaultdict(list)
            time_to_pcd_segs = defaultdict(list)
            
            # rgbs = data[0]['rgb']   # (t, H, W, 3)

            # for t in range(rgbs.shape[0]):

            #     # Visualize with cv2
            #     rgb = rgbs[t]
            #     cv2.imshow(f"RGB_{t}", rgb)
            #     cv2.waitKey(0)
            
            for key in data[0].keys():
                if match := re.fullmatch(r'point_cloud_(\d+)sampling', key):
                    seg_id = int(match.group(1))
                else:
                    continue

                segmented_pcds = data[0][key]  # (t, N, 3)
                T, N = segmented_pcds.shape[:2]
                for t in range(T):
                    time_to_pcds[t].append(segmented_pcds[t])

                for t in range(T):
                    segmentation_mask = np.ones(N) * seg_id
                    time_to_pcd_segs[t].append(segmentation_mask)

            for t in range(T):
                time_to_pcds[t] = np.concatenate(time_to_pcds[t], axis=0)
                time_to_pcd_segs[t] = np.concatenate(time_to_pcd_segs[t], axis=0)

                
            for t in range(1, 2):
                assert t == 1
                prev_pcd = time_to_pcds[t-1].astype('float')
                pcd = time_to_pcds[t].astype('float')
                pcd_seg = time_to_pcd_segs[t].astype('float')

                # Figure out moved object and create object mask
                diff = np.linalg.norm(pcd - prev_pcd, axis=1)
                max_diff = 0
                moved_obj = None
                for seg_id in np.unique(pcd_seg):
                    if diff[pcd_seg == seg_id].mean() > max_diff:
                        max_diff = diff[pcd_seg == seg_id].mean()
                        moved_obj = seg_id

                obj_mask = np.where(pcd_seg == moved_obj, 0, 1)

                # Combine pcd[i-1] with pcd of moved object from pcd[i] to get placement_pcd
                # placement_pcd = np.concatenate([prev_pcd[pcd_seg != moved_obj], pcd[pcd_seg == moved_obj]])
                # placement_pcd_seg = np.concatenate([pcd_seg[pcd_seg != moved_obj], pcd_seg[pcd_seg == moved_obj]])        

                placement_pcd = prev_pcd.copy()
                placement_pcd[pcd_seg == moved_obj] = pcd[pcd_seg == moved_obj]
                
                # plot_pcd(prev_pcd, obj_mask)
                # plot_pcd(placement_pcd, obj_mask)

                # self.save_transition(prev_pcd, placement_pcd, pcd_seg, obj_mask, moved_obj, mode=mode)
                self.save_initial_state(prev_pcd, pcd_seg)

            return pcd, pcd_seg


if __name__ == "__main__":
    converter = DataConverter()

    # example_name = "cupboard_3"
    # folder_name = os.path.join("assets", "data", "points2plans", "Training", example_name)
    folder_name = os.path.join("assets", "data", "points2plans", "Test")
    for demo_num in tqdm(range(1, len(os.listdir(folder_name)) + 1)):
        demo_str = f"demo_{demo_num:06d}.pickle"
        file_name = os.path.join(folder_name, demo_str)
        converter.convert_data(file_name, mode="train")
