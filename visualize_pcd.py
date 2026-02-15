import open3d as o3d
import numpy as np
import torch

import matplotlib.pyplot as plt

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