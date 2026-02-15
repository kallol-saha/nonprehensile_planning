import os

import numpy as np
import yaml

from demo import KeyboardDemo

with open("global_config.yaml", "r") as config:
    args = yaml.load(config, Loader=yaml.FullLoader)

demo = KeyboardDemo(args, -1)

path = args["demo_folder"] + args["exp_name"] + "/"

# Loop through the folder and get all .npz files
npz_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npz")]

out_dir = args["demo_folder"] + args["exp_name"] + "/train2/"

k = 0

# Example: Load and print content of each .npz file
for file in npz_files:
    data = np.load(file)
    pcd = data["clouds"]
    pcd_seg = data["masks"]
    classes = data["classes"]

    demo.plot_pcd(pcd, pcd_seg)

    n = 4
    table_indices = np.where(pcd_seg == 0)[0]
    keep_indices = table_indices[::n]
    mask = np.ones(pcd_seg.shape, dtype=bool)
    mask[table_indices] = False
    mask[keep_indices] = True
    new_pcd = pcd[mask]
    new_pcd_seg = pcd_seg[mask]
    new_classes = classes[mask]

    demo.plot_pcd(new_pcd, new_classes)

    np.savez(
        out_dir + str(k) + "_teleport_obj_points.npz",
        clouds=new_pcd,
        masks=new_pcd_seg,
        classes=new_classes,
    )

    k += 1
