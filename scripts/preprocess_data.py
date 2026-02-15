import os

import numpy as np
import yaml

from demo import KeyboardDemo

with open("global_config.yaml", "r") as config:
    args = yaml.load(config, Loader=yaml.FullLoader)

demo = KeyboardDemo(args, -1)

path = args["demo_folder"] + args["exp_name"] + "/demo_0/"

# Loop through the folder and get all .npz files
npz_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npz")]

out_dir = args["demo_folder"] + args["exp_name"] + "/cross_object/"

k = 0

# Example: Load and print content of each .npz file
for file in npz_files:
    data = np.load(file)
    pcd = data["clouds"]
    pcd_seg = data["masks"]
    classes = data["classes"]

    objs = np.unique(pcd_seg)
    action = np.unique(pcd_seg[classes == 0])[0]
    possible_anchors = objs[objs != action]

    for anchor in possible_anchors:
        indices = (pcd_seg == action) | (pcd_seg == anchor)
        new_pcd = pcd[indices]
        new_classes = classes[indices]
        new_pcd_seg = pcd_seg[indices]

        # demo.plot_pcd(new_pcd, new_classes)

        np.savez(
            out_dir + str(k) + "_teleport_obj_points.npz",
            clouds=new_pcd,
            masks=new_pcd_seg,
            classes=new_classes,
        )
        k += 1
