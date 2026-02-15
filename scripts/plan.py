import numpy as np
import yaml

from vtamp.planning.geometric_suggester import Suggester
from vtamp.pybullet_env.scene_gen.generate_scene import Scene

with open("plan_config.yaml", "r") as config:
    args = yaml.load(config, Loader=yaml.FullLoader)

path = args["demo_folder"] + args["exp_name"] + "/demo_0/"
data = np.load(path + "pcd_" + str(2) + ".npz")
pcd = data["clouds"]
pcd_seg = data["masks"]

suggester = Suggester(args)
env = Scene(args, gui=True)

suggester = None
goal = None
start = env.get_goal()
