import yaml
import argparse
import hydra

from vtamp.demo_collection import KeyboardDemo
from vtamp.pybullet_env.scene_gen.tetris3d_scene import Scene

def main(cfg):

    env = Scene(cfg, gui=True, robot=False)

    demo = KeyboardDemo(cfg, env)

    print("\nLeft = <--")
    print("Right = -->")
    print("Up = Arrow up")
    print("Down = Arrow down")
    print("Roll in, roll out = [ ]")
    print("Pitch in, pitch out = ; ' ")
    print("Yaw in, yaw out = , . ")
    print("Change focus object = /\n")

    demo.collect()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate the planning algorithm")
    parser.add_argument(
        "--config",
        type=str,
        default="tetris3d.yaml",
        help="config for the planner",
    )

    args = parser.parse_args()

    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(args.config)
    
    main(cfg)