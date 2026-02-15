import argparse
import hydra

from vtamp.pybullet_env.scene_gen.tetris3d_scene import Scene

def main(cfg):

    env = Scene(cfg, gui=True, robot=True)
    env.wait_for_stability()
    env.reset()
    env.step()  # Call the step function to start the simulation
    env.close()  # Close the simulation when done

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