import argparse
import os
import re
import shutil
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from vtamp.evaluation import update_plan_metrics, evaluate_plan
from vtamp.mde import MDE
from vtamp.object_suggester import ObjectSuggester
# from vtamp.planners.a_star import Astar as Planner
from vtamp.planners.beam_search import BeamSearch as Planner
from vtamp.suggester import Suggester
from vtamp.suggester_utils import set_seed
from vtamp.heuristics import table_bussing_heuristic as heuristic
from vtamp.heuristics import table_bussing_goal as goal

from vtamp.pybullet_env.scene_gen.generate_scene import Scene

from vtamp.utils.pcd_utils import plot_pcd

def main(cfg, seed, overwrite: bool = False):
    # Reproducibility:
    set_seed(
        seed
    )  # NOTE that the torch seed gets reset in the get_data() function inside TestPointCloudDataset, should be fine for now.
    OmegaConf.update(cfg, "seed", seed, force_add=True)

    # Prepare the output directory for the benchmark:
    output_dir = os.path.join("benchmarks", f"{cfg.benchmark_name}_{seed}")
    if os.path.exists(output_dir):
        if not overwrite:
            raise FileExistsError(
                f"The directory {output_dir} already exists and cannot be "
                "overwritten. Use --overwrite if you want to overwrite"
            )
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save the configs that are used into the output folder
    shutil.copy(
        os.path.join("configs/", str(args.config)),
        os.path.join(output_dir, "plan.yaml"),
    )
    shutil.copy(
        os.path.join("configs/training", str(cfg.suggester.config_file)),
        os.path.join(output_dir, "suggester.yaml"),
    )

    # Load models and planner
    suggester = Suggester(cfg)
    mde = MDE(cfg.mde)
    object_suggester = ObjectSuggester(cfg.object_suggester)
    planner = Planner(cfg, suggester, mde, object_suggester)

    # Inside each benchmark, there is a directory for each eval folder
    # inside each eval folder, there is a directory for each example

    # Create simulator
    env = Scene(
        cfg,
        seed=seed,
        gui=cfg.gui,
        robot=True,
    )

    initial_pcd, initial_pcd_seg, rgb = env.get_observation()

    plan_start = time.time()
    plan, start_node, pruned_node_count, exp_node_count = planner.plan(
        initial_pcd,
        initial_pcd_seg,
        heuristic,
        goal
    )
    planning_time = time.time() - plan_start

    found_plan, path_length, gen_node_count, exp_node_count, total_suggestions_made = evaluate_plan(
        cfg,
        output_dir,
        plan,
        start_node,
        initial_pcd,
        initial_pcd_seg,
        remove_outliers=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the planning algorithm")
    parser.add_argument(
        "--config",
        type=str,
        default="blocks.yaml",
        help="config for the planner",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="force overwrite existing results",
    )
    args = parser.parse_args()

    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(args.config)

    if cfg.ablation == "greedy":
        assert (
            cfg.suggester.num_suggestions == 1
        ), "branching factor must be 1 to run greedy ablation"

    main(cfg, seed=0, overwrite=args.overwrite)
