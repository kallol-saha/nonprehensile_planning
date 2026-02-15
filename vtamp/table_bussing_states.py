import argparse

import hydra
import numpy as np

from vtamp.pybullet_env.scene_gen.generate_scene import Scene

# Default z's
BOWL_Z = 0.64
CUP_Z = 0.69
PLATE_Z = 0.626


def sample_initial_poses(env: Scene):
    """
    Assumes all objects are loaded on the table, and that the table is centered at (0, 0)
    """

    objs = env.movable_objects

    placement_limit = 0.3
    sampling_grid_density = int((placement_limit * 2) / 0.0001)

    # Create a dense uniform grid for sampling bowl and cup positions
    x_grid = np.linspace(-placement_limit, placement_limit, sampling_grid_density)
    y_grid = np.linspace(-placement_limit, placement_limit, sampling_grid_density)
    X, Y = np.meshgrid(x_grid, y_grid)
    possible_positions = np.vstack([X.ravel(), Y.ravel()]).T

    plates = []
    bowls = []
    cups = []
    for i, obj in enumerate(objs):
        if "plate" in obj:
            plates.append(i)
        elif "bowl" in obj:
            bowls.append(i)
        elif "cup" in obj:
            cups.append(i)

    # Loop over all objects in dict, and initialize a flag of whether something can be placed in the object:
    is_object_occupied = {}
    for i, obj in enumerate(objs):
        is_object_occupied[obj] = False

    # First place the plates:
    for plate in plates:
        obj_name = objs[plate]
        pose = env.get_object_pose(obj_name)

        # Sample random (x, y) coordinates within the placement limit
        limit = placement_limit - env.object_radii[obj_name]
        x, y = np.random.uniform(-limit, limit, 2)
        # Update the pose
        pose[:3] = [x, y, PLATE_Z]
        env.set_object_pose(obj_name, pose)

    # Place the bowls either on top of a plate or on the table (without collision)

    bowl_mask = np.ones(possible_positions.shape[0], dtype=bool)
    for bowl in bowls:
        mask = bowl_mask.copy()

        # Get the radius of the bowl
        bowl_radius = env.object_radii[objs[bowl]]

        # Choose where to place the bowl, either on a plate or on the table
        choice = np.random.choice(["plate", "table"])

        if choice == "plate":
            # Choose a random plate to place the bowl on
            available_plates = [
                plate for plate in plates if not is_object_occupied[objs[plate]]
            ]
            if len(available_plates) == 0:  # If no plates are available, place on table
                choice = "table"
            else:
                plate = np.random.choice(available_plates)
                is_object_occupied[objs[plate]] = True

                # Get the position of the plate
                plate_pose = env.get_object_pose(objs[plate])
                plate_pos = plate_pose[:2]

                # Get the radius of the plate
                plate_radius = env.object_radii[objs[plate]]

                # Get the distance threshold
                distance = plate_radius - bowl_radius

                # Get the mask of the possible positions
                mask = mask & (
                    np.linalg.norm(possible_positions - plate_pos, axis=1) < distance
                )  # Can only place inside of the plate

        if choice == "table":
            # Mask out the plates
            for plate in plates:
                # Get the position of the plate
                plate_pose = env.get_object_pose(objs[plate])
                plate_pos = plate_pose[:2]

                # Get the radius of the plate
                plate_radius = env.object_radii[objs[plate]]

                # Get the distance threshold
                distance = plate_radius + bowl_radius

                # Get the mask of the possible positions
                mask = mask & (
                    np.linalg.norm(possible_positions - plate_pos, axis=1) > distance
                )  # Can only place outside of the plate

            # Visualize mask with cv2

            # vis_mask = mask.reshape(sampling_grid_density, sampling_grid_density)
            # plt.imshow(vis_mask, cmap='gray')
            # plt.title("Mask")
            # plt.show()

            # Shrink the edges of the table
            distance = bowl_radius
            mask = mask & (
                placement_limit - np.max(np.abs(possible_positions), axis=1) > distance
            )

        obj_name = objs[bowl]
        pose = env.get_object_pose(obj_name)

        # Visualize mask with cv2
        # vis_mask = mask.reshape(sampling_grid_density, sampling_grid_density)
        # plt.imshow(vis_mask, cmap='gray')
        # plt.title("Mask")
        # plt.show()

        valid_positions = possible_positions[mask]
        x, y = valid_positions[np.random.choice(valid_positions.shape[0])]
        # Update the pose
        pose[:3] = [x, y, BOWL_Z]
        env.set_object_pose(obj_name, pose)

        # Mask out the bowl placed currently
        bowl_mask = bowl_mask & (
            np.linalg.norm(possible_positions - [x, y], axis=1) > bowl_radius * 2
        )

    cup_mask = np.ones(possible_positions.shape[0], dtype=bool)
    for cup in cups:
        mask = cup_mask.copy()

        # Get the radius of the cup
        cup_radius = env.object_radii[objs[cup]]

        # Choose where to place the cup, either on a plate or on a bowl, or on the table
        choice = np.random.choice(["plate", "bowl", "table"])

        if choice == "plate":
            # Choose a random plate to place the cup on
            available_plates = [
                plate for plate in plates if not is_object_occupied[objs[plate]]
            ]
            if len(available_plates) == 0:
                choice = "bowl"
            else:
                plate = np.random.choice(available_plates)
                is_object_occupied[objs[plate]] = True

                # Get the position of the plate
                plate_pose = env.get_object_pose(objs[plate])
                plate_pos = plate_pose[:2]

                # Get the radius of the plate
                plate_radius = env.object_radii[objs[plate]]

                # Get the distance threshold
                distance = plate_radius - cup_radius

                # Get the mask of the possible positions
                mask = mask & (
                    np.linalg.norm(possible_positions - plate_pos, axis=1) < distance
                )

        if choice == "bowl":
            # Choose a random bowl to place the cup on
            available_bowls = [
                bowl for bowl in bowls if not is_object_occupied[objs[bowl]]
            ]
            if len(available_bowls) == 0:
                choice = "table"
            else:
                bowl = np.random.choice(available_bowls)
                is_object_occupied[objs[bowl]] = True

                # Get the position of the bowl
                bowl_pose = env.get_object_pose(objs[bowl])
                bowl_pos = bowl_pose[:2]

                # Get the radius of the bowl
                bowl_radius = env.object_radii[objs[bowl]]

                # Get the distance threshold
                distance = bowl_radius - cup_radius

                # Get the mask of the possible positions
                mask = mask & (
                    np.linalg.norm(possible_positions - bowl_pos, axis=1) < distance
                )

        if choice == "table":
            # Mask out the plates
            for plate in plates:
                # Get the position of the plate
                plate_pose = env.get_object_pose(objs[plate])
                plate_pos = plate_pose[:2]

                # Get the radius of the plate
                plate_radius = env.object_radii[objs[plate]]

                # Get the distance threshold
                distance = plate_radius + cup_radius

                # Get the mask of the possible positions
                mask = mask & (
                    np.linalg.norm(possible_positions - plate_pos, axis=1) > distance
                )

            # Mask out the bowls
            for bowl in bowls:
                # Get the position of the bowl
                bowl_pose = env.get_object_pose(objs[bowl])
                bowl_pos = bowl_pose[:2]

                # Get the radius of the bowl
                bowl_radius = env.object_radii[objs[bowl]]

                # Get the distance threshold
                distance = bowl_radius + cup_radius

                # Get the mask of the possible positions
                mask = mask & (
                    np.linalg.norm(possible_positions - bowl_pos, axis=1) > distance
                )

            # Shrink the edges of the table
            distance = cup_radius
            mask = mask & (
                placement_limit - np.max(np.abs(possible_positions), axis=1) > distance
            )

        obj_name = objs[cup]
        pose = env.get_object_pose(obj_name)

        # # Visualize mask with cv2
        # vis_mask = mask.reshape(sampling_grid_density, sampling_grid_density)
        # plt.imshow(vis_mask, cmap='gray')

        # plt.title("Mask")
        # plt.show()

        valid_positions = possible_positions[mask]
        x, y = valid_positions[np.random.choice(valid_positions.shape[0])]
        # Update pose
        pose[:3] = [x, y, CUP_Z]
        env.set_object_pose(obj_name, pose)

        # Mask out the cup placed currently
        cup_mask = cup_mask & (
            np.linalg.norm(possible_positions - [x, y], axis=1) > cup_radius * 2
        )

    env.wait_for_stability()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the planning algorithm")
    parser.add_argument(
        "--config",
        type=str,
        default="blocks.yaml",
        help="config for the planner",
    )
    args = parser.parse_args()

    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(args.config)

    env = Scene(cfg, gui=True, robot=True)

    sample_initial_poses(cfg, env)
