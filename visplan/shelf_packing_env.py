import json
import os
import copy
import time
import sapien
from sympy.core import I
import torch
import shutil
import numpy as np
from typing import Tuple
from omegaconf import OmegaConf

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.utils import randomization
# from plan_scene_builder import TableSceneBuilder
from mani_skill.utils.building import articulations, actors
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from sapien.physx import PhysxRigidBodyComponent

from visplan.robot_controller import RobotController
from visplan.action_primitives import ActionPrimitives
from visplan.action_sampling import ActionSampler
from visplan.env_utils import ManiSkillEnvUtils
from visplan.graphs.directed_acyclic import DAG

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from tqdm import tqdm
from visplan.utils import (
    load_grasps, 
    to_numpy_pose, 
    to_sapien_pose, 
    to_torch_pose,
    flip_quat_about_z,
    flip_pose_quat_about_z,
    load_acronym_object_and_grasps,
    point_in_cuboid,
    generate_placement_quaternions
)

from visplan.submodules.robo_utils.robo_utils.visualization.plotting import (
    plot_pcd, 
    plot_pcd_with_highlighted_segment
)
from visplan.submodules.robo_utils.robo_utils.visualization.point_cloud_structures import make_gripper_visualization
from visplan.submodules.robo_utils.robo_utils.conversion_utils import (
    pose_to_transformation, 
    invert_transformation, 
    transform_pcd, 
    transformation_to_pose,
    move_pose_along_local_z,
    move_pose_along_local_x,
    move_pose_along_local_y,
    furthest_point_sample,
    downsample_point_cloud
)
from models.flowmatch_actor.modeling.policy.denoise_actor_3d_packing import DenoiseActor
from models.flowmatch_actor.modeling.policy.value_network import ValueNetwork
from models.flowmatch_actor.utils.common_utils import count_parameters

from visplan.generation_utils import (
    create_shelf, 
    sample_object_pose_on_table, 
    sample_object_pose_on_table_multi_object,
    sample_object_pose_on_shelf, 
    sample_point_in_fixed_rectangle, 
    sample_point_in_fixed_rectangle_uniformly,
    compute_ray_box_intersection
)

CUROBO_ASSETS_PATH = "visplan/submodules/curobo/src/curobo/content/assets/"
POST_GRASP_LIFT = 0.15
# GRASP_DEPTH = 0.2
GRASP_DEPTH = 0.25
GRASP_RETRACTION_DISTANCE = 0.15
EE_LINK_CENTER_TO_GRIPPER_TIP = 0.13

CUROBO_SUCCESSES = 0

OPEN = 1
CLOSED = -1


def load_checkpoint_for_eval(checkpoint_path, model):
    """Load from checkpoint."""
    print("=> trying checkpoint '{}'".format(checkpoint_path))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    try:
        model_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
    except Exception as e:
        if "numpy.core.multiarray.scalar" in str(e):
            print(f"Warning: Falling back to weights_only=False due to numpy scalar in checkpoint: {e}")
            model_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False
            )
            # Convert numpy scalars to Python types if present
            if "best_loss" in model_dict and hasattr(model_dict["best_loss"], "item"):
                model_dict["best_loss"] = float(model_dict["best_loss"].item())
        else:
            raise
    # Load weights flexibly
    msn, unxpct = model.load_state_dict(model_dict["weight"], strict=False)
    if msn:
        print(f"Missing keys (not found in checkpoint): {len(msn)}")
        print(msn)
    if unxpct:
        print(f"Unexpected keys (ignored): {len(unxpct)}")
        print(unxpct)
    if not msn and not unxpct:
        print("All keys matched successfully!")

    print("=> loaded successfully '{}' (step {})".format(
        checkpoint_path, model_dict.get("iter", 0)
    ))
    del model_dict
    torch.cuda.empty_cache()

    return model


@register_env("ShelfPackingMultiObject-v1", max_episode_steps=10000)
class ShelfPackingMultiObject(BaseEnv, ManiSkillEnvUtils):
    
    # ---------------- INITIALIZATION ---------------- #

    def __init__(
        self, 
        *args, 
        num_objects=1,
        model_path=None,
        value_model_path=None,
        env_path="assets/environments/train",
        env_index=0,
        execution_mode = "rollout",     # TODO: "rollout" for rolling out data collection/policy,  or "replay" for replaying a recorded rollout.
        **kwargs):

        self.robot_init_qpos_noise = 0.
        self.action_primitives = None

        self.execution_mode = execution_mode

        self.num_objects = num_objects
        self.env_path = env_path
        self.env_index = env_index
        
        # self.scene_bounds = [-0.7, -0.7, -0.1, 0.2, 0.7, 0.8]        # [x_min, y_min, z_min, x_max, y_max, z_max]
        # NOTE: I define new bounds for the camera positioning based on the real world workspace, hopefully this does not affect the point cloud:
        self.scene_bounds = [-0.4, -0.5, 0.01, 0.3, 0.5, 0.8]        # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.camera_bounds = [-0.4, -0.75, 0.01, 0.45, 0.75, 0.8]        # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.pcd_bounds = [-0.4, -0.8, 0.01, 0.45, 0.8, 0.8]        # [x_min, y_min, z_min, x_max, y_max, z_max]

        # Load replay data before super().__init__() because initialize_from_replay_data() 
        # (called from _load_scene during super().__init__()) needs self.scene_data
        if self.execution_mode == "replay":
            self.load_replay_data()

        super().__init__(*args, robot_uids="panda", **kwargs)
        
        if model_path is not None:
            self.model_path = model_path
            self.load_placement_policy()
        
        if value_model_path is not None:
            self.value_model_path = value_model_path
            self.load_value_function()

        self.action_sampler = ActionSampler(self)

    def load_replay_data(self):
        # Load in the environment data from environment.yaml only
        env_yaml_path = os.path.join(self.env_path, f"env_{self.env_index}", "environment.yaml")
        
        if not os.path.exists(env_yaml_path):
            raise FileNotFoundError(f"environment.yaml not found in {self.env_path}/env_{self.env_index}")
        
        self.scene_data = OmegaConf.load(env_yaml_path)
    
    def load_placement_policy(self):        
        print(f"Loading model from {self.model_path}")
        self.placement_model = DenoiseActor(
            embedding_dim=120,
            num_attn_heads=8,
            nhist=1,
            num_shared_attn_layers=4,
            relative=False,
            rotation_format='quat_wxyz',
            denoise_timesteps=10,
            denoise_model='rectified_flow',
            lv2_batch_size=1
        )

        # Print basic modules' parameters
        count_parameters(self.placement_model)

        # Useful for some placement_models to ensure parameters are contiguous
        for name, param in self.placement_model.named_parameters():
            if param.requires_grad and param.ndim > 1 and not param.is_contiguous():
                print(f"Fixing layout for: {name}")
                param.data = param.contiguous()

        self.placement_model = load_checkpoint_for_eval(self.model_path, self.placement_model)   

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.placement_model.to(device)
        self.placement_model.eval()
    
    def load_value_function(self):
        """Load the value function model."""
        print(f"Loading value function from {self.value_model_path}")
        self.value_model = ValueNetwork(
            embedding_dim=120,
            num_attn_heads=8,
            nhist=1,
            num_shared_attn_layers=4
        )

        # Print basic modules' parameters
        count_parameters(self.value_model)

        # Useful for some models to ensure parameters are contiguous
        for name, param in self.value_model.named_parameters():
            if param.requires_grad and param.ndim > 1 and not param.is_contiguous():
                print(f"Fixing layout for: {name}")
                param.data = param.contiguous()

        self.value_model = load_checkpoint_for_eval(self.value_model_path, self.value_model)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.value_model.to(device)
        self.value_model.eval()

    @property
    def _default_sim_config(self):          
        return SimConfig(spacing=3)              # Use the default simulation parameters

    @property
    def _default_sensor_configs(self):
        # registers one 512x512 camera looking at the robot and objects

        # NOTE:Everything is defined in world frame (center of table)

        self.shelf_front_center = np.array([
                self.shelf_pose[0] - (self.shelf_depth / 2) * np.cos(self.shelf_rotation),
                self.shelf_pose[1] - (self.shelf_depth / 2) * np.sin(self.shelf_rotation),
                self.shelf_distance_from_floor + self.wall_thickness + self.shelf_height/2
            ])

        # NOTE: I limit the bounds so the shelf camera can only see halfway across the back wall of the shelf.

        azimuth_bound = np.arctan2(self.shelf_width / 2, self.shelf_depth)  # Tan inverse of (width/2 by depth)
        azimuth_angle = -(np.pi - self.shelf_rotation) + np.random.uniform(-azimuth_bound, azimuth_bound)

        elevation_bound = np.arctan2(self.shelf_height / 2, self.shelf_depth)  # Tan inverse of (height/2 by depth)
        elevation_angle = np.random.uniform(-elevation_bound, elevation_bound)

        shelf_cam_point = compute_ray_box_intersection(
            ray_origin=self.shelf_front_center,
            azimuth_angle=azimuth_angle,
            elevation_angle=elevation_angle,
            box_min=self.camera_bounds[:3],
            box_max=self.camera_bounds[3:]
        )        # (x, y, z) of the shelf camera point

        # TODO: Hardcoding this for now, but later, randomize the object camera:
        object_cam_look_at = np.array([-0.065, 0., 0.05])
        object_look_at_to_shelf_cam = object_cam_look_at - shelf_cam_point
        shelf_cam_angle = np.arctan2(object_look_at_to_shelf_cam[1], object_look_at_to_shelf_cam[0])
        object_cam_azimuth_angle = shelf_cam_angle + np.pi/2
        object_cam_elevation_angle = np.pi / 4     # 45 degrees

        object_cam_point = compute_ray_box_intersection(
            ray_origin=object_cam_look_at,
            azimuth_angle=object_cam_azimuth_angle,
            elevation_angle=object_cam_elevation_angle,
            box_min=self.camera_bounds[:3],
            box_max=self.camera_bounds[3:]
        )        # (x, y, z) of the object camera point

        # TODO: Randomize a bit the camera point, bring it in by a bit randomly, bringing it out may go into the table sometimes

        # Shelf camera pose:
        self.camera_1_pose = sapien_utils.look_at(
            eye=shelf_cam_point,
            target=self.shelf_front_center
        )
        print(f"shelf_cam_point: {shelf_cam_point}, shelf_front_center: {self.shelf_front_center}")
        # Object camera pose:
        self.camera_2_pose = sapien_utils.look_at(
            eye=object_cam_point,
            target=object_cam_look_at
        )

        self.camera_1_transform = pose_to_transformation(self.camera_1_pose.raw_pose[0].cpu().numpy(), format='wxyz')
        self.camera_2_transform = pose_to_transformation(self.camera_2_pose.raw_pose[0].cpu().numpy(), format='wxyz')

        return [
            CameraConfig(
                "camera_1",
                pose=self.camera_1_pose,
                width=256,
                height=256,
                fov= np.pi/2,
                # near=0.01,
                # far=100,
            ),
            CameraConfig(
                "camera_2",
                pose=self.camera_2_pose,
                width=256,
                height=256,
                fov= np.pi/2,
                # near=0.01,
                # far=100,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering
        width = 512
        height = 512
        # pose = sapien_utils.look_at([-0.2, 0.75, 0.6], [0.0, 0.0, 0.35])
        pose = sapien_utils.look_at(
            eye = [0.3, -0.75, self.shelf_front_center[2] + 0.2],
            target = self.shelf_front_center
        )
        self.video_frames = np.zeros((0, width, height, 3))
        return CameraConfig(
            "render_camera", pose=pose, width=width, height=height, fov=1.2, near=0.01, far=100
        )
    
    def _load_agent(self, options: dict):
        '''
        Loads in the robot
        '''
        # set a reasonable initial pose for the agent that doesn't intersect other objects

        self.robot_base_pose = sapien.Pose(p=[0.615, 0, 0], q=[1, 0, 0, 0])
        self.world_to_robot_base_transform = pose_to_transformation(
            np.concatenate([self.robot_base_pose.p, self.robot_base_pose.q]),
            format='wxyz'
            )
        self.world_to_robot_base_transform = torch.tensor(self.world_to_robot_base_transform, dtype=torch.float32, device=self.device)
        super()._load_agent(options, self.robot_base_pose)

        self.robot_links = self.agent.robot.get_links()
        # self.robot_link_names is already defined in the super()._load_agent() function.
        self.gripper_link_names = self.robot_link_names[-6:]

        # self.set_gripper_friction(friction_coefficient=2.0)
        
    def initialize_from_replay_data(self):

        self.object_names = []

        object_data = self.scene_data.objects
        self.num_objects = len(object_data)

        grasp_dir = "assets/final_grasps"

        for i, (_, object_properties) in enumerate(object_data.items()):

            object_name = object_properties.object_id
            object_pose = np.array(object_properties.pose)

            # New format: Load URDF path, bounds, quat, and scale_factor from YAML
            urdf_path = os.path.join(self.env_path, f"env_{self.env_index}", object_properties.urdf_filename)
            if not os.path.exists(urdf_path):
                raise FileNotFoundError(f"Object URDF not found at {urdf_path}")
            
            # Load bounds, quaternion, and scale_factor from YAML
            self.bounds[object_name] = np.array(object_properties.bounds)
            self.object_quaternions[object_name] = np.array(object_properties.quaternion)
            self.object_scale_factors[object_name] = float(object_properties.scale_factor)
            
            # Only load grasps and grasp_costs from object directory (load_full=False)
            object_dir = os.path.join("assets/acronym_objects", object_name)
            grasps, grasp_costs, object_volume = load_acronym_object_and_grasps(object_dir, grasp_dir, load_full=False)
            self.grasps[object_name] = grasps
            self.grasp_costs[object_name] = grasp_costs
            self.object_volumes[object_name] = object_volume
            self.object_poses[object_name] = object_pose
            self.object_names.append(object_name)

            actor_builders = self.urdf_loader.parse(urdf_path)["actor_builders"]
            builder = actor_builders[0]
            builder.initial_pose = to_sapien_pose(object_pose)
            self.objects[object_name] = builder.build(name=object_name)
            self.object_ids[object_name] = self.objects[object_name].per_scene_id[0].item()
            self.actors.append(self.objects[object_name])
            self.movable_actors.append(self.objects[object_name])

        # ----------------- SHELF ----------------- #
        
        shelf_data = self.scene_data.shelf

        # Dimensions:
        self.shelf_width = shelf_data.width
        self.shelf_depth = shelf_data.depth
        self.shelf_height = shelf_data.height
        self.shelf_distance_from_floor = shelf_data.distance_from_floor
        self.wall_thickness = shelf_data.wall_thickness
        self.bottom_wall_height = shelf_data.bottom_wall_height

        self.shelf_ceiling = self.shelf_distance_from_floor + self.wall_thickness + self.shelf_height

        shelf_urdf_path = os.path.join(self.env_path, f"env_{self.env_index}", shelf_data.urdf_filename)
        if not os.path.exists(shelf_urdf_path):
            raise FileNotFoundError(f"Shelf URDF not found at {shelf_urdf_path}")

        # Pose:
        self.shelf_pose = np.array(shelf_data.pose)

        # Extract shelf rotation:
        quat = np.array(shelf_data.pose[3:])
        rotation = R.from_quat(quat, scalar_first=True)
        self.shelf_rotation = rotation.as_euler('xyz')[2]

        # Extract shelf position:
        self.shelf_position = np.array(shelf_data.pose[:3])

        # Load in the shelf:
        actor_builders = self.urdf_loader.parse(shelf_urdf_path)["articulation_builders"]
        builder = actor_builders[0]
        builder.initial_pose = to_sapien_pose(self.shelf_pose)
        self.shelf = builder.build(name="shelf")
        self.actors.append(self.shelf)

        self.total_volume = sum(self.object_volumes.values())

    def initialize_randomly(self):

        # Automatically load in random object from json file:
        # Load random objects from final_grasps folder
        grasp_dir = "assets/final_grasps"
        available_objects = [f.replace('.npz', '') for f in os.listdir(grasp_dir) if f.endswith('.npz')]
        self.object_names = np.random.choice(
            available_objects,
            size=self.num_objects,
            replace=False
        ).tolist()

        # First pass: Load metadata only (needed for positioning)
        self.urdf_paths = {}
        for object_name in self.object_names:

            object_dir = os.path.join("assets/acronym_objects", object_name)
            urdf_path, grasps, bounds, quat, scale_factor, grasp_costs, object_volume = load_acronym_object_and_grasps(object_dir, grasp_dir, load_full=True)

            self.urdf_paths[object_name] = urdf_path
            self.grasps[object_name] = grasps
            self.bounds[object_name] = bounds
            self.object_quaternions[object_name] = quat
            self.object_scale_factors[object_name] = scale_factor
            self.grasp_costs[object_name] = grasp_costs
            self.object_volumes[object_name] = object_volume

        # SHELF GENERATION:

        self.shelf_width = np.random.uniform(0.35, 0.55)    # X-axis Real Shelf is 0.38m wide, has to be atleast 0.35 for gripper to fit
        self.shelf_depth = np.random.uniform(0.2, 0.55)    # Y-axis Real Shelf is 0.43m deep
        self.shelf_height = np.random.uniform(0.18, 0.4)    # Z-axis Real Shelf is 0.20m high, has to be atleast 0.18 for all objects to fit, and atleast 0.12 for gripper to fit, and atleast 0.3 for gripper to fit top to bottom.
        # self.shelf_height = fixed_volume / (self.shelf_width * self.shelf_depth)
        self.shelf_distance_from_floor = np.random.uniform(0.1, 0.4)    # Z-axis Real Shelf is 0.36m from the floor

        self.wall_thickness = np.random.uniform(0.01, 0.05) # Real Shelf is 0.038m thick
        self.bottom_wall_height = np.random.uniform(self.wall_thickness, (self.shelf_distance_from_floor - self.wall_thickness)) # Real Shelf is 0.088m from the floor

        self.shelf_ceiling = self.shelf_distance_from_floor + self.wall_thickness + self.shelf_height

        create_shelf(
            shelf_width = self.shelf_width, 
            shelf_height = self.shelf_height, 
            shelf_depth = self.shelf_depth,     # 0.15 to 0.5
            wall_thickness = self.wall_thickness, 
            shelf_distance_from_floor = self.shelf_distance_from_floor,
            bottom_wall_height = self.bottom_wall_height,
            color = (0.6, 0.4, 0.2) # Brown
        )

        # Load in the shelf:
        actor_builders = self.urdf_loader.parse("assets/shelf/shelf.urdf")["articulation_builders"]    # Articulation builders are for urdfs with multiple links
        builder = actor_builders[0]
        builder.initial_pose = sapien.Pose(p=[0., 0.5, 1.])
        self.shelf = builder.build(name="shelf") #, fix_root_link=False)
        self.actors.append(self.shelf)

        # Random initialization of shelf and object poses:
        # This also updates self.object_names to only include successfully placed objects
        self.position_scene_elements()

        # Second pass: Build actors only for successfully placed objects
        for object_name in self.object_names:
            urdf_path = self.urdf_paths[object_name]
            object_dim_z = self.bounds[object_name][2][1] - self.bounds[object_name][2][0]

            actor_builders = self.urdf_loader.parse(urdf_path)["actor_builders"]
            builder = actor_builders[0]
            builder.initial_pose = sapien.Pose(p=[0., 0., object_dim_z/2 + 0.02], q=self.object_quaternions[object_name])
            self.objects[object_name] = builder.build(name=object_name)
            self.object_ids[object_name] = self.objects[object_name].per_scene_id[0].item()
            self.actors.append(self.objects[object_name])
            self.movable_actors.append(self.objects[object_name])

        self.total_volume = sum(self.object_volumes.values())
    
    def _load_scene(self, options: dict):
        
        # Load in the table:        
        self.scene_builder = TableSceneBuilder(env = self,
                                             robot_init_qpos_noise = self.robot_init_qpos_noise)    # Noise in starting position of robot
        self.scene_builder.build()  # NOTE: The table scene is built such that z=0 is the tables surface

        self.actors = []
        self.movable_actors = []

        self.table = self.scene_builder.table
        self.table_id = self.table.per_scene_id[0].item()
        self.actors.append(self.table)

        # Load in the urdf loader
        self.urdf_loader = self.scene.create_urdf_loader()
        self.urdf_loader.load_multiple_collisions_from_file = True

        # self.object_filenames = np.random.choice(os.listdir("assets/acronym_objects_valid"), size=1)
        # self.object_filenames = ["bfb42be12a21fd047485047779434488"]
        # self.object_filenames = ["8589e65944de365351c41225db8e334"]
        
        # Initialize object properties dictionary
        self.objects = {}
        self.object_ids = {}
        self.grasps = {}
        self.bounds = {}
        self.grasp_costs = {}
        self.object_quaternions = {}
        self.object_scale_factors = {}
        self.object_poses = {}
        self.object_angles = {}
        self.object_volumes = {}

        print("")

        if self.execution_mode == "rollout":
            self.initialize_randomly()
        elif self.execution_mode == "replay":
            self.initialize_from_replay_data()       
            # NOTE: No need to position scene elements for replay mode, since the poses are stored in the demo yaml file.
        
        x_min = -self.shelf_depth/2
        x_max = self.shelf_depth/2
        y_min = -self.shelf_width/2
        y_max = self.shelf_width/2
        z_min = self.shelf_distance_from_floor + self.wall_thickness/2
        z_max = z_min + self.shelf_height
        
        # For calculating bounds:
        self.shelf_corners = np.array(
            [[x_min, y_min, z_min], 
            [x_max, y_min, z_min], 
            [x_min, y_max, z_min], 
            [x_max, y_max, z_min], 
            [x_min, y_min, z_max], 
            [x_max, y_min, z_max], 
            [x_min, y_max, z_max], 
            [x_max, y_max, z_max]])

        shelf_transform = pose_to_transformation(self.shelf_pose, format='wxyz')
        self.transform_to_axis_aligned_shelf = invert_transformation(shelf_transform)
        # self.shelf_corners = transform_pcd(self.shelf_corners, shelf_transform)

    def position_object(
        self, 
        object_name,
        fixed_rect_centers,
        fixed_rect_dimensions,
        fixed_rect_angles,
        ):
        
        object_dim_x = self.bounds[object_name][0][1] - self.bounds[object_name][0][0]
        object_dim_y = self.bounds[object_name][1][1] - self.bounds[object_name][1][0]
        object_dim_z = self.bounds[object_name][2][1] - self.bounds[object_name][2][0]

        # Add clearance to the object dimensions:
        object_dim_x += 0.1
        object_dim_y += 0.1

        # Sample object poses:
        result = sample_object_pose_on_table_multi_object(
            x_min = 0.25, # Minimum X bound of the sampling rectangle   TODO: 0.615 is hardcoded.
            x_max = 0.7, # Maximum X bound of the sampling rectangle   NOTE: These values are hardcoded according to real world bounds, so do not change without good reason.
            y_min = -0.5, # Minimum Y bound of the sampling rectangle
            y_max = 0.07, # Maximum Y bound of the sampling rectangle
            fixed_rect_centers = fixed_rect_centers,
            fixed_rect_dimensions = fixed_rect_dimensions,
            fixed_rect_angles = fixed_rect_angles,
            sampled_rect_dimensions = (object_dim_x, object_dim_y)
            )

        if result is None:
            return None

        cx, cy, angle = result
        rotation = R.from_euler('z', [angle])
        quaternion = rotation.as_quat().squeeze()  # Returns [x, y, z, w]
        
        return np.array([
            cx - 0.615,
            cy,
            object_dim_z/2 + 0.01,
            quaternion[3],               # Convert to [w, x, y, z] format for SAPIEN
            quaternion[0],
            quaternion[1],
            quaternion[2]
        ]), angle

    def position_scene_elements(self):

        # ------------------ POSITIONING THE SHELF ------------------ #
            
        # TODO: Initially, I am not randomizing. Later, I will bring this back for the final benchmark.
        
        radius = 0.7 #np.random.uniform(0.6, 0.8)
        theta = 1.3 #np.random.uniform(-np.pi/3, np.pi/3)     # Shelf position angle
        # NOTE: Shelf base frame is exactly at the center of the shelf.

        self.shelf_rotation = 1.5708 #theta + np.random.uniform(-0.35, 0.35)        # Shelf orientation angle
        self.shelf_rotation += np.random.uniform(-0.15, 0.15)
        
        # Convert z-axis rotation to quaternion using scipy
        rotation = R.from_rotvec([0, 0, self.shelf_rotation])
        quaternion = rotation.as_quat()  # Returns [x, y, z, w]
        
        self.shelf_position = np.array([0.42 - 0.615, 0.35, 0.])     # TODO: Hardcoding with respect to real world shelf position.
        self.shelf_position += np.array([
            np.random.uniform(-0.02, 0.02), 
            np.random.uniform(-0.07, 0.07),
             0.
             ])


        self.shelf_pose = np.array([
            self.shelf_position[0],
            self.shelf_position[1],
            self.shelf_position[2],
            quaternion[3],               # Convert to [w, x, y, z] format for SAPIEN
            quaternion[0],
            quaternion[1],
            quaternion[2]
        ])

        # -------------------------------------------------------------- #

        # ------------------ POSITION OBJECTS PROCEDURALLY ------------------ #

        fixed_rect_centers = []
        fixed_rect_dimensions = []
        fixed_rect_angles = []
        fixed_rect_centers.append((self.shelf_pose[0] + 0.615, self.shelf_pose[1]))
        fixed_rect_dimensions.append((self.shelf_width + 0.1, self.shelf_depth + 0.1))
        fixed_rect_angles.append(self.shelf_rotation + np.pi/2) 
        # NOTE: We have to add 90 degree to the rotation because shelf_width is actually along Y and shelf_depth is along X. 
        # Remember that from the robot's perspective, X is forward and Y is towards the left.
        
        successfully_placed_objects = []
        for object_name in self.object_names:

            result = self.position_object(
                object_name,
                fixed_rect_centers = fixed_rect_centers,
                fixed_rect_dimensions = fixed_rect_dimensions,
                fixed_rect_angles = fixed_rect_angles
            )

            # If position_object returns None, stop placing more objects
            if result is None:
                continue

            object_pose, object_angle = result

            object_dim_x = self.bounds[object_name][0][1] - self.bounds[object_name][0][0]
            object_dim_y = self.bounds[object_name][1][1] - self.bounds[object_name][1][0]

            fixed_rect_centers.append((object_pose[0] + 0.615, object_pose[1]))
            fixed_rect_dimensions.append((object_dim_x, object_dim_y))
            fixed_rect_angles.append(object_angle)

            self.object_poses[object_name] = object_pose
            self.object_angles[object_name] = object_angle
            successfully_placed_objects.append(object_name)

        # Clean up metadata for objects that weren't successfully placed
        objects_to_remove = set(self.object_names) - set(successfully_placed_objects)
        for object_name in objects_to_remove:
            del self.urdf_paths[object_name]
            del self.grasps[object_name]
            del self.bounds[object_name]
            del self.object_quaternions[object_name]
            del self.object_scale_factors[object_name]
            del self.grasp_costs[object_name]

        # Update object_names and num_objects to only include successfully placed objects
        self.object_names = successfully_placed_objects
        self.num_objects = len(successfully_placed_objects)

        # -------------------------------------------------------------- #
 
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):

            # b = len(env_idx)
            self.scene_builder.initialize(env_idx)      # The robot gets initialized here TODO: Need to edit this so the robot base is at 0.

            # Set the shelf pose:
            self.set_object_pose(self.shelf, self.shelf_pose)

            # Set the object poses:
            if hasattr(self, 'initial_object_poses'):
                for object_name in self.object_names:
                    self.set_object_pose(object_name, self.initial_object_poses[object_name])
            else:
                for object_name in self.object_names:
                    self.set_object_pose(object_name, self.object_poses[object_name])

            # Initialize the robot controller:
            self.robot_controller = RobotController(
                self,
                debug=False,
                vis=False, #(self.execution_mode == "replay"),       # Visualize the target grasp pose only in replay mode.
                base_pose=self.unwrapped.agent.robot.pose[0],
                visualize_target_grasp_pose=True,
                print_env_info=False,
            )

    def _before_control_step(self):
        """Code that runs before each action is taken.
        On GPU simulation this is called right after observations are fetched from the GPU buffers."""

        pass
    
    def _after_control_step(self):
        """Code that runs after each action has been taken.
        On GPU simulation this is called right before observations are fetched from the GPU buffers."""
        # if self.save_video:
        #     self.capture_video_frame()

        if self.render_mode is not None:
            self.render()
    
    def setup_primitives(self):
        self.initial_object_poses = {}
        for object_name in self.object_names:
            object_pose = self.get_object_pose(object_name)
            self.initial_object_poses[object_name] = copy.deepcopy(object_pose)

        self.home_pose = self.get_gripper_pose().clone()
        self.home_pose = move_pose_along_local_z(self.home_pose, -EE_LINK_CENTER_TO_GRIPPER_TIP)
        self.home_pose = to_torch_pose(self.home_pose)
        self.home_joints = self.get_joint_state().clone()

        self.action_primitives = ActionPrimitives(env=self)
    
    def get_object_current_dims(self, object_name):
        object_mesh = self.objects[object_name].get_collision_meshes()[0]     # Specifically for the cracker box.
        object_dim_x = object_mesh.vertices[:, 0].max() - object_mesh.vertices[:, 0].min()
        object_dim_y = object_mesh.vertices[:, 1].max() - object_mesh.vertices[:, 1].min()
        object_dim_z = object_mesh.vertices[:, 2].max() - object_mesh.vertices[:, 2].min()
        return object_dim_x, object_dim_y, object_dim_z
        
    # ------------------------------------------------------------------------- #
    
    # ---------------- TASK SPECIFIC SAMPLING AND VERIFICATION ---------------- #

    # ------------------------------------------------------------------------- #

    def check_object_within_gripper(self, object_name, env_idx = 0):
        
        """
        Check if the gripper is in contact with the specified object.
        """
        # Update contact information
        # self.scene.get_contacts()
        
        # Get the object and robot gripper links
        object_actor = self.objects[object_name]
        robot_links = self.agent.robot.get_links()
        
        # Check contact between object and each gripper finger
        left_finger = robot_links[-4]  # Assuming second-to-last link is left finger
        right_finger = robot_links[-3]  # Assuming last link is right finger
        
        # Check pairwise contacts
        left_contact_forces = self.scene.get_pairwise_contact_forces(object_actor, left_finger)[env_idx]
        right_contact_forces = self.scene.get_pairwise_contact_forces(object_actor, right_finger)[env_idx]
        
        # Check if there's significant contact (force magnitude > threshold)
        contact_threshold = 0.01  # Adjust as needed
        left_contact = torch.norm(left_contact_forces, dim=-1) > contact_threshold
        right_contact = torch.norm(right_contact_forces, dim=-1) > contact_threshold
        
        # Object is grasped if both fingers are in contact
        is_grasped = left_contact & right_contact
        
        return is_grasped.item()

    def check_gripper_collision(self):
        """
        Check if the gripper is in collision with any object for all environments (parallel).
        Returns a boolean tensor of shape (num_envs,) where True indicates collision.
        """
        # self.scene.get_contacts()
        gripper_contact_forces = self.agent.robot.get_net_contact_forces(self.gripper_link_names)
        # gripper contact forces is a tensor of shape (num_envs, 6, 3)
        contact_threshold = 0.01  # Adjust as needed TODO: Make this a hyperparameter.
        contact_norms = torch.norm(gripper_contact_forces, dim=-1)  # Shape: (num_envs, 6)
        return (contact_norms > contact_threshold).any(dim=-1)  # Shape: (num_envs,)

    def get_object_contact_norms(self, object_name):
        if  isinstance(object_name, str):
            contact_forces = self.objects[object_name].get_net_contact_forces()
        else:
            contact_forces = object_name.get_net_contact_forces()

        contact_norms = torch.norm(contact_forces, dim=-1)
        return contact_norms
        
    def check_object_inside_shelf(self, object_name, env_idx = 0):
        
        obj_position = self.get_object_pose(object_name)[env_idx, :3].unsqueeze(0)
        axis_aligned_obj_position = transform_pcd(obj_position.cpu().numpy(), self.transform_to_axis_aligned_shelf)
        return point_in_cuboid(self.shelf_corners, axis_aligned_obj_position[0])
    
    def number_of_objects_in_shelf(self, env_idx = 0):
        
        N = 0
        object_indices_in_shelf = []
        for i in range(len(self.object_names)):
            if self.check_object_inside_shelf(self.object_names[i], env_idx = env_idx):
                N += 1
                object_indices_in_shelf.append(i)

        return N, object_indices_in_shelf
    
    def number_of_objects_on_table(self, env_idx = 0):
        """
        Count the number of objects on the table using the same bounds as position_object.
        
        Args:
            env_idx: Environment index to check
            
        Returns:
            N: Number of objects on the table
            object_indices_on_table: List of object indices that are on the table
        """
        # Table bounds (same as in position_object, but in world frame)
        # In position_object: x_min=0.25, x_max=0.7, but then adjusted by -0.615
        # So in world frame: x from (0.25 - 0.615) = -0.365 to (0.7 - 0.615) = 0.085
        x_min = 0.25 - 0.615  # -0.365
        x_max = 0.7 - 0.615   # 0.085
        y_min = -0.5
        y_max = 0.07
        z_min = -0.1
        z_max = 0.4
        
        N = 0
        object_indices_on_table = []
        
        for i in range(len(self.object_names)):
            object_name = self.object_names[i]
            obj_position = self.get_object_pose(object_name)[env_idx, :3].cpu().numpy()  # (3,)
            
            # Check if object is within table bounds
            if (x_min <= obj_position[0] <= x_max and
                y_min <= obj_position[1] <= y_max and
                z_min <= obj_position[2] <= z_max):
                N += 1
                object_indices_on_table.append(i)
        
        return N, object_indices_on_table

    def compute_values(self, start_object_state: torch.Tensor):

        current_object_state = self.object_poses_tensor
        all_values = []
        for i in range(self.num_envs):
            N, _ = self.number_of_objects_in_shelf(env_idx = i)
            all_values.append(N)

        all_values = torch.tensor(all_values, device=self.device, dtype=torch.float32)

        disturbances = torch.mean(torch.abs(current_object_state - start_object_state), dim=(1, 2))
        all_values = all_values + 1 / (1 + disturbances)
        
        return all_values
    
    def set_state_batch(self, states: torch.Tensor):
        """
        Set states for a batch that may not match num_envs.
        
        Args:
            states: (B, state_size) tensor of states where B <= num_envs
            
        Raises:
            ValueError: If B > num_envs
        """
        batch_size = states.shape[0]
        
        if batch_size > self.num_envs:
            raise ValueError(
                f"Batch size ({batch_size}) exceeds num_envs ({self.num_envs}). "
                f"Cannot set more states than available environments."
            )
        
        # Get current state and update only the first batch_size environments
        current_state = self.get_state().clone()
        current_state[:batch_size] = states
        self.set_state(current_state)
    
    def calculate_mean_and_std_value(self):

        all_values = []
        
        for i in range(self.num_envs):
            value, _ = self.number_of_objects_in_shelf(env_idx = i)
            all_values.append(value)

        mean_value = np.mean(all_values)
        variance_value = np.var(all_values)
        std_value = np.sqrt(variance_value)  # Standard deviation = sqrt(variance)

        return mean_value, std_value

    # ------------------------------------------------------------------------- #

    # ---------------- VISUALIZATION ---------------- #
    
    def visualize_policy_predictions(self, out):
        # Colors for pairs (R, G, B) in [0,1]
        pair_colors = [
            (1.0, 0.0, 0.0),  # red
            (0.0, 1.0, 0.0),  # green
            (0.0, 0.0, 1.0),  # blue
            (1.0, 0.65, 0.0), # orange
            (0.56, 0.0, 1.0), # purple
        ]

        # Build visualization point clouds
        grasp_pcds = []
        grasp_rgbs = []
        place_pcds = []
        place_rgbs = []

        # TODO: This only considers 5 predictions at most.
        for i in range(min(5, out.shape[0])):
            pred_grasp_pose = out[i, 0, :7].cpu().detach().numpy()
            pred_place_pose = out[i, 1, :7].cpu().detach().numpy()

            pred_grasp_transform = pose_to_transformation(pred_grasp_pose, format="wxyz")
            pred_place_transform = pose_to_transformation(pred_place_pose, format="wxyz")

            # Use the same color for grasp and its paired place
            color = pair_colors[i]

            pred_grasp_gripper_pcd, pred_grasp_gripper_colors = make_gripper_visualization(
                rotation=pred_grasp_transform[:3, :3],
                translation=pred_grasp_transform[:3, 3],
                length=0.05,
                density=50,
                color=color,
            )

            pred_place_gripper_pcd, pred_place_gripper_colors = make_gripper_visualization(
                rotation=pred_place_transform[:3, :3],
                translation=pred_place_transform[:3, 3],
                length=0.05,
                density=50,
                color=color,
            )

            grasp_pcds.append(pred_grasp_gripper_pcd)
            grasp_rgbs.append(pred_grasp_gripper_colors)
            place_pcds.append(pred_place_gripper_pcd)
            place_rgbs.append(pred_place_gripper_colors)

        scene_pcd, scene_rgb, scene_seg, scene_mask = self.get_point_cloud()
        scene_rgb = scene_rgb.cpu().numpy() / 255.0

        full_pcd = np.vstack([scene_pcd] + grasp_pcds + place_pcds)
        full_rgb = np.vstack([scene_rgb] + grasp_rgbs + place_rgbs)

        plot_pcd(full_pcd, full_rgb) #, base_frame=True)
    
    def visualize_grasp_poses(self, grasp_poses):
        """
        Visualize grasp poses only.
        
        Args:
            grasp_poses: (n, 7) numpy array or torch tensor of grasp poses (x, y, z, qw, qx, qy, qz)
        """
        # Colors for grasps (R, G, B) in [0,1]
        grasp_colors = [
            (1.0, 0.0, 0.0),  # red
            (0.0, 1.0, 0.0),  # green
            (0.0, 0.0, 1.0),  # blue
            (1.0, 0.65, 0.0), # orange
            (0.56, 0.0, 1.0), # purple
            (1.0, 0.0, 1.0),  # magenta
            (0.0, 1.0, 1.0),  # cyan
            (1.0, 1.0, 0.0),  # yellow
        ]
        
        # Convert to numpy if torch tensor
        if isinstance(grasp_poses, torch.Tensor):
            grasp_poses = grasp_poses.cpu().detach().numpy()
        
        num_grasps = grasp_poses.shape[0]
        
        # Build visualization point clouds
        grasp_pcds = []
        grasp_rgbs = []
        
        for i in range(num_grasps):
            grasp_pose = grasp_poses[i, :7]
            
            grasp_transform = pose_to_transformation(grasp_pose, format="wxyz")
            
            # Cycle through colors
            color = grasp_colors[i % len(grasp_colors)]
            
            grasp_gripper_pcd, grasp_gripper_colors = make_gripper_visualization(
                rotation=grasp_transform[:3, :3],
                translation=grasp_transform[:3, 3],
                length=0.05,
                density=50,
                color=color,
            )
            
            grasp_pcds.append(grasp_gripper_pcd)
            grasp_rgbs.append(grasp_gripper_colors)
        
        scene_pcd, scene_rgb, scene_seg, scene_mask = self.get_point_cloud()
        scene_rgb = scene_rgb.cpu().numpy() / 255.0

        if isinstance(scene_pcd, torch.Tensor):
            scene_pcd = scene_pcd.cpu().detach().numpy()
        if isinstance(scene_rgb, torch.Tensor):
            scene_rgb = scene_rgb.cpu().detach().numpy()
        
        full_pcd = np.vstack([scene_pcd] + grasp_pcds)
        full_rgb = np.vstack([scene_rgb] + grasp_rgbs)
        
        plot_pcd(full_pcd, full_rgb) #, base_frame=True)
    
    def visualize_gripper_poses(self, gripper_poses):
        """
        Visualize gripper poses using sampled point cloud.
        
        Args:
            gripper_poses: (n, 7) numpy array or torch tensor of gripper poses (x, y, z, qw, qx, qy, qz)
        """
        # Colors for gripper poses (R, G, B) in [0,1]
        gripper_colors = [
            (1.0, 0.0, 0.0),  # red
            (0.0, 1.0, 0.0),  # green
            (0.0, 0.0, 1.0),  # blue
            (1.0, 0.65, 0.0), # orange
            (0.56, 0.0, 1.0), # purple
            (1.0, 0.0, 1.0),  # magenta
            (0.0, 1.0, 1.0),  # cyan
            (1.0, 1.0, 0.0),  # yellow
        ]
        
        # Convert to numpy if torch tensor
        if isinstance(gripper_poses, torch.Tensor):
            gripper_poses = gripper_poses.cpu().detach().numpy()
        
        num_poses = gripper_poses.shape[0]
        
        # Build visualization point clouds
        gripper_pcds = []
        gripper_rgbs = []
        
        for i in range(num_poses):
            gripper_pose = gripper_poses[i, :7]
            
            gripper_transform = pose_to_transformation(gripper_pose, format="wxyz")
            
            # Cycle through colors
            color = gripper_colors[i % len(gripper_colors)]
            
            gripper_gripper_pcd, gripper_gripper_colors = make_gripper_visualization(
                rotation=gripper_transform[:3, :3],
                translation=gripper_transform[:3, 3],
                length=0.05,
                density=50,
                color=color,
            )
            
            gripper_pcds.append(gripper_gripper_pcd)
            gripper_rgbs.append(gripper_gripper_colors)
        
        # Use sampled point cloud instead of regular point cloud
        scene_pcd = self.get_sampled_point_cloud()
        scene_rgb = np.ones_like(scene_pcd) * 0.7  # Grey color for scene points
        
        full_pcd = np.vstack([scene_pcd] + gripper_pcds)
        full_rgb = np.vstack([scene_rgb] + gripper_rgbs)
        
        plot_pcd(full_pcd, full_rgb) #, base_frame=True)
    
    def sample_object_grasped_poses(self, object_name, n_poses=16):
        """
        Sample n_poses random poses for an object within the original table bounds,
        with z between 0.2 and 0.6, and random orientation, then place the object at those poses.
        
        Args:
            object_name: Name of the object to sample poses for
            n_poses: Number of poses to sample (default: 16)
        
        Returns:
            poses: (n_poses, 7) numpy array of poses [x, y, z, qw, qx, qy, qz]
        """

        # Table bounds (from position_object)
        x_min = 0.25
        x_max = 0.7
        y_min = -0.5
        y_max = 0.07
        z_min = 0.2
        z_max = 0.6
        
        # Sample poses
        poses = []
        for _ in range(n_poses):
            # Sample x, y within bounds
            cx = np.random.uniform(x_min, x_max)
            cy = np.random.uniform(y_min, y_max)
            
            # Sample z between 0.2 and 0.6
            z = np.random.uniform(z_min, z_max)
            
            # Sample random orientation (z-axis rotation)
            angle = np.random.uniform(0, 2 * np.pi)
            rotation = R.from_euler('z', [angle])
            quaternion = rotation.as_quat().squeeze()  # Returns [x, y, z, w]
            
            # Convert to SAPIEN format [w, x, y, z] and adjust x by -0.615 (same as position_object)
            pose = np.array([
                cx - 0.615,
                cy,
                z,
                quaternion[3],  # w
                quaternion[0],  # x
                quaternion[1],  # y
                quaternion[2]   # z
            ])
            poses.append(pose)
        
        poses = np.array(poses)  # (n_poses, 7)
        
        # Place the object at these poses (repeat last element if needed for num_envs)
        poses_tensor = torch.tensor(poses, dtype=torch.float32, device=self.device)
        
        return poses_tensor
    
    # ------------------------------------------------------------------------- #

    # ---------------- TASK SPECIFIC PLANNING AND EXECUTION ---------------- #
    
    def test_primitives(self):
        
        grasp_poses, grasp_costs, grasp_object_ids = self.action_sampler.sample_grasp_poses()
        num_grasps = min(self.num_envs, len(grasp_poses))
        
        trajectories, success = self.action_primitives.Pick(
            grasp_poses=grasp_poses[:num_grasps],
            object_indices=grasp_object_ids[:num_grasps].tolist(),
            lift_distances=torch.tensor([POST_GRASP_LIFT for _ in range(num_grasps)], device="cuda:0", dtype=torch.float32)
        )

        # # TODO: Important, all the environments should be in the same state before the execution of a primitive

        # Select first environment with success
        success_indices = np.where(success)[0]
        if len(success_indices) > 0:
            selected_idx = int(success_indices[0])
        else:
            return

        # self.visualize_grasp_poses(grasp_poses[selected_idx].unsqueeze(0))
        
        current_state = self.get_state().clone()
        selected_state = current_state[selected_idx].unsqueeze(0)
        current_state[:] = selected_state.expand(self.num_envs, -1)
        self.set_state(current_state)
        if self.render_mode is not None:
            self.render()

        # -------------------------------------------------------------------------------- #

        self.action_primitives.update_collision_world()

        object_index = grasp_object_ids[selected_idx].item()
        
        shelf_poses, shelf_joints = self.sample_gripper_poses_in_shelf(
            n=100,
            object_name=self.object_names[object_index],
            depth_tolerance=0.05,
            width_tolerance=0.05,
            height_tolerance=0.05
        )

        trajectories, success = self.action_primitives.Place(
            place_poses=shelf_poses[:self.num_envs],
            object_index=object_index
        )

        self.action_primitives.OpenGripper()
        self.wait_for_stability()

        # Find first environment with at least one object in shelf
        object_index = None
        for env_idx in range(self.num_envs):
            num_objects_in_shelf, object_indices_in_shelf = self.number_of_objects_in_shelf(env_idx=env_idx)
            if num_objects_in_shelf > 0:
                object_index = object_indices_in_shelf[0]
                selected_idx = env_idx
                break

        # self.visualize_grasp_poses(shelf_poses[selected_idx].unsqueeze(0))

        current_state = self.get_state().clone()
        selected_state = current_state[selected_idx].unsqueeze(0)
        current_state[:] = selected_state.expand(self.num_envs, -1)
        self.set_state(current_state)
        if self.render_mode is not None:
            self.render()

        self.action_primitives.update_collision_world()

        # self.set_joint_state(self.home_joints)
        if self.render_mode is not None:
            self.render()

        push_start_poses, \
        push_directions, \
        push_distances = self.sample_push_actions(
            object_name=self.object_names[object_index], 
            n=100,
            min_push_distance=0.12,
            max_push_distance=0.22,
            max_offset_angle=0., #np.pi / 12,
            env_idx=0
        )
        
        self.action_primitives.CloseGripper()
        
        trajectories, success = self.action_primitives.Push(
            push_start=push_start_poses[:self.num_envs],
            push_direction=push_directions[:self.num_envs],
            push_distance=push_distances[:self.num_envs],
        )

        success_indices = np.where(success)[0]
        if len(success_indices) > 0:
            selected_idx = int(success_indices[0])
        else:
            return
        
        self.visualize_push_poses(
            push_start_pose = push_start_poses[selected_idx].cpu().numpy(),
            push_direction = push_directions[selected_idx].item(),
            push_distance = push_distances[selected_idx].item()
        )

        return grasp_poses, grasp_costs, grasp_object_ids
    
    def visualize_push_poses(self, push_start_pose, push_direction, push_distance):
        """
        Visualize push poses.
        
        Args:
            push_start_pose: (1, 7) numpy array or torch tensor of push start pose (x, y, z, qw, qx, qy, qz)
            push_direction: (1,) numpy array or torch tensor of push direction (0=X, 1=Y, 2=Z)
            push_distance: (1,) numpy array or torch tensor of push distance (in meters)
        """

        # Move pose along the specified axis
        push_start_pose = push_start_pose.reshape(1, -1)  # (1, 7) for move_pose functions
        if push_direction == 0:  # X axis
            push_end_pose = move_pose_along_local_x(push_start_pose, push_distance, format='wxyz')
        elif push_direction == 1:  # Y axis
            push_end_pose = move_pose_along_local_y(push_start_pose, push_distance, format='wxyz')
        else:  # push_direction == 2, Z axis
            push_end_pose = move_pose_along_local_z(push_start_pose, push_distance, format='wxyz')
        
        # Concatenate start and end poses
        poses_to_visualize = np.concatenate([push_start_pose, push_end_pose], axis=0)  # (2, 7)
        
        # Visualize both poses
        self.visualize_grasp_poses(poses_to_visualize)
    
    def filter_object_poses_in_shelf(self, object_poses):
        """
        Filter object poses to find which ones result in the object being in the shelf.
        
        Args:
            object_poses: (n, 7) torch tensor or numpy array of object poses in world frame
        
        Returns:
            indices: numpy array of indices where object is in shelf
        """
        # Convert to numpy if needed
        if isinstance(object_poses, torch.Tensor):
            object_poses_np = object_poses.cpu().numpy()
        else:
            object_poses_np = object_poses
        
        # Transform object poses to axis-aligned shelf frame
        axis_aligned_obj_points = transform_pcd(object_poses_np[:, :3], self.transform_to_axis_aligned_shelf)
        
        # Check which objects are in shelf
        in_shelf_indices = []
        for i in range(len(axis_aligned_obj_points)):
            if point_in_cuboid(self.shelf_corners, axis_aligned_obj_points[i]):
                in_shelf_indices.append(i)
        
        return np.array(in_shelf_indices)
    
    def collect_placement_data(self, batch_size=8, num_points=4096):
        
        object_name = self.object_names[0]

        grasp_poses = self.grasps[object_name]
        grasp_transformation = pose_to_transformation(grasp_poses, format='wxyz')
        grasp_idx = np.random.randint(0, len(grasp_poses))
        grasp_transformation = grasp_transformation[grasp_idx]
        grasp_transform_inverse = invert_transformation(grasp_transformation)

        gripper_pose = self.action_primitives.fk(self.get_joint_state()[0].to("cuda:0")).cpu().detach().numpy()
        gripper_pose[..., 0] = gripper_pose[..., 0] - 0.615
        gripper_transform = pose_to_transformation(gripper_pose, format='wxyz')
        gripper_transform =  gripper_transform @ grasp_transform_inverse
        attached_object_pose = transformation_to_pose(gripper_transform, format='wxyz')
        attached_object_pose = to_torch_pose(attached_object_pose, device="cuda:0")
        # attached_object_pose[0] = attached_object_pose[0] - 0.615
        
        self.set_object_pose(object_name, attached_object_pose)
        if self.render_mode is not None:
            self.render()

        self.action_primitives.update_collision_world()
        self.action_primitives.attach_object_to_robot(object_name)
        # self.action_primitives.visualize_planner_world()

        # Sample gripper poses
        poses_inside = self.sample_gripper_poses_in_shelf(n=2000)
        poses_outside = self.sample_gripper_poses_outside_shelf(n=2000)
        
        # Compute object poses for each set
        transforms_inside = pose_to_transformation(poses_inside, format='wxyz')
        transforms_outside = pose_to_transformation(poses_outside, format='wxyz')
        
        object_transforms_inside = transforms_inside @ grasp_transform_inverse
        object_transforms_outside = transforms_outside @ grasp_transform_inverse
        
        object_poses_inside = transformation_to_pose(object_transforms_inside, format='wxyz')
        object_poses_outside = transformation_to_pose(object_transforms_outside, format='wxyz')
        
        object_poses_inside = to_torch_pose(object_poses_inside, device="cuda:0")
        object_poses_outside = to_torch_pose(object_poses_outside, device="cuda:0")
        
        # Filter to find which result in object in shelf
        valid_indices_inside = self.filter_object_poses_in_shelf(object_poses_inside)
        valid_indices_outside = self.filter_object_poses_in_shelf(object_poses_outside)

        if len(valid_indices_inside) == 0 and len(valid_indices_outside) == 0:
            print("No valid poses found")
            return False
        
        # Filter gripper poses and object poses using valid_indices
        # Limit to 800 poses max
        n_inside_limit = min(800, len(valid_indices_inside))
        n_outside_limit = min(800, len(valid_indices_outside))
        
        poses_inside = poses_inside[valid_indices_inside[:n_inside_limit]]
        poses_outside = poses_outside[valid_indices_outside[:n_outside_limit]]
        
        object_poses_inside_filtered = object_poses_inside[valid_indices_inside[:n_inside_limit]]
        object_poses_outside_filtered = object_poses_outside[valid_indices_outside[:n_outside_limit]]

        # Convert to torch for curobo operations
        poses_inside_curobo = torch.tensor(poses_inside, device="cuda:0", dtype=torch.float32)
        poses_outside_curobo = torch.tensor(poses_outside, device="cuda:0", dtype=torch.float32)
        poses_inside_curobo[..., 0] = poses_inside_curobo[..., 0] + 0.615
        poses_outside_curobo[..., 0] = poses_outside_curobo[..., 0] + 0.615

        place_joints_inside, feasible_indices_inside = self.action_primitives.inverse_kinematics(poses_inside_curobo)
        place_joints_outside, feasible_indices_outside = self.action_primitives.inverse_kinematics(poses_outside_curobo)

        # Filter using feasible indices
        feasible_mask_inside = feasible_indices_inside[:, 0]
        feasible_mask_outside = feasible_indices_outside[:, 0]
        
        place_joints_inside = place_joints_inside[feasible_mask_inside][:self.num_envs]
        place_joints_outside = place_joints_outside[feasible_mask_outside][:self.num_envs]
        
        poses_inside = poses_inside[feasible_mask_inside.cpu().numpy()][:self.num_envs]
        poses_outside = poses_outside[feasible_mask_outside.cpu().numpy()][:self.num_envs]
        
        object_poses_inside_filtered = object_poses_inside_filtered[feasible_mask_inside][:self.num_envs]
        object_poses_outside_filtered = object_poses_outside_filtered[feasible_mask_outside][:self.num_envs]

        #  ======================== OUTSIDE POSES ======================== #
        self.set_joint_state(place_joints_outside)
        self.set_object_pose(object_name, object_poses_outside_filtered)
        if self.render_mode is not None:
            self.render()
        self.wait(steps=50)

        object_still_in_shelf_indices = []
        for i in range(len(place_joints_outside)):
            if self.check_object_inside_shelf(object_name, env_idx = i) and not self.check_object_within_gripper(object_name, env_idx = i):
                object_still_in_shelf_indices.append(i)

        if len(object_still_in_shelf_indices) == 0:
            print("No objects still in shelf")
            return False

        valid_outside_poses = poses_outside[object_still_in_shelf_indices]
        
        #  ======================== INSIDE POSES ======================== #
        self.set_joint_state(place_joints_inside)
        self.set_object_pose(object_name, object_poses_inside_filtered)
        if self.render_mode is not None:
            self.render()
        self.wait(steps=50)

        object_still_in_shelf_indices = []
        for i in range(len(place_joints_inside)):
            if self.check_object_inside_shelf(object_name, env_idx = i) and not self.check_object_within_gripper(object_name, env_idx = i):
                object_still_in_shelf_indices.append(i)

        if len(object_still_in_shelf_indices) == 0:
            print("No objects still in shelf")
            return False

        valid_inside_poses = poses_inside[object_still_in_shelf_indices]

        num_poses_to_take = min(len(valid_outside_poses), len(valid_inside_poses), batch_size // 2)

        valid_outside_poses = valid_outside_poses[:num_poses_to_take]
        valid_inside_poses = valid_inside_poses[:num_poses_to_take]

        all_target_poses = np.concatenate([valid_outside_poses, valid_inside_poses], axis=0)
        num_poses = len(all_target_poses)

        object_grasped_poses = self.sample_object_grasped_poses(object_name, n_poses=num_poses)
        self.set_object_pose(object_name, object_grasped_poses)
        if self.render_mode is not None:
            self.render()
        
        shelf_pcd = self.get_sampled_actor_point_cloud(self.shelf, num_points = num_points // 2)
        
        # Collect object PCDs for all poses
        object_pcds = []
        for i in range(num_poses):
            object_pcd = self.get_sampled_actor_point_cloud(self.objects[object_name], env_idx = i, num_points = num_points // 2)
            object_pcds.append(object_pcd)
        
        # Reshape shelf_pcd: (num_shelf_points, 3) -> (1, num_shelf_points, 3) -> (num_poses, num_shelf_points, 3)
        shelf_pcd_reshaped = shelf_pcd[np.newaxis, :, :]  # (1, num_shelf_points, 3)
        repeated_shelf_pcd = np.repeat(shelf_pcd_reshaped, num_poses, axis=0)  # (num_poses, num_shelf_points, 3)
        
        # Stack object_pcds: list of (num_object_points, 3) -> (num_poses, num_object_points, 3)
        stacked_object_pcd = np.stack(object_pcds, axis=0)  # (num_poses, num_object_points, 3)
        
        # Concatenate along num_points axis (axis=1): (num_poses, num_shelf_points + num_object_points, 3)
        combined_pcd = np.concatenate([repeated_shelf_pcd, stacked_object_pcd], axis=1)  # (num_poses, num_points, 3)

        object_grasped_transformations = pose_to_transformation(object_grasped_poses.cpu().numpy(), format='wxyz')
        starting_grasp_transformation = object_grasped_transformations @ grasp_transformation[np.newaxis, :, :]
        starting_grasp_poses = transformation_to_pose(starting_grasp_transformation, format='wxyz')

        # OUTPUTS: starting_grasp_poses, combined_pcd, all_target_poses

        return starting_grasp_poses, combined_pcd, all_target_poses
    
    def test_pushing(self):

        push_start_poses, \
        push_directions, \
        push_distances = self.sample_push_actions(
            object_name=self.object_names[0], 
            n=1000,
            min_push_distance=0.12,
            max_push_distance=0.22,
            max_offset_angle=0., #np.pi / 12,
            env_idx=0
        )

        # Sort push_start_poses by z-axis value (min to max)
        z_values = push_start_poses[:, 2]
        sort_indices = torch.argsort(z_values)
        push_start_poses = push_start_poses[sort_indices]
        push_directions = push_directions[sort_indices]
        push_distances = push_distances[sort_indices]
        
        self.action_primitives.CloseGripper()
        
        self.action_primitives.Push(
            push_start=push_start_poses[:self.num_envs],
            push_direction=push_directions[:self.num_envs],
            push_distance=push_distances[:self.num_envs],
        )
        
        grasp_poses = []
        grasp_costs = []
        grasp_object_ids = []
        
        for object_index in range(len(self.object_names)):
            # Get the transformed grasp pose tensors for the specified object:
            object_name = self.object_names[object_index]
            obj_grasp_poses = self.get_object_grasp_poses(
                object_name, 
                env_idx=0,
                grasp_poses=self.grasps[object_name]
            )
            # self.visualize_grasp_poses(grasp_poses)
            grasp_poses.append(torch.tensor(obj_grasp_poses, device="cuda:0", dtype=torch.float32))
            grasp_costs.append(torch.tensor(self.grasp_costs[object_name], device="cuda:0", dtype=torch.float32))
            grasp_object_ids.append(torch.full((obj_grasp_poses.shape[0],), object_index, device="cuda:0", dtype=torch.int64))

        grasp_poses = torch.cat(grasp_poses, dim=0)
        grasp_costs = torch.cat(grasp_costs, dim=0)
        grasp_object_ids = torch.cat(grasp_object_ids, dim=0)

        print(f"Doing IK for grasps")
        curobo_grasp_poses = grasp_poses.clone()
        curobo_grasp_poses[..., 0] = curobo_grasp_poses[..., 0] + 0.615
        grasp_joints, feasible_indices = self.action_primitives.inverse_kinematics(curobo_grasp_poses)

        grasp_poses = grasp_poses[torch.where(feasible_indices[:, 0])[0]]

        push_start_poses = move_pose_along_local_y(grasp_poses[:self.num_envs], 0.15)
        push_start_poses = move_pose_along_local_z(push_start_poses, 0.08)
        push_start_poses = to_torch_pose(push_start_poses)
        push_direction = torch.tensor([1], device="cuda:0", dtype=torch.int64).repeat(self.num_envs)
        push_distance = torch.tensor([-0.2], device="cuda:0", dtype=torch.float32).repeat(self.num_envs)

        self.action_primitives.CloseGripper()
        
        self.action_primitives.Push(
            push_start=push_start_poses,
            push_direction=push_direction,
            push_distance=push_distance,
        )
    
    def generate_plans(self):
        """
        Test the grasps by executing them.
        """
        # Update the motion planner with the current scene:
        self.update_motion_planner()
        self.set_gripper_friction(friction_coefficient=5.0)

        print("Sampling shelf poses")
        shelf_poses, shelf_joints = self.sample_gripper_poses_in_shelf(
            n=self.num_envs,
            depth_tolerance=0.05,
            width_tolerance=0.05,
            height_tolerance=0.05
        )

        available_object_ids = list(np.arange(len(self.object_names)))

        grasp_poses = []    # (N, 7) torch tensor of grasp poses in the world frame.
        placement_poses = []    # (N, 7) torch tensor of placement poses in the world frame.
        grasp_joints = []    # (N, 7) torch tensor of grasp joint positions.
        placement_joints = []    # (N, 7) torch tensor of placement joint positions.
        object_order = []
        object_poses = []

        while len(available_object_ids) > 0:

            object_id, \
            object_pose, \
            grasp_pose, \
            grasp_joint_config, \
            placement_pose, \
            placement_joint_config = self.get_grasp_and_placement_pose(available_object_ids, shelf_poses)

            grasp_poses.append(grasp_pose)
            placement_poses.append(placement_pose)
            grasp_joints.append(grasp_joint_config)
            placement_joints.append(placement_joint_config)
            object_order.append(object_id)
            object_poses.append(object_pose)

            available_object_ids.remove(object_id)

        grasp_poses = torch.stack(grasp_poses, dim=0)
        placement_poses = torch.stack(placement_poses, dim=0)
        grasp_joints = torch.stack(grasp_joints, dim=0)
        placement_joints = torch.stack(placement_joints, dim=0)
        object_poses = torch.stack(object_poses, dim=0)
        object_order = torch.tensor(object_order, device="cuda:0", dtype=torch.int64)

        return grasp_poses, object_poses, placement_poses, grasp_joints, placement_joints, object_order   

    ###########################################################################
    
    # NOTE: This is the latest version of the saving function.
    def save_environment(self, save_dir, env_id: int = 0):
        """
        Save current environment state including object poses, shelf parameters, and URDF files.
        
        Args:
            save_dir: Directory to save environment in
            env_id: Environment ID (integer) to create folder env_{env_id}
        
        Returns:
            env_path: Path to the saved environment folder
        """
        
        # Create environment folder
        env_folder_name = f"env_{env_id}"
        env_path = os.path.join(save_dir, env_folder_name)
        os.makedirs(env_path, exist_ok=True)
        
        print(f"Saving environment to: {env_path}")
        
        # Get current object poses
        object_info = {}
        for i, object_name in enumerate(self.object_names):
            # Get current pose (for all environments, take first one)
            current_pose = self.get_object_pose(object_name)[0].cpu().numpy()  # (7,)
            
            # Get URDF path
            urdf_path = None
            if hasattr(self, 'urdf_paths') and object_name in self.urdf_paths:
                urdf_path = self.urdf_paths[object_name]
            
            # Get bounds, quaternion, and scale_factor
            bounds = self.bounds.get(object_name)
            quat = self.object_quaternions.get(object_name)
            scale_factor = self.object_scale_factors.get(object_name)
            
            # Convert bounds to list of lists with Python floats (bounds is always a list)
            if bounds is not None:
                bounds = [[float(b) for b in bound] for bound in bounds]
            
            # Convert quaternion to list with Python floats
            if quat is not None:
                if isinstance(quat, np.ndarray):
                    quat = quat.tolist()
                else:
                    quat = [float(q) for q in quat]
            
            object_info[f'object_{i}'] = {
                'object_id': str(object_name),
                'pose': current_pose.tolist(),
                'urdf_path': urdf_path,
                'bounds': bounds,
                'quaternion': quat,
                'scale_factor': float(scale_factor) if scale_factor is not None else None
            }
            
            # Copy object URDF to environment folder
            if urdf_path and os.path.exists(urdf_path):
                object_urdf_name = f"object_{i}_{object_name}.urdf"
                dest_urdf_path = os.path.join(env_path, object_urdf_name)
                shutil.copy2(urdf_path, dest_urdf_path)
                object_info[f'object_{i}']['urdf_filename'] = object_urdf_name
            else:
                raise ValueError(f"URDF path for {object_name} is not found")
        
        # Get shelf information
        shelf_pose = self.get_object_pose(self.shelf)[0].cpu().numpy() if hasattr(self, 'shelf') else self.shelf_pose
        
        shelf_info = {
            'width': float(self.shelf_width),
            'height': float(self.shelf_height),
            'depth': float(self.shelf_depth),
            'wall_thickness': float(self.wall_thickness),
            'distance_from_floor': float(self.shelf_distance_from_floor),
            'bottom_wall_height': float(self.bottom_wall_height) if hasattr(self, 'bottom_wall_height') else None,
            'pose': shelf_pose.tolist()
        }
        
        # Copy shelf URDF and all OBJ files to environment folder
        shelf_folder = "assets/shelf"
        shelf_obj_files = ["top_wall.obj", "vertical_wall.obj", "back_wall.obj", "bottom_wall.obj"]
        shelf_urdf_source = os.path.join(shelf_folder, "shelf.urdf")
        
        if os.path.exists(shelf_urdf_source):
            # Copy URDF
            shelf_urdf_dest = os.path.join(env_path, "shelf.urdf")
            shutil.copy2(shelf_urdf_source, shelf_urdf_dest)
            shelf_info['urdf_filename'] = "shelf.urdf"
            
            # Copy all OBJ files
            for obj_file in shelf_obj_files:
                obj_source = os.path.join(shelf_folder, obj_file)
                if os.path.exists(obj_source):
                    obj_dest = os.path.join(env_path, obj_file)
                    shutil.copy2(obj_source, obj_dest)
                else:
                    raise ValueError(f"Shelf OBJ file {obj_file} not found at {obj_source}")

        else:
            raise ValueError(f"Shelf URDF not found at {shelf_urdf_source}")
        
        # Combine all information
        env_data = {
            'shelf': shelf_info,
            'objects': object_info,
        }
        
        # Convert numpy types to Python types for OmegaConf compatibility
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, (np.str_, np.unicode_)):
                return str(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        env_data = convert_numpy_types(env_data)
        
        # Save to YAML file using OmegaConf
        yaml_path = os.path.join(env_path, "environment.yaml")
        OmegaConf.save(env_data, yaml_path)
        
        print(f"Environment saved successfully to {env_path}")
        
        return env_path
    
    def infer_value_function(self, batch_size: int = None):

        # TODO: Environment should already be reset to the desired state before calling this function.
        # TODO: So environment 0 should be the one from which everything is loaded.

        if batch_size is None:
            batch_size = self.num_envs
        if batch_size > self.num_envs:
            raise ValueError(
                f"Batch size ({batch_size}) exceeds num_envs ({self.num_envs}). "
            )
        
        # Get point cloud for each environment
        pcd_list = []
        for env_idx in tqdm(range(batch_size)):
            pcd = self.get_sampled_point_cloud(env_idx=env_idx) #, num_points=4096)
            pcd = downsample_point_cloud(pcd, num_points=4096)
            pcd_list.append(pcd)
        
        input_pcd = torch.tensor(np.stack(pcd_list, axis=0), device="cuda:0", dtype=torch.float32)
        input_pcd[..., 0] = input_pcd[..., 0] + 0.615

        # Get gripper pose for each environment
        joint_states = self.get_joint_state()[:batch_size].to("cuda:0")
        gripper_poses = []
        for env_idx in range(batch_size):
            gripper_pose = self.action_primitives.fk(joint_states[env_idx])
            gripper_poses.append(gripper_pose)
        gripper_pose = torch.stack(gripper_poses, dim=0).unsqueeze(1)  # (batch_size, 1, 7)
        # Add offset to match training data format (dataset adds +0.615)
        # gripper_pose[..., 0] = gripper_pose[..., 0] + 0.615

        vis_gripper_pose = gripper_pose.clone()[0]
        # vis_gripper_pose[..., 0] = vis_gripper_pose[..., 0] - 0.615
        # self.visualize_gripper_poses(vis_gripper_pose)
        scene_pcd = self.get_sampled_point_cloud()
        scene_rgb = np.ones_like(scene_pcd) * 0.7  # Grey color for scene points
        # plot_pcd(scene_pcd, scene_rgb)

        # Add gripper state (8th dimension) - model expects (B, nhist, 8)
        # TODO: Predicting for after pick, gripper is open
        gripper_state = torch.ones((batch_size, 1, 1), device=gripper_pose.device, dtype=gripper_pose.dtype)
        gripper_pose = torch.cat([gripper_pose, gripper_state], dim=-1)  # (batch_size, 1, 8)

        max_batch_size = 4      # Reduce according to available CUDA memory
        all_values = []
        
        with torch.no_grad():
            for i in range(0, batch_size, max_batch_size):
                end_idx = min(i + max_batch_size, batch_size)
                chunk_pcd = input_pcd[i:end_idx]
                chunk_proprioception = gripper_pose[i:end_idx]

                plot_pcd(chunk_pcd[0].cpu().numpy(), base_frame=True)
                
                value_chunk = self.value_model(
                    pcd = chunk_pcd,
                    proprioception = chunk_proprioception,
                    target_value = None
                )
                
                value_chunk = value_chunk.squeeze(0)
                all_values.append(value_chunk)
        
        values = torch.cat(all_values, dim=0)
        print(values[0].item())
        return values
    
    def evaluate(self):
        """
        Placeholder evaluation function.
        Returns a constant success state.
        """
        # TODO: Implement actual success conditions for shelf packing task
        # For now, return constant failure state
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

    def compute_dense_reward(self, obs=None, action=None, info=None):
        """
        Compute dense reward based on volume of objects in shelf.
        Returns the ratio of packed object volume to total object volume for each environment.
        """
        # Initialize reward tensor
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        # Loop over all environments
        for env_idx in range(self.num_envs):
            # Get object indices in shelf for this environment
            _, object_indices_in_shelf = self.number_of_objects_in_shelf(env_idx=env_idx)
            
            # Sum volumes of objects in shelf (returns 0.0 if empty)
            packed_volume = sum(self.object_volumes[self.object_names[i]] for i in object_indices_in_shelf)
            
            # Compute reward as ratio of packed volume to total object volume
            rewards[env_idx] = packed_volume / self.total_volume
        
        return rewards

    def compute_normalized_dense_reward(self, obs=None, action=None, info=None):
        """
        Placeholder normalized dense reward function.
        Returns a constant normalized reward value.
        """
        # TODO: Implement actual normalized reward computation
        # For now, return constant normalized reward
        return torch.ones(self.num_envs, dtype=torch.float32, device=self.device) * 0.5