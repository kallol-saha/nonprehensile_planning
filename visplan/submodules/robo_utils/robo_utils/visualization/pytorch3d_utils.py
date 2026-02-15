import torch
import numpy as np
import imageio

from pytorch3d.structures import Pointclouds
from pytorch3d import structures

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    PointLights,
    TexturesVertex,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    RasterizationSettings,
)

def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def get_points_renderer(image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def capture_image(structures, renderer, device, R = None, T = None, fov = 60, lights = None):

    if (R is None) or (T is None):
        R, T = look_at_view_transform(dist=3., elev=5, azim=0)
    
    # Prepare the camera:
    cameras = FoVPerspectiveCameras(
        R=R, T=T, fov=fov, device=device
    )

    # Place a point light in front of the cow.
    if lights is None:
        lights = PointLights(location=[[0, 0, -3]], device=device)

    if lights is not False:
        image = renderer(structures, cameras=cameras, lights=lights)
    else:
        image = renderer(structures, cameras=cameras)
    image = image.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

    image_uint8 = (image * 255).astype(np.uint8)

    return image_uint8

def render_360_gif(structures: structures, 
                   output_file: str, 
                   image_size: int = 512,
                   light_location: list[float] = [0, 0, -3],
                   fps: int = 30, 
                   dist: float = 3, 
                   elev: float = 5,
                   point_radius: float = 0.01,
                   center_of_rotation = None,   
                   azimuth_range: tuple = (0, 360, 5),
                   up_vector: list[float] = [0, 0, 1],
                   fov: float = 60):
    """
    Render a 360-degree GIF with customizable camera parameters.
    
    Args:
        structures: Pytorch3D structures to render
        output_file: Output GIF file path
        image_size: Rendered image size
        light_location: Light position [x, y, z]
        fps: Frames per second
        dist: Camera distance from center
        elev: Fixed elevation angle
        point_radius: Point cloud point radius
        center_of_rotation: Center point for rotation [x, y, z] (if None, uses scene centroid)
        azimuth_range: (start, end, step) in degrees
        up_vector: The rotation axis direction [x, y, z]
    """
    
    device = structures.device

    # Calculate center of rotation
    if center_of_rotation is None:
        pcd = structures.get_cloud(0)[0]
        center_of_rotation = pcd.mean(dim=0).cpu().numpy().tolist()
    
    # Normalize up_vector to get rotation axis
    up_vector = np.array(up_vector, dtype=np.float32)
    up_vector = up_vector / np.linalg.norm(up_vector)
    
    # Create rotation matrix to align up_vector with Y-axis (standard rotation axis)
    # We want to rotate the scene so that up_vector becomes [0, 1, 0]
    y_axis = np.array([0, 1, 0], dtype=np.float32)
    
    # If up_vector is already aligned with Y-axis, no rotation needed
    if np.allclose(up_vector, y_axis) or np.allclose(up_vector, -y_axis):
        rotation_matrix = np.eye(3)
    else:
        # Calculate rotation axis and angle
        rotation_axis = np.cross(up_vector, y_axis)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Calculate rotation angle
        cos_angle = np.dot(up_vector, y_axis)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Create rotation matrix using Rodrigues' rotation formula
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                     [rotation_axis[2], 0, -rotation_axis[0]],
                     [-rotation_axis[1], rotation_axis[0], 0]])
        rotation_matrix = (np.eye(3) + 
                          np.sin(angle) * K + 
                          (1 - np.cos(angle)) * np.dot(K, K))
    
    # Transform the point cloud
    points = structures.get_cloud(0)[0]
    center_tensor = torch.tensor(center_of_rotation, device=device, dtype=torch.float32)
    
    # Center the points, rotate, then translate back
    points_centered = points - center_tensor
    rotation_tensor = torch.tensor(rotation_matrix, device=device, dtype=torch.float32)
    points_transformed = torch.matmul(points_centered, rotation_tensor.T) + center_tensor
    
    # Create transformed point cloud
    features = structures.features_list()[0]
    transformed_structures = Pointclouds(
        points=[points_transformed],
        features=[features]
    ).to(device)

    # Calculate azimuth angles
    start_azim, end_azim, step_azim = azimuth_range
    azims = np.arange(start_azim, end_azim + step_azim, step_azim)
    frames = len(azims)

    lights = PointLights(location=[light_location], device=device)
    renderer = get_points_renderer(image_size=image_size, radius=point_radius, device=device)
    
    images = []

    print(f"Rendering {frames} frames with azimuth range: {start_azim}° to {end_azim}° (step: {step_azim}°)")
    print(f"Camera distance: {dist}, Elevation: {elev}°")
    print(f"Center of rotation: {center_of_rotation}")
    print(f"Rotation axis (up_vector): {up_vector}")
    
    for azim in azims:
        R, T = look_at_view_transform(
            dist=dist, 
            elev=elev, 
            azim=azim,
            at=(center_of_rotation, ),
            up=((0, 1, 0), )  # Always use Y-axis as up since we transformed the scene
        )
        R = R.to(device)
        T = T.to(device)

        image = capture_image(transformed_structures, renderer, device, R, T, fov=fov, lights=lights)
        images.append(image)

    imageio.mimsave(output_file, images, duration=frames // fps)
    print(f"Saved GIF to: {output_file}")


