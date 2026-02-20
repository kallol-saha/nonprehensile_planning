import os
import xml.etree.ElementTree as ET
import trimesh
import numpy as np
import pybullet as p
import shapely
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union, voronoi_diagram
from shapely import MultiPoint
import matplotlib.pyplot as plt

GRID_SIZE = 0.001

p.connect(p.DIRECT)

# ------------------ MESH PROCESSING UTILS ------------------ #

def process_mesh_with_vhacd():
    # Load the original mesh
    input_file = "assets/shelf/back_wall_raw.obj"
    output_file = "assets/shelf/back_wall.obj"
    log_file = "vhacd_log.txt"    
    
    p.vhacd(input_file, output_file, log_file)
    
    print(f"Original mesh processed with vhacd")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Log: {log_file}")

def scale_urdf_meshes(urdf_file, scale_factor, output_file):
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    for mesh in root.findall(".//mesh"):
        scale = mesh.get("scale")
        if scale:
            current_scale = [float(x) for x in scale.split()]
            new_scale = [x * scale_factor for x in current_scale]
        else:
            # Default scale is 1,1,1 if not defined
            new_scale = [scale_factor] * 3

        mesh.set("scale", " ".join(f"{x:.6f}" for x in new_scale))

    tree.write(output_file, encoding="utf-8", xml_declaration=True)

# ------------------ CAMERA POSITION SAMPLING UTILS ------------------ #

def compute_ray_box_intersection(
  ray_origin,         # (x, y, z) origin, must be inside box bounds
  azimuth_angle,      # Orientation angle in XY plane in radians
  elevation_angle,    # Elevation angle in radians
  box_min,            # (x_min, y_min, z_min) bounds
  box_max             # (x_max, y_max, z_max) bounds
):

  # Assert that ray_origin is inside box bounds
  assert ray_origin[0] >= box_min[0] and ray_origin[0] <= box_max[0]
  assert ray_origin[1] >= box_min[1] and ray_origin[1] <= box_max[1]
  assert ray_origin[2] >= box_min[2] and ray_origin[2] <= box_max[2]

  # Random point inside cuboid
  p0 = np.array(ray_origin)
  box_min = np.array(box_min)
  box_max = np.array(box_max)

  # Direction vector of ray
  d = np.array([
      np.cos(elevation_angle) * np.cos(azimuth_angle),
      np.cos(elevation_angle) * np.sin(azimuth_angle),
      np.sin(elevation_angle)
  ])

  # 1. Subtacting the max and min corners from the point gets us the vectors towards corners with the point as origin
  # 2. Element-wise dividing these vectors by the direction vector gives us the number of steps the 
  # ray takes along each axis to reach each of the bounds corresponding to the axes.
  tmin = (box_min - p0) / d
  tmax = (box_max - p0) / d

  # Element-wise minimum gives us the number of backward steps, so that is when the ray enters the box.
  # Element-wise maximum gives us the number of forward steps, so that is when the ray exits the box.
  t1 = np.minimum(tmin, tmax)
  t2 = np.maximum(tmin, tmax)

  # Maximum of the backward steps gives us the number of steps the ray takes to enter the box.
  # Minimum of the forward steps gives us the number of steps the ray takes to exit the box.
  t_enter = np.max(t1)
  t_exit = np.min(t2)

  if t_exit < 0 or t_enter > t_exit:
      # The ray is either going backward or the exit point is before the enter point.
      return None
  else:
      # The ray is going forward and the exit point is after the enter point, compute the intersection point.
      intersection = p0 + t_exit * d
      return np.array(intersection)

# ------------------ OBJECT POSITION SAMPLING UTILS ------------------ #

def semicircle(radius, lower_bound=0, upper_bound=np.pi, n_points=200):
    angles = [lower_bound + (upper_bound - lower_bound) * i / (n_points - 1) for i in range(n_points)]
    points = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    points += [(0, 0)]  # close the shape
    return Polygon(points)

def c_shape(r_inner, r_outer, lower_bound=0, upper_bound=np.pi):
    outer = semicircle(r_outer, lower_bound=lower_bound, upper_bound=upper_bound)
    inner = semicircle(r_inner, lower_bound=lower_bound, upper_bound=upper_bound)
    c_region = outer.difference(inner, grid_size=GRID_SIZE)
    
    return c_region

def sample_rectangle_in_cshape(c_region, forbidden, width, height, max_tries=10000):
    minx, miny, maxx, maxy = c_region.bounds
    for _ in range(max_tries):
        cx = np.random.uniform(minx, maxx)
        cy = np.random.uniform(miny, maxy)
        angle = np.random.uniform(0, 2 * np.pi)
        
        rect = box(-width/2, -height/2, width/2, height/2)
        rect = rotate(rect, angle, use_radians=True)
        rect = translate(rect, cx, cy)
        
        # Check: fully inside C and no overlap with forbidden region
        if rect.within(c_region) and rect.intersects(forbidden) is False:
            return rect, cx, cy, angle
    return None  # no valid sample found

def sample_rectangle_in_axis_aligned_rect(rect_region, forbidden, width, height, max_tries=10000):
    """
    Sample an axis-aligned rectangle within an axis-aligned bounding rectangle.
    
    Args:
        rect_region: Axis-aligned rectangle (shapely Polygon)
        forbidden: Forbidden regions to avoid
        width: Width of the sampled rectangle
        height: Height of the sampled rectangle
        max_tries: Maximum number of sampling attempts
    
    Returns:
        (rect, cx, cy, angle) where angle is always 0 for axis-aligned rectangles
    """
    minx, miny, maxx, maxy = rect_region.bounds
    
    # Check if the rectangle can fit within the bounds
    if (maxx - minx) < width or (maxy - miny) < height:
        return None  # Rectangle is too large to fit
    
    for _ in range(max_tries):
        # Sample center position, ensuring the rectangle fits within bounds
        cx = np.random.uniform(minx + width/2, maxx - width/2)
        cy = np.random.uniform(miny + height/2, maxy - height/2)
        
        # Create axis-aligned rectangle (no rotation)
        rect = box(cx - width/2, cy - height/2, cx + width/2, cy + height/2)
        
        # Check: fully inside bounding rectangle and no overlap with forbidden region
        if rect.within(rect_region) and (forbidden.is_empty or rect.intersects(forbidden) is False):
            return rect, cx, cy, 0.0
    return None  # no valid sample found

def sample_object_pose_on_shelf(
    fixed_rect_center, # (x, y) tuple center of the fixed rectangle
    fixed_rect_dimensions, # (width, height) tuple dimensions of the fixed rectangle
    fixed_rect_angle, # Orientation angle in radians of the fixed rectangle
    sampled_rect_dimensions, # (width, height) tuple dimensions of the sampled rectangle
    max_tries=10000
    ):
    """
    Samples a pose for an object that is inside the fixed rectangle.
    
    Args:
        fixed_rect_center: (x, y) tuple center of the fixed rectangle
        fixed_rect_dimensions: (width, height) tuple dimensions of the fixed rectangle
        fixed_rect_angle: Orientation angle in radians of the fixed rectangle
        sampled_rect_dimensions: (width, height) tuple dimensions of the sampled rectangle
    """
    fixed_rect = box(-fixed_rect_dimensions[0]/2, -fixed_rect_dimensions[1]/2, fixed_rect_dimensions[0]/2, fixed_rect_dimensions[1]/2)
    fixed_rect = rotate(fixed_rect, fixed_rect_angle, use_radians=True)
    fixed_rect = translate(fixed_rect, fixed_rect_center[0], fixed_rect_center[1])

    minx, miny, maxx, maxy = fixed_rect.bounds
    for _ in range(max_tries):
        cx = np.random.uniform(minx, maxx)
        cy = np.random.uniform(miny, maxy)
        angle = np.random.uniform(0, 2 * np.pi)
        
        rect = box(-sampled_rect_dimensions[0]/2, -sampled_rect_dimensions[1]/2, sampled_rect_dimensions[0]/2, sampled_rect_dimensions[1]/2)
        rect = rotate(rect, angle, use_radians=True)
        rect = translate(rect, cx, cy)
        
        # Check: fully inside C and no overlap with forbidden region
        if rect.within(fixed_rect):

            # fig, ax = plt.subplots()
            # for geom, color in [(fixed_rect, 'blue'), (rect, 'green')]:
            #     if geom:
            #         xs, ys = geom.exterior.xy
            #         ax.fill(xs, ys, alpha=0.5, fc=color, ec='black')

            # ax.set_aspect('equal')
            # ax.set_xlim(0, 1)
            # ax.set_ylim(-1, 1)
            # plt.show()

            return cx, cy, angle

    print("Object is too large to fit inside the fixed rectangle")
    return None  # no valid sample found

def sample_point_in_fixed_rectangle(
    fixed_rect_center,
    fixed_rect_dimensions,
    fixed_rect_angle,
    num_points=1,
):
    """
    Uniformly sample point(s) (x, y) inside a rotated rectangle.

    Args:
        fixed_rect_center: (x, y) center of the rectangle
        fixed_rect_dimensions: (width, height) of the rectangle
        fixed_rect_angle: rotation of the rectangle in radians
        num_points: number of points to sample (default: 1)

    Returns:
        If num_points=1: (x, y) tuple
        If num_points>1: (N, 2) array of (x, y) coordinates
    """
    half_w = fixed_rect_dimensions[0] * 0.5
    half_h = fixed_rect_dimensions[1] * 0.5

    # Sample in the rectangle's local frame
    dx = np.random.uniform(-half_w, half_w, size=num_points)
    dy = np.random.uniform(-half_h, half_h, size=num_points)

    # Rotate by fixed_rect_angle and translate to world frame
    c = np.cos(fixed_rect_angle)
    s = np.sin(fixed_rect_angle)
    x = fixed_rect_center[0] + c * dx - s * dy
    y = fixed_rect_center[1] + s * dx + c * dy

    if num_points == 1:
        return (x[0], y[0])
    else:
      return np.column_stack([x, y])

def sample_point_in_fixed_rectangle_uniformly(
    fixed_rect_center,
    fixed_rect_dimensions,
    fixed_rect_angle,
    grid_size = 10,
):
    """
    Uniformly sample point(s) (x, y) inside a rotated rectangle.

    Args:
        fixed_rect_center: (x, y) center of the rectangle
        fixed_rect_dimensions: (width, height) of the rectangle
        fixed_rect_angle: rotation of the rectangle in radians
        grid_size: size of the grid to sample points in (number of cells (if squared) to divide the rectangle into)

    Returns:
        If num_points=1: (x, y) tuple
        If num_points>1: (N, 2) array of (x, y) coordinates
    """
    half_w = fixed_rect_dimensions[0] * 0.5
    half_h = fixed_rect_dimensions[1] * 0.5

    # Sample in the rectangle's local frame
    dx = np.linspace(-half_w, half_w, grid_size)
    dy = np.linspace(-half_h, half_h, grid_size)

    x, y = np.meshgrid(dx, dy)

    # Rotate by fixed_rect_angle and translate to world frame
    c = np.cos(fixed_rect_angle)
    s = np.sin(fixed_rect_angle)
    x = fixed_rect_center[0] + c * x - s * y
    y = fixed_rect_center[1] + s * x + c * y

    x = x.flatten()
    y = y.flatten()

    return np.column_stack([x, y])

def sample_object_pose_on_table(
    r_min, # Minimum radius of c-space semicircle
    r_max, # Maximum radius of c-space semicircle
    lower_angle, # Lower angle of c-space semicircle
    upper_angle, # Upper angle of c-space semicircle
    fixed_rect_center, # (x, y) tuple center of the fixed rectangle
    fixed_rect_dimensions, # (width, height) tuple dimensions of the fixed rectangle
    fixed_rect_angle, # Orientation angle in radians of the fixed rectangle
    sampled_rect_dimensions # (width, height) tuple dimensions of the sampled rectangle
    ):
    """
    Samples a pose for an object in the c-space that does not overlap with the fixed rectangle.
    
    Args:
        r_min: Minimum radius of c-space semicircle
        r_max: Maximum radius of c-space semicircle
        lower_angle: Lower angle of c-space semicircle
        upper_angle: Upper angle of c-space semicircle
        fixed_rect_center: (x, y) tuple center of the fixed rectangle
        fixed_rect_dimensions: (width, height) tuple dimensions of the fixed rectangle
        fixed_rect_angle: Orientation angle in radians of the fixed rectangle
    
    Returns:
        pose: (x, y, theta) 2D pose of the object

    """
    
    fixed_rect = box(-fixed_rect_dimensions[0]/2, -fixed_rect_dimensions[1]/2, fixed_rect_dimensions[0]/2, fixed_rect_dimensions[1]/2)
    fixed_rect = rotate(fixed_rect, fixed_rect_angle, use_radians=True)
    fixed_rect = translate(fixed_rect, fixed_rect_center[0], fixed_rect_center[1])

    c_region = c_shape(r_min, r_max, lower_bound=lower_angle, upper_bound=upper_angle)

    forbidden = fixed_rect.intersection(c_region, grid_size=GRID_SIZE)

    rect_sampled, cx, cy, angle = sample_rectangle_in_cshape(c_region, forbidden, width=sampled_rect_dimensions[0], height=sampled_rect_dimensions[1])

    # fig, ax = plt.subplots()
    # for geom, color in [(c_region, 'lightgray'), (fixed_rect, 'blue'), (rect_sampled, 'green')]:
    #     if geom:
    #         xs, ys = geom.exterior.xy
    #         ax.fill(xs, ys, alpha=0.5, fc=color, ec='black')

    # ax.set_aspect('equal')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(-1, 1)
    # plt.show()

    return cx, cy, angle


def sample_object_pose_on_table_multi_object(
    x_min, # Minimum X bound of the sampling rectangle
    x_max, # Maximum X bound of the sampling rectangle
    y_min, # Minimum Y bound of the sampling rectangle
    y_max, # Maximum Y bound of the sampling rectangle
    fixed_rect_centers, # List of (x, y) tuple centers of the fixed rectangles
    fixed_rect_dimensions, # List of (width, height) tuple dimensions of the fixed rectangles
    fixed_rect_angles, # List of orientation angles in radians of the fixed rectangles
    sampled_rect_dimensions # (width, height) tuple dimensions of the sampled rectangle
    ):
    """
    Samples a pose for an object in an axis-aligned rectangle that does not overlap with any of the fixed rectangles.
    
    Args:
        x_min: Minimum X bound of the sampling rectangle
        x_max: Maximum X bound of the sampling rectangle
        y_min: Minimum Y bound of the sampling rectangle
        y_max: Maximum Y bound of the sampling rectangle
        fixed_rect_centers: List of (x, y) tuple centers of the fixed rectangles
        fixed_rect_dimensions: List of (width, height) tuple dimensions of the fixed rectangles
        fixed_rect_angles: List of orientation angles in radians of the fixed rectangles
        sampled_rect_dimensions: (width, height) tuple dimensions of the sampled rectangle
    
    Returns:
        pose: (x, y, theta) 2D pose of the object, where theta is always 0 (axis-aligned)

    """
    
    # Create axis-aligned bounding rectangle
    rect_region = box(x_min, y_min, x_max, y_max)
    
    # Create all fixed rectangles and compute their intersections with rect_region
    forbidden_regions = []
    for center, dimensions, angle in zip(fixed_rect_centers, fixed_rect_dimensions, fixed_rect_angles):
        fixed_rect = box(-dimensions[0]/2, -dimensions[1]/2, dimensions[0]/2, dimensions[1]/2)
        fixed_rect = rotate(fixed_rect, angle, use_radians=True)
        fixed_rect = translate(fixed_rect, center[0], center[1])
        
        # Get intersection of this fixed rectangle with rect_region
        forbidden_region = fixed_rect.intersection(rect_region, grid_size=GRID_SIZE)
        if forbidden_region and not forbidden_region.is_empty:
            forbidden_regions.append(forbidden_region)
    
    # Combine all forbidden regions into a single geometry
    if forbidden_regions:
        forbidden = unary_union(forbidden_regions)
    else:
        # If no forbidden regions, create an empty polygon
        forbidden = Polygon()

    result = sample_rectangle_in_axis_aligned_rect(rect_region, forbidden, width=sampled_rect_dimensions[0], height=sampled_rect_dimensions[1])
    if result is None:
        print("Failed to sample rectangle within bounds")
        return None
    
    rect_sampled, cx, cy, angle = result

    # visualize_regions([(rect_region, 'lightgray'), (forbidden, 'red'), (rect_sampled, 'green')])

    return cx, cy, angle

def visualize_regions(regions):
  """
  Visualize a list of regions.
  Args:
    regions: List of regions to visualize in the format [(region, color)]
  """
  fig, ax = plt.subplots()
  for geom, color in regions:
      if geom:
          # Handle MultiPolygon by iterating over constituent polygons
          if isinstance(geom, MultiPolygon):
              for poly in geom.geoms:
                  xs, ys = poly.exterior.xy
                  ax.fill(xs, ys, alpha=0.5, fc=color, ec='black')
          elif isinstance(geom, Polygon):
              xs, ys = geom.exterior.xy
              ax.fill(xs, ys, alpha=0.5, fc=color, ec='black')
          else:
              # For other geometry types, try to get exterior if available
              if hasattr(geom, 'exterior'):
                  xs, ys = geom.exterior.xy
                  ax.fill(xs, ys, alpha=0.5, fc=color, ec='black')
  ax.set_aspect('equal')
  ax.set_xlim(0, 1)
  ax.set_ylim(-1, 1)
  plt.show()


# ------------------ VORONOI GENERATION UTILS ------------------ #

def generate_voronoi_meshes(
    num_points: int = 5,
    side_length: float = 0.2,
    scale_factor: float = 0.9,
    extrusion_height: float = 0.03,
    seed: int = 0,
):
    """
    Generate 3D Voronoi polygon meshes by creating a 2D Voronoi diagram,
    scaling down each cell, and extruding to 3D.

    Returns:
        meshes: list of trimesh.Trimesh objects
        centroids: list of (x, y) centroid positions for each polygon
    """
    rng = np.random.RandomState(seed)
    sampled_points = rng.rand(num_points, 2) * side_length - side_length / 2
    points = MultiPoint(sampled_points)
    vor = voronoi_diagram(points)

    env_boundary = Polygon([
        (-side_length / 2, -side_length / 2),
        (-side_length / 2, side_length / 2),
        (side_length / 2, side_length / 2),
        (side_length / 2, -side_length / 2),
    ])

    meshes = []
    centroids = []

    for poly in vor.geoms:
        clipped = poly.intersection(env_boundary)
        centroid = clipped.centroid
        scaled = shapely.affinity.scale(clipped, xfact=scale_factor, yfact=scale_factor, origin='centroid')
        # Translate polygon so its centroid is at the origin before extruding,
        # so that the mesh is centered at (0,0) in its local frame.
        centered = shapely.affinity.translate(scaled, xoff=-centroid.x, yoff=-centroid.y)
        mesh = trimesh.creation.extrude_polygon(centered, height=extrusion_height)
        meshes.append(mesh)
        centroids.append((centroid.x, centroid.y))

      
    # plot_voronoi_meshes(meshes, centroids)

    return meshes, centroids


def plot_voronoi_meshes(meshes, centroids):
    """Plot the 2D cross-sections of Voronoi meshes with their centroids."""
    fig, ax = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 1, len(meshes)))

    for i, (mesh, (cx, cy)) in enumerate(zip(meshes, centroids)):
        # Project 3D mesh to 2D by taking the XY outline
        vertices = mesh.vertices[:, :2]
        hull = Polygon(vertices).convex_hull
        xs, ys = hull.exterior.xy
        ax.fill(xs, ys, alpha=0.5, fc=colors[i], ec='black', label=f'piece {i}')
        ax.plot(cx, cy, 'k+', markersize=10, markeredgewidth=2)
        ax.annotate(str(i), (cx, cy), fontsize=8, ha='center', va='bottom')

    ax.set_aspect('equal')
    ax.set_title(f'{len(meshes)} Voronoi pieces')
    ax.legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    plt.show()


def _save_voronoi_urdf(obj_filename: str, centroid, color, urdf_path: str):
    """Generate a URDF file referencing an OBJ mesh file."""
    urdf_content = f'''<?xml version="1.0"?>
<robot name="convex_mesh">
<link name="base_link">
    <visual>
        <geometry>
            <mesh filename="{obj_filename}" scale="1 1 1"/>
        </geometry>
        <material name="custom_color">
            <color rgba="{color[0]} {color[1]} {color[2]} 1.0"/>
        </material>
    </visual>
    <collision>
        <geometry>
            <mesh filename="{obj_filename}" scale="1 1 1"/>
        </geometry>
    </collision>
    <inertial>
        <mass value="0.1"/>
        <origin xyz="{centroid[0]} {centroid[1]} {centroid[2]}"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0"
                iyy="0.001" iyz="0.0"
                izz="0.001"/>
    </inertial>
</link>
</robot>'''
    with open(urdf_path, "w") as f:
        f.write(urdf_content)


def save_voronoi_assets(meshes, output_dir: str, seed: int = 0):
    """
    Export meshes to OBJ, run VHACD convex decomposition, and generate URDFs.

    Args:
        meshes: list of trimesh.Trimesh objects
        output_dir: directory to write OBJ and URDF files
        seed: random seed for colors

    Returns:
        urdf_paths: list of absolute paths to generated URDF files
    """
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(seed + 42)
    colors = rng.rand(len(meshes), 3)

    # Connect pybullet in DIRECT mode for VHACD
    physics_client = p.connect(p.DIRECT)

    urdf_paths = []
    for i, mesh in enumerate(meshes):
        obj_filename = f"polygon_{i}.obj"
        urdf_filename = f"polygon_{i}.urdf"
        obj_path = os.path.join(output_dir, obj_filename)
        urdf_path = os.path.join(output_dir, urdf_filename)

        mesh.export(obj_path)
        p.vhacd(obj_path, obj_path, os.path.join(output_dir, "vhacd_log.txt"),
                physicsClientId=physics_client)

        _save_voronoi_urdf(obj_filename, mesh.centroid, colors[i], urdf_path)
        urdf_paths.append(os.path.abspath(urdf_path))

    p.disconnect(physics_client)
    return urdf_paths

