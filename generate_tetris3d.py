
import os
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import shapely
from shapely.ops import voronoi_diagram
from shapely import MultiPoint, Polygon, envelope, voronoi_polygons, normalize
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import trimesh
import pybullet as p

p.connect(p.DIRECT)

side_length = 0.2  # Found this through simulation
np.random.seed(0)

def scale_polygon(vertices, scale):

    # Compute centroid
    xs, ys = zip(*vertices)
    center = (sum(xs) / len(xs), sum(ys) / len(ys))
    
    cx, cy = center
    scaled_vertices = []
    for x, y in vertices:
        x_new = cx + scale * (x - cx)
        y_new = cy + scale * (y - cy)
        scaled_vertices.append((x_new, y_new))
    
    return scaled_vertices

def get_random_voronoi_polygons(num_points: int, side_length: float) -> list:
    """
    Returns: a list of (n, 2) representing the vertices of the Voronoi polygons
    """    

    # Assume square is centered at the origin

    # Generate random points inside the square
    sampled_points = np.random.rand(num_points, 2) * side_length - side_length / 2
    points = MultiPoint(sampled_points)
    vor = voronoi_diagram(points)

    # Define the bounding polygon:
    env = Polygon([(-side_length / 2, -side_length / 2), (-side_length / 2, side_length / 2), 
                   (side_length / 2, side_length / 2), (side_length / 2, -side_length / 2)])
    
    polygons = []

    for poly in vor.geoms:

        intersected_poly = poly.intersection(env)

        poly_pts = list(intersected_poly.exterior.coords)
        poly_pts = np.array(poly_pts)

        polygons.append(poly_pts)

    return polygons, sampled_points

def get_random_voronoi_polygon_meshes(num_points: int, side_length: float) -> list:
    """
    Returns: a list of (n, 2) representing the vertices of the Voronoi polygons
    """    

    # Assume square is centered at the origin

    # Generate random points inside the square
    sampled_points = np.random.rand(num_points, 2) * side_length - side_length / 2
    points = MultiPoint(sampled_points)
    vor = voronoi_diagram(points)

    # Define the bounding polygon:
    env = Polygon([(-side_length / 2, -side_length / 2), (-side_length / 2, side_length / 2), 
                   (side_length / 2, side_length / 2), (side_length / 2, -side_length / 2)])
    
    meshes = []

    for poly in vor.geoms:

        intersected_poly = poly.intersection(env)

        # Scale down the intersected polygon
        intersected_poly = shapely.affinity.scale(intersected_poly, xfact=0.9, yfact=0.9, origin='centroid')

        mesh = trimesh.creation.extrude_polygon(intersected_poly, height = 0.03)        # TODO: Remove this hardcoding

        meshes.append(mesh)

    return meshes

def plot_voronoi_polygons(polygons: list, points: np.ndarray):
    """
    Plots the Voronoi polygons
    """

    # Scatter plot of the points
    plt.scatter(points[:, 0], points[:, 1], color='blue')

    for poly_pts in polygons:
        plt.plot(poly_pts[:, 0], poly_pts[:, 1], color='black')
        plt.scatter(poly_pts[:, 0], poly_pts[:, 1], color='red')

    plt.show()

def get_3d_polygons(polygons: list, lower_height: float, upper_height: float) -> list:
    """
    Returns the 3D polygons
    """

    polygons_3d = []

    for poly_pts in polygons:

        # Remove the last point which is the same as the first point
        poly_pts = poly_pts[:-1]

        # Create the 3D polygon
        poly_pts_3d = np.zeros((poly_pts.shape[0] * 2, 3))

        # Lower face
        poly_pts_3d[:poly_pts.shape[0], :2] = poly_pts
        poly_pts_3d[:poly_pts.shape[0], 2] = lower_height

        # Upper face
        poly_pts_3d[poly_pts.shape[0]:, :2] = poly_pts
        poly_pts_3d[poly_pts.shape[0]:, 2] = upper_height

        polygons_3d.append(poly_pts_3d)

    polygons_3d_faces = []

    for poly_pts_3d in polygons_3d:

        faces_array = []

        # Lower face
        faces_array.append(list(range(poly_pts_3d.shape[0] // 2)))

        # Upper face
        faces_array.append(list(range(poly_pts_3d.shape[0] // 2, poly_pts_3d.shape[0])))

        # Side faces:
        for i in range(poly_pts_3d.shape[0] // 2):
            faces_array.append([i, (i + 1) % (poly_pts_3d.shape[0] // 2), (i + 1) % (poly_pts_3d.shape[0] // 2) + poly_pts_3d.shape[0] // 2, i + poly_pts_3d.shape[0] // 2])

        polygons_3d_faces.append(faces_array)

    return polygons_3d, polygons_3d_faces

def save_obj(vertices: np.ndarray, faces: list, filename="polygon.obj"):
    """
    Saves a 3D convex polyhedron as an OBJ file.
    
    :param vertices: (nx3) ndarray representing the vertices.
    :param filename: Output OBJ file name.
    """
    if vertices.shape[1] != 3:
        raise ValueError("Input vertices must be an (nx3) ndarray.")

    # Compute the convex hull
    hull = ConvexHull(vertices)
    faces = hull.simplices  # Indices of the faces

    with open(filename, "w") as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write faces
        for face in faces:
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")

def save_urdf(obj_filename, centroid, color, urdf_filename="convex_mesh.urdf"):
    """
    Generates a URDF file that references the given OBJ file.

    :param obj_filename: Name of the OBJ file to be used.
    :param centroid: A numpy array of length 3 representing the centroid of the object.
    :param color: A numpy array of length 3 representing the RGB color of the object.
    :param urdf_filename: Name of the output URDF file.
    """
    urdf_content = f'''<?xml version="1.0"?>
    <robot name="convex_mesh">
    <link name="base_link">
        <!-- Visual Representation -->
        <visual>
            <geometry>
                <mesh filename="{obj_filename}" scale="1 1 1"/>
            </geometry>
            <material name="custom_color">
                <color rgba="{color[0]} {color[1]} {color[2]} 1.0"/>
            </material>
        </visual>

        <!-- Collision Geometry -->
        <collision>
            <geometry>
                <mesh filename="{obj_filename}" scale="1 1 1"/>
            </geometry>
        </collision>

        <!-- Inertial Properties -->
        <inertial>
            <mass value="1.0"/>
            <origin xyz="{centroid[0]} {centroid[1]} {centroid[2]}"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0"
                    iyy="0.01" iyz="0.0"
                    izz="0.01"/>
        </inertial>
    </link>
    </robot>'''

    # Save to file
    with open(urdf_filename, "w") as f:
        f.write(urdf_content)

    print(f"URDF saved as {urdf_filename}")

# polygons, points = get_random_voronoi_polygons(num_points=5, side_length=0.2)
polygon_meshes = get_random_voronoi_polygon_meshes(num_points=5, side_length=0.2)
# plot_voronoi_polygons(polygons, points)

colors = np.random.rand(len(polygon_meshes), 3)

for i in range(len(polygon_meshes)):
    obj_filename = f"polygon_{i}.obj"
    urdf_filename = f"polygon_{i}.urdf"
    file_path = os.path.join("assets/objects", obj_filename)
    urdf_file_path = os.path.join("assets/objects", urdf_filename)
    
    # Save the mesh as an OBJ file
    polygon_meshes[i].export(file_path)

    name_log = "log.txt"
    p.vhacd(file_path, file_path, name_log)

    # Save the URDF file referencing the OBJ file
    save_urdf(obj_filename, polygon_meshes[i].centroid, colors[i], urdf_filename=urdf_file_path)

# polygons_3d, polygon_faces_3d = get_3d_polygons(polygons, lower_height=0, upper_height=0.05)

# for i, (polygon, face) in enumerate(zip(polygons_3d, polygon_faces_3d)):
#     obj_filename = f"polygon_{i}.obj"
#     urdf_filename = f"polygon_{i}.urdf"
#     file_path = os.path.join("assets/objects", obj_filename)
#     urdf_file_path = os.path.join("assets/objects", urdf_filename)
#     save_obj(polygon, face, filename=file_path)
#     save_urdf(obj_filename, urdf_filename=urdf_file_path)

