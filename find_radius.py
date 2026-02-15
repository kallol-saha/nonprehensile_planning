import numpy as np

def compute_obj_radius(file_path):
    vertices = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("v "):  # Vertex line
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))

    if not vertices:
        raise ValueError("No vertices found in the OBJ file.")

    vertices = np.array(vertices)
    
    # Compute bounding box
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    
    # Compute centroid
    centroid = (min_coords + max_coords) / 2
    
    # Compute radius (farthest distance from centroid)
    distances = np.linalg.norm(vertices - centroid, axis=1)
    radius = np.max(distances)
    
    return radius

# Example usage
file_path = "assets/objects/cup.obj"
radius = compute_obj_radius(file_path) 
print("Bounding Sphere Radius:", radius)
radius *= 1.
print("Bounding Sphere Radius:", radius)
