#!/usr/bin/env python3
"""
Script to sample points from OBJ mesh files in the assets/objects directory.

This script loads any OBJ mesh file and samples a specified number of points
from the mesh surface using trimesh's built-in sampling functionality.
The sampled points can be saved in various formats (PLY, TXT, NPZ).
"""

import os
import argparse
import glob
import numpy as np
import trimesh
from pathlib import Path
import open3d as o3d
from sklearn.neighbors import BallTree, KDTree
from scipy.spatial import cKDTree
import time

from visplan.submodules.robo_utils.robo_utils.visualization.plotting import plot_pcd
from visplan.submodules.robo_utils.robo_utils.visualization.point_cloud_structures import make_line


def sample_point_pairs_within_distance(points, min_distance, max_pairs=None, method='ball_tree'):
    """
    Efficiently sample pairs of points that are within a minimum distance of each other.
    
    Args:
        points (np.ndarray): Array of points with shape (N, 3)
        min_distance (float): Minimum distance between points in a pair
        max_pairs (int, optional): Maximum number of pairs to return. If None, returns all pairs.
        method (str): Method to use - 'ball_tree', 'kdtree', or 'open3d'
    
    Returns:
        tuple: (pairs, distances) where pairs is (M, 2) array of indices and distances is (M,) array
    """
    print(f"Finding point pairs within distance {min_distance} using {method} method...")
    print(f"Input points shape: {points.shape}")
    
    start_time = time.time()
    
    if method == 'ball_tree':
        pairs, distances = _sample_pairs_ball_tree(points, min_distance, max_pairs)
    elif method == 'kdtree':
        pairs, distances = _sample_pairs_kdtree(points, min_distance, max_pairs)
    elif method == 'open3d':
        pairs, distances = _sample_pairs_open3d(points, min_distance, max_pairs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed_time = time.time() - start_time
    print(f"Found {len(pairs)} pairs within distance {min_distance} in {elapsed_time:.3f} seconds")
    
    return pairs, distances


def _sample_pairs_ball_tree(points, min_distance, max_pairs=None):
    """Use scikit-learn BallTree for radius-based neighbor queries."""
    tree = BallTree(points, metric='euclidean')
    
    # Find all neighbors within min_distance for each point
    neighbor_indices = tree.query_radius(points, r=min_distance)
    
    pairs = []
    distances = []
    
    for i, neighbors in enumerate(neighbor_indices):
        # Remove self-neighbor and only consider neighbors with index > i to avoid duplicates
        valid_neighbors = neighbors[neighbors > i]
        
        if len(valid_neighbors) > 0:
            # Calculate distances to valid neighbors
            neighbor_points = points[valid_neighbors]
            point_distances = np.linalg.norm(neighbor_points - points[i], axis=1)
            
            # Add pairs
            for j, dist in zip(valid_neighbors, point_distances):
                pairs.append([i, j])
                distances.append(dist)
    
    pairs = np.array(pairs)
    distances = np.array(distances)
    
    # Limit number of pairs if requested
    if max_pairs is not None and len(pairs) > max_pairs:
        # Randomly sample max_pairs pairs
        indices = np.random.choice(len(pairs), max_pairs, replace=False)
        pairs = pairs[indices]
        distances = distances[indices]
    
    return pairs, distances


def _sample_pairs_kdtree(points, min_distance, max_pairs=None):
    """Use scipy cKDTree for radius-based neighbor queries."""
    tree = cKDTree(points)
    
    pairs = []
    distances = []
    
    # Query radius for each point
    for i in range(len(points)):
        # Find all points within min_distance
        neighbor_indices = tree.query_ball_point(points[i], min_distance)
        
        # Remove self and only consider neighbors with index > i
        valid_neighbors = [j for j in neighbor_indices if j > i]
        
        if valid_neighbors:
            # Calculate distances
            neighbor_points = points[valid_neighbors]
            point_distances = np.linalg.norm(neighbor_points - points[i], axis=1)
            
            # Add pairs
            for j, dist in zip(valid_neighbors, point_distances):
                pairs.append([i, j])
                distances.append(dist)
    
    pairs = np.array(pairs)
    distances = np.array(distances)
    
    # Limit number of pairs if requested
    if max_pairs is not None and len(pairs) > max_pairs:
        indices = np.random.choice(len(pairs), max_pairs, replace=False)
        pairs = pairs[indices]
        distances = distances[indices]
    
    return pairs, distances


def _sample_pairs_open3d(points, min_distance, max_pairs=None):
    """Use Open3D for radius-based neighbor queries."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Build KDTree
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    pairs = []
    distances = []
    
    for i in range(len(points)):
        # Search for neighbors within radius
        [k, idx, dist_squared] = kdtree.search_radius_vector_3d(points[i], min_distance)
        
        # Remove self and only consider neighbors with index > i
        valid_neighbors = [j for j in idx[1:] if j > i]  # Skip first element (self)
        
        if valid_neighbors:
            # Calculate actual distances
            neighbor_points = points[valid_neighbors]
            point_distances = np.linalg.norm(neighbor_points - points[i], axis=1)
            
            # Add pairs
            for j, dist in zip(valid_neighbors, point_distances):
                pairs.append([i, j])
                distances.append(dist)
    
    pairs = np.array(pairs)
    distances = np.array(distances)
    
    # Limit number of pairs if requested
    if max_pairs is not None and len(pairs) > max_pairs:
        indices = np.random.choice(len(pairs), max_pairs, replace=False)
        pairs = pairs[indices]
        distances = distances[indices]
    
    return pairs, distances


def sample_points_from_mesh(mesh_path, num_points=10000, method='surface'):
    """
    Sample points from a mesh file.
    
    Args:
        mesh_path (str): Path to the OBJ mesh file
        num_points (int): Number of points to sample
        method (str): Sampling method - 'surface' or 'volume'
    
    Returns:
        numpy.ndarray: Array of sampled points with shape (num_points, 3)
    """
    try:
        # Load the mesh
        mesh = trimesh.load(mesh_path)
        
        # Ensure it's a single mesh (not a scene)
        if hasattr(mesh, 'geometry'):
            raise ValueError("Mesh is a scene, not a single mesh")
        
        print(f"Loaded mesh: {mesh_path}")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Bounds: {mesh.bounds}")

        points, face_indices = mesh.sample(num_points, return_index=True)
        points = np.array(points)
        points_colors = np.zeros_like(points)

        # plot_pcd(points, points_colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.asarray(pcd.normals)

        normals_points, normals_colors = make_line(points, points + 0.01*normals, density=1000, color=(0, 1, 0))

        points_and_normals = np.concatenate([points, normals_points], axis=0)
        points_and_normals_colors = np.concatenate([points_colors, normals_colors], axis=0)

        plot_pcd(points_and_normals, points_and_normals_colors)

        pairs, distances =sample_point_pairs_within_distance(points, 0.06, max_pairs=1000, method='kdtree')

        p1 = points[pairs[:, 0]]
        p2 = points[pairs[:, 1]]

        # TODO: Make this a robo utils function
        vector_between_points = p2 - p1
        vector_between_points = vector_between_points / np.linalg.norm(vector_between_points, axis=1, keepdims=True)

        n1 = normals[pairs[:, 0]]
        n2 = normals[pairs[:, 1]]

        alignment = np.sum(vector_between_points*n1, axis=1)
        alignment_mask = alignment > 0.75

        p1, p2, n1, n2 = p1[alignment_mask], p2[alignment_mask], n1[alignment_mask], n2[alignment_mask]

        tolerance_angle = np.pi/12      # 15 degrees
        tolerance = np.cos(np.pi - tolerance_angle)

        antipodal_mask = np.sum(n1*n2, axis=1) < tolerance

        p1, p2, n1, n2 = p1[antipodal_mask], p2[antipodal_mask], n1[antipodal_mask], n2[antipodal_mask]

        antipodal_points, antipodal_colors = make_line(p1, p2, density=1000, color=(1, 0, 0))

        points_and_antipodal_normals = np.concatenate([points_and_normals, antipodal_points], axis=0)
        points_and_antipodal_colors = np.concatenate([points_and_normals_colors, antipodal_colors], axis=0)

        plot_pcd(points_and_antipodal_normals, points_and_antipodal_colors)
                
        return points
        
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {str(e)}")
        return None

def main():
    # Example usage of point pair sampling
    mesh_file = "assets/ocrtoc_objects/models/book_2/collision.obj"
    points = sample_points_from_mesh(mesh_file, num_points=1000)
    
    if points is not None:
        print("\n" + "="*50)
        print("DEMONSTRATING POINT PAIR SAMPLING")
        print("="*50)
        
        # Example 1: Find all pairs within a small distance
        min_distance = 0.01  # 1cm
        pairs, distances = sample_point_pairs_within_distance(
            points, min_distance, max_pairs=1000, method='ball_tree'
        )
        
        print(f"\nFound {len(pairs)} pairs within {min_distance} units")
        if len(pairs) > 0:
            print(f"Distance range: {distances.min():.4f} to {distances.max():.4f}")
            print(f"Average distance: {distances.mean():.4f}")
            
            # Show first few pairs
            print("\nFirst 5 pairs (indices and distances):")
            for i in range(min(5, len(pairs))):
                print(f"  Pair {i+1}: points {pairs[i][0]} <-> {pairs[i][1]}, distance: {distances[i]:.4f}")
        
        # Example 2: Compare different methods
        print("\n" + "-"*30)
        print("COMPARING METHODS")
        print("-"*30)
        
        methods = ['ball_tree', 'kdtree', 'open3d']
        for method in methods:
            try:
                pairs_method, distances_method = sample_point_pairs_within_distance(
                    points, min_distance, max_pairs=500, method=method
                )
                print(f"{method:10}: {len(pairs_method)} pairs found")
            except Exception as e:
                print(f"{method:10}: Error - {str(e)}")
        
        # Example 3: Different distance thresholds
        print("\n" + "-"*30)
        print("DIFFERENT DISTANCE THRESHOLDS")
        print("-"*30)
        
        distances_to_test = [0.005, 0.01, 0.02, 0.05]
        for dist_thresh in distances_to_test:
            pairs_dist, _ = sample_point_pairs_within_distance(
                points, dist_thresh, max_pairs=1000, method='ball_tree'
            )
            print(f"Distance {dist_thresh:5.3f}: {len(pairs_dist):6d} pairs")


if __name__ == "__main__":
    main()
