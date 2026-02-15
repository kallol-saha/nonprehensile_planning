from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Union
# torch is not always required to run stuff from robo_utils
try:
    import torch
except ImportError:
    torch = None
import open3d as o3d

# Type hint that works whether torch is available or not
if torch is not None:
    TensorType = Union[np.ndarray, torch.Tensor]
else:
    TensorType = Union[np.ndarray, "torch.Tensor"]  # String literal for when torch is not available

def quaternion_to_matrix(quaternion, format='xyzw'):
    """
    Convert a quaternion to a rotation matrix.
    """
    quaternion = np.asarray(quaternion)
    if format == 'wxyz':
        # Reorder from [w, x, y, z] to [x, y, z, w]
        return R.from_quat(quaternion[..., [1, 2, 3, 0]]).as_matrix()
    elif format == 'xyzw':
        # Already in xyzw format, use as is
        return R.from_quat(quaternion).as_matrix()
    else:
        raise ValueError(f"Invalid quaternion format: {format}")
    
def matrix_to_quaternion(matrix, format='xyzw'):
    """
    Convert a rotation matrix to a quaternion.
    """
    # as_quat() returns xyzw format by default (scalar last)
    quat = R.from_matrix(matrix).as_quat()
    
    if format == 'wxyz':
        # Reorder from [x, y, z, w] to [w, x, y, z]
        return quat[..., [3, 0, 1, 2]]
    elif format == 'xyzw':
        # Already in xyzw format, return as is
        return quat
    else:
        raise ValueError(f"Invalid matrix format: {format}")

def angle_between_vectors(v1, v2, eps=1e-12):
    """
    Computes the angle between vectors v1 and v2.
    v1, v2: shape (..., 3)
    Returns: angle in radians, shape (...)
    """

    if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
        raise ValueError(f"v1 and v2 must be a numpy arrays")

    # Reshape to (..., 3) if needed:
    if v1.ndim == 1:
        v1 = v1[np.newaxis, :]
    if v2.ndim == 1:
        v2 = v2[np.newaxis, :]

    assert v1.ndim == 2 and v2.ndim == 2, "v1 and v2 must have shape (..., 3)"
    assert v1.shape[-1] == 3 and v2.shape[-1] == 3, "v1 and v2 must have shape (..., 3)"

    v1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + eps)
    v2 = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + eps)

    cos_theta = np.sum(v1 * v2, axis=-1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)

def pose_to_transformation(pose: np.ndarray, format='xyzw'):
    """
    Convert a pose to a transformation matrix.
    pose is a numpy array of shape (7,) or (N, 7)
    Returns a numpy array of shape (4, 4) or (N, 4, 4)
    """

    # Make dimensions consistent:
    if not isinstance(pose, np.ndarray):
        raise ValueError(f"Pose must be a numpy array")

    reshape_pose = False
    if len(pose.shape) == 1:
        reshape_pose = True
        pose = pose[np.newaxis, :]
    transformation_matrix = np.eye(4, 4)[np.newaxis, :, :].repeat(pose.shape[0], axis=0)

    # Convert to transformation matrix:
    transformation_matrix[..., :3, :3] = quaternion_to_matrix(pose[..., 3:], format=format)
    transformation_matrix[..., :3, 3] = pose[..., :3]

    if reshape_pose:
        transformation_matrix = transformation_matrix[0, :, :]
    return transformation_matrix

def transformation_to_pose(transformation_matrix, format='xyzw'):
    """
    Convert a transformation matrix to a pose.
    Accepts numpy array of shape (4, 4) or (N, 4, 4).
    Returns numpy array of shape (7,) or (N, 7) correspondingly.
    """
    if not isinstance(transformation_matrix, np.ndarray):
        raise ValueError("transformation_matrix must be a numpy array")

    reshape_out = False
    if transformation_matrix.ndim == 2:
        reshape_out = True
        transformation_matrix = transformation_matrix[np.newaxis, ...]  # (1,4,4)

    # Positions
    pos = transformation_matrix[:, :3, 3]  # (N,3)
    # Rotations to quaternions (vectorized)
    quat = matrix_to_quaternion(transformation_matrix[:, :3, :3], format=format)  # (N,4)

    poses = np.concatenate([pos, quat], axis=-1)  # (N,7)
    if reshape_out:
        poses = poses[0]
    return poses

def invert_transformation(transformation_matrix):
    """Inverts the given transformation matrix.
    
    Args:
        transformation_matrix: (..., 4, 4) transformation matrix (numpy ndarray only)
        
    Returns:
        inverse_transformation_matrix: (..., 4, 4) inverted transformation matrix (numpy ndarray)
    """
    if not isinstance(transformation_matrix, np.ndarray):
        raise ValueError("transformation_matrix must be a numpy array")

    if transformation_matrix.ndim == 2:
        tm = transformation_matrix[np.newaxis, ...]
        squeeze_out = True
    else:
        tm = transformation_matrix
        squeeze_out = False

    R = tm[..., :3, :3]
    t = tm[..., :3, 3]
    R_inv = np.linalg.inv(R)
    t_inv = -np.matmul(R_inv, t[..., None])[..., 0]

    batch_shape = tm.shape[:-2]
    I = np.eye(4, dtype=tm.dtype).reshape((1, 4, 4)).repeat(np.prod(batch_shape, dtype=int) if len(batch_shape) > 0 else 1, axis=0)
    I = I.reshape(*batch_shape, 4, 4)
    I[..., :3, :3] = R_inv
    I[..., :3, 3] = t_inv

    if squeeze_out:
        return I[0]
    return I

def invert_pose(pose: np.ndarray, format='xyzw') -> np.ndarray:
    """Inverts the given pose.
    pose is a numpy array of shape (7,)
    Returns a numpy array of shape (7,)
    """
    inverse_transformation_matrix = invert_transformation(pose_to_transformation(pose, format=format))
    return transformation_to_pose(inverse_transformation_matrix, format=format)

def transform_pcd(pcd, transform):
    """Transforms the given point cloud by the given transformation matrix.

    Args:
    -----
        pcd: Nx3 point cloud (numpy array or torch tensor)
        transform: 4x4 transformation matrix (numpy array or torch tensor)

    Returns:
    --------
            pcd_new: Nx3 transformed point cloud (same type as input)
    """

    # Convert both to numpy arrays if types don't match
    is_pcd_torch = torch is not None and isinstance(pcd, torch.Tensor)
    is_transform_torch = torch is not None and isinstance(transform, torch.Tensor)
    
    if is_pcd_torch != is_transform_torch:
        print("WARNING: Point cloud and transformation matrix must be same type (both numpy arrays or both torch tensors)")
        if is_pcd_torch:
            pcd = pcd.cpu().numpy()
        if is_transform_torch:
            transform = transform.cpu().numpy()
    
    if is_pcd_torch:
        if pcd.shape[1] != 4:
            ones = torch.ones((pcd.shape[0], 1), device=pcd.device)
            pcd = torch.cat((pcd, ones), dim=1)
        pcd_new = torch.matmul(transform, pcd.T)[:-1, :].T
    else:
        if pcd.shape[1] != 4:
            pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
        pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new

def furthest_point_sample(pcd: TensorType, num_points: int = 1024):
    """
    """
    if torch is not None and isinstance(pcd, torch.Tensor):
        original_type = "torch"
        tensor_device = pcd.device
        pcd = pcd.detach().cpu().numpy()
    elif not isinstance(pcd, np.ndarray):
        raise ValueError("pcd must be a numpy array or torch tensor")
    else:
        original_type = "numpy"
        tensor_device = None

    if pcd.shape[1] != 3 and len(pcd.shape) != 2:
        raise ValueError("pcd must be a numpy array or torch tensor of shape (N, 3)")

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)

    downsampled_pcd = pcd_o3d.farthest_point_down_sample(num_samples=num_points)
    downsampled_pcd = np.asarray(downsampled_pcd.points)

    if original_type == "torch":
        if torch is None:
            raise ImportError("torch is not available. Cannot return torch tensor.")
        return torch.from_numpy(downsampled_pcd).to(tensor_device)
    else:
        return downsampled_pcd

def move_pose_along_local_z(pose: TensorType, distance: float, format: str = 'wxyz'):
    """Translate pose(s) along their local +Z axis by a given distance.

    Args:
        pose: Pose as (7,) or (N,7), numpy ndarray or torch tensor, ordered
              (x, y, z, qw, qx, qy, qz) if format=='wxyz', or (x, y, z, qx, qy, qz, qw) if 'xyzw'.
        distance: Scalar translation to apply along local Z (meters).
        format: Quaternion convention, 'wxyz' or 'xyzw'.

    Returns:
        Pose(s) with updated position, same shape/type/device as input.
    """

    if torch is not None and isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()

    if isinstance(pose, np.ndarray):
        single = False
        p = pose
        if p.ndim == 1:
            p = p[np.newaxis, :]
            single = True
        pos = p[:, :3]
        if format == 'wxyz':
            quat = p[:, 3:7]
        elif format == 'xyzw':
            quat = p[:, 3:7][:, [3, 0, 1, 2]]  # to wxyz
        else:
            raise ValueError(f"Invalid quaternion format: {format}")

        R = quaternion_to_matrix(quat, format='wxyz')  # (N,3,3)
        z_axis = R[:, :, 2]  # (N,3)
        pos_new = pos + (distance * z_axis)
        out = p.copy()
        out[:, :3] = pos_new
        if single:
            out = out[0]
        return out

    else:
        raise ValueError("pose must be a numpy array or torch tensor")

def move_pose_along_local_x(pose: TensorType, distance: float, format: str = 'wxyz'):
    """Translate pose(s) along their local +X axis by a given distance.

    Args:
        pose: Pose as (7,) or (N,7), numpy ndarray or torch tensor, ordered
              (x, y, z, qw, qx, qy, qz) if format=='wxyz', or (x, y, z, qx, qy, qz, qw) if 'xyzw'.
        distance: Scalar translation to apply along local X (meters).
        format: Quaternion convention, 'wxyz' or 'xyzw'.

    Returns:
        Pose(s) with updated position, same shape/type/device as input.
    """

    if torch is not None and isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()

    if isinstance(pose, np.ndarray):
        single = False
        p = pose
        if p.ndim == 1:
            p = p[np.newaxis, :]
            single = True
        pos = p[:, :3]
        if format == 'wxyz':
            quat = p[:, 3:7]
        elif format == 'xyzw':
            quat = p[:, 3:7][:, [3, 0, 1, 2]]  # to wxyz
        else:
            raise ValueError(f"Invalid quaternion format: {format}")

        R = quaternion_to_matrix(quat, format='wxyz')  # (N,3,3)
        x_axis = R[:, :, 0]  # (N,3)
        pos_new = pos + (distance * x_axis)
        out = p.copy()
        out[:, :3] = pos_new
        if single:
            out = out[0]
        return out

    else:
        raise ValueError("pose must be a numpy array or torch tensor")

def move_pose_along_local_y(pose: TensorType, distance: float, format: str = 'wxyz'):
    """Translate pose(s) along their local +Y axis by a given distance.

    Args:
        pose: Pose as (7,) or (N,7), numpy ndarray or torch tensor, ordered
              (x, y, z, qw, qx, qy, qz) if format=='wxyz', or (x, y, z, qx, qy, qz, qw) if 'xyzw'.
        distance: Scalar translation to apply along local Y (meters).
        format: Quaternion convention, 'wxyz' or 'xyzw'.

    Returns:
        Pose(s) with updated position, same shape/type/device as input.
    """

    if torch is not None and isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()

    if isinstance(pose, np.ndarray):
        single = False
        p = pose
        if p.ndim == 1:
            p = p[np.newaxis, :]
            single = True
        pos = p[:, :3]
        if format == 'wxyz':
            quat = p[:, 3:7]
        elif format == 'xyzw':
            quat = p[:, 3:7][:, [3, 0, 1, 2]]  # to wxyz
        else:
            raise ValueError(f"Invalid quaternion format: {format}")

        R = quaternion_to_matrix(quat, format='wxyz')  # (N,3,3)
        y_axis = R[:, :, 1]  # (N,3)
        pos_new = pos + (distance * y_axis)
        out = p.copy()
        out[:, :3] = pos_new
        if single:
            out = out[0]
        return out

    else:
        raise ValueError("pose must be a numpy array or torch tensor")

def rotate_pose_around_local_x(pose: TensorType, angle: float, format: str = 'wxyz'):
    """Rotate pose(s) around their local +X axis by a given angle.

    Args:
        pose: Pose as (7,) or (N,7), numpy ndarray or torch tensor, ordered
              (x, y, z, qw, qx, qy, qz) if format=='wxyz', or (x, y, z, qx, qy, qz, qw) if 'xyzw'.
        angle: Rotation angle in radians (positive = counterclockwise when looking along +X).
        format: Quaternion convention, 'wxyz' or 'xyzw'.

    Returns:
        Pose(s) with updated orientation, same shape/type/device as input.
    """

    if torch is not None and isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()

    if isinstance(pose, np.ndarray):
        single = False
        p = pose
        if p.ndim == 1:
            p = p[np.newaxis, :]
            single = True
        pos = p[:, :3]
        if format == 'wxyz':
            quat = p[:, 3:7]
        elif format == 'xyzw':
            quat = p[:, 3:7][:, [3, 0, 1, 2]]  # to wxyz
        else:
            raise ValueError(f"Invalid quaternion format: {format}")

        # Get current rotation
        current_rot = R.from_quat(quat[..., [1, 2, 3, 0]])  # Convert wxyz to xyzw for scipy
        
        # Create rotation around local X axis
        # To rotate around the local X axis:
        # 1. The local X axis in world coordinates is the first column of the rotation matrix
        # 2. Create a rotation around that world axis using axis-angle representation
        # 3. Compose: R_new = R_around_world_axis * R_current
        
        current_matrix = current_rot.as_matrix()  # (N, 3, 3)
        x_axis_world = current_matrix[:, :, 0]  # (N, 3) - local X axis direction in world frame
        
        # Create rotation around the world axis (which is the local X axis direction)
        # Use axis-angle representation: rotation vector = angle * normalized_axis
        # Normalize the axis (should already be unit, but normalize to be safe)
        x_axis_normalized = x_axis_world / (np.linalg.norm(x_axis_world, axis=-1, keepdims=True) + 1e-10)
        rotvec = angle * x_axis_normalized  # (N, 3)
        
        # Handle batch case: scipy Rotation.from_rotvec can handle (N, 3) arrays
        x_axis_rot = R.from_rotvec(rotvec)
        
        # Compose rotations: rotate around world axis (local X direction), then apply current rotation
        # This gives us rotation around the local X axis
        new_rot = x_axis_rot * current_rot
        
        # Convert back to quaternion
        new_quat = new_rot.as_quat()  # Returns xyzw, shape (N, 4)
        
        # Convert back to desired format
        if format == 'wxyz':
            new_quat = new_quat[..., [3, 0, 1, 2]]  # xyzw to wxyz
        # else: already in xyzw format
        
        out = p.copy()
        out[:, 3:7] = new_quat
        if single:
            out = out[0]
        return out

    else:
        raise ValueError("pose must be a numpy array or torch tensor")

def downsample_point_cloud(pcd, num_points=4096):
    """Downsample point cloud to specified number of points using Open3D."""
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.cpu().numpy()
    
    # Create Open3D point cloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
    
    # Downsample if needed
    if len(pcd) > num_points:
        o3d_pcd = o3d_pcd.farthest_point_down_sample(num_points)
    elif len(pcd) < num_points:
        # If we have fewer points, we can't upsample, so just return as is
        # or pad with zeros (but that's not ideal)
        pass
    
    # Convert back to numpy
    downsampled_pcd = np.asarray(o3d_pcd.points)
    
    # Ensure exactly num_points (pad or truncate if needed)
    if len(downsampled_pcd) < num_points:
        # Pad with last point
        padding = np.tile(downsampled_pcd[-1:], (num_points - len(downsampled_pcd), 1))
        downsampled_pcd = np.vstack([downsampled_pcd, padding])
    elif len(downsampled_pcd) > num_points:
        # Truncate
        downsampled_pcd = downsampled_pcd[:num_points]
    
    return downsampled_pcd

def move_transformation_along_local_z(transformation_matrix: np.ndarray, distance: float):
    """Transforms the given transformation matrix along the local +Z axis by a given distance.

    Args:
        transformation_matrix: (4, 4) transformation matrix (numpy array or torch tensor)
        distance: Scalar translation to apply along local Z (meters).

    Returns:
        transformation_matrix_new: (4, 4) transformed transformation matrix (same type as input)
    """
    if not isinstance(transformation_matrix, np.ndarray):
        raise ValueError("transformation_matrix must be a numpy array")

    single = False
    T = transformation_matrix
    if T.ndim == 2:
        if T.shape != (4, 4):
            raise ValueError("Expected (4,4) or (N,4,4) transformation matrix")
        T = T[np.newaxis, ...]
        single = True
    elif T.ndim == 3:
        if T.shape[-2:] != (4, 4):
            raise ValueError("Expected (4,4) or (N,4,4) transformation matrix")
    else:
        raise ValueError("Expected (4,4) or (N,4,4) transformation matrix")

    R = T[:, :3, :3]
    t = T[:, :3, 3]
    z_axis = R[:, :, 2]
    t_new = t + distance * z_axis

    out = T.copy()
    out[:, :3, 3] = t_new
    if single:
        return out[0]
    return out