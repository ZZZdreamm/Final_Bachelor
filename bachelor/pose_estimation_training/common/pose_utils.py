"""
Shared pose and rotation utilities.
Used across neural network training, inference, and pose refinement.
"""

import torch
import numpy as np


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        quaternion: torch.Tensor of shape (4,) or (N, 4) [w, x, y, z]

    Returns:
        Rotation matrix of shape (3, 3) or (N, 3, 3)
    """
    if quaternion.dim() == 1:
        quaternion = quaternion.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)

    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    R = torch.zeros((quaternion.shape[0], 3, 3), device=quaternion.device)

    R[:, 0, 0] = 1 - 2*y**2 - 2*z**2
    R[:, 0, 1] = 2*x*y - 2*w*z
    R[:, 0, 2] = 2*x*z + 2*w*y

    R[:, 1, 0] = 2*x*y + 2*w*z
    R[:, 1, 1] = 1 - 2*x**2 - 2*z**2
    R[:, 1, 2] = 2*y*z - 2*w*x

    R[:, 2, 0] = 2*x*z - 2*w*y
    R[:, 2, 1] = 2*y*z + 2*w*x
    R[:, 2, 2] = 1 - 2*x**2 - 2*y**2

    if squeeze:
        R = R.squeeze(0)

    return R


def normalize_quaternion(quat):
    """
    Normalize quaternion to unit length.

    Args:
        quat: numpy array or torch tensor of shape (4,) or (N, 4)

    Returns:
        Normalized quaternion
    """
    if isinstance(quat, torch.Tensor):
        return quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    else:
        return quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)


def quaternion_distance(q1, q2):
    """
    Compute geodesic distance between quaternions.

    Args:
        q1: First quaternion(s) [w, x, y, z]
        q2: Second quaternion(s) [w, x, y, z]

    Returns:
        Angle in radians
    """
    if isinstance(q1, torch.Tensor):
        q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-8)
        q2 = q2 / (torch.norm(q2, dim=-1, keepdim=True) + 1e-8)
        dot_product = torch.abs(torch.sum(q1 * q2, dim=-1))
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        return 2 * torch.acos(dot_product)
    else:
        q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-8)
        q2 = q2 / (np.linalg.norm(q2, axis=-1, keepdims=True) + 1e-8)
        dot_product = np.abs(np.sum(q1 * q2, axis=-1))
        dot_product = np.clip(dot_product, -1.0, 1.0)
        return 2 * np.arccos(dot_product)
