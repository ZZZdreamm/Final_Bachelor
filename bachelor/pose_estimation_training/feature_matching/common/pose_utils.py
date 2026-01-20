"""
Shared pose estimation utilities for feature-based methods.
Provides intrinsics estimation, 3D correspondences, PnP solving, and data structures.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PoseResult:
    """Result of pose estimation with all necessary metadata"""
    success: bool
    model_name: str
    confidence: float
    intrinsic_K: np.ndarray
    rotation_R: np.ndarray
    translation_t: np.ndarray
    transform_4x4: np.ndarray
    num_inliers: int
    reprojection_error: float
    matched_render_idx: int
    method: str = ""
    num_matches: int = 0
    num_3d_correspondences: int = 0

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        return {
            'success': self.success,
            'model_name': self.model_name,
            'confidence': self.confidence,
            'intrinsic_K': self.intrinsic_K.tolist() if self.intrinsic_K is not None else None,
            'rotation_R': self.rotation_R.tolist() if self.rotation_R is not None else None,
            'translation_t': self.translation_t.tolist() if self.translation_t is not None else None,
            'transform_4x4': self.transform_4x4.tolist() if self.transform_4x4 is not None else None,
            'num_inliers': self.num_inliers,
            'reprojection_error': self.reprojection_error,
            'matched_render_idx': self.matched_render_idx,
            'method': self.method,
            'num_matches': self.num_matches,
            'num_3d_correspondences': self.num_3d_correspondences
        }


def estimate_camera_intrinsics(image_shape: Tuple[int, int],
                               focal_length_mm: float = 28.0,
                               sensor_width_mm: float = 6.17) -> np.ndarray:
    """
    Estimate camera intrinsic matrix from image dimensions.

    Default values are for typical smartphone camera.

    Args:
        image_shape: (height, width) of the image
        focal_length_mm: Focal length in millimeters
        sensor_width_mm: Sensor width in millimeters

    Returns:
        3x3 camera intrinsic matrix K
    """
    height, width = image_shape
    fx = fy = (width * focal_length_mm) / sensor_width_mm
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_render_image(renders_root: Path, model_name: str, render_idx: int) -> Optional[np.ndarray]:
    """
    Load a render image from disk.

    Tries both 'model/images/render_XXXX.png' and 'model/render_XXXX.png' paths.

    Args:
        renders_root: Root directory containing model folders
        model_name: Name of the model
        render_idx: Index of the render

    Returns:
        BGR image or None if not found
    """
    img_name = f"render_{render_idx:04d}.png"

    # Try images/ subdirectory first
    path = renders_root / model_name / "images" / img_name
    if path.exists():
        return cv2.imread(str(path))

    # Try direct path
    path = renders_root / model_name / img_name
    if path.exists():
        return cv2.imread(str(path))

    return None


def find_3d_correspondences(points_2d_query: np.ndarray,
                           points_2d_render: np.ndarray,
                           render_data: Dict,
                           model_data: Dict,
                           snap_threshold: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map matched 2D render points to 3D world points.

    This function "snaps" 2D matches from the render image to the nearest
    pre-computed 2D projection of 3D surface points, then returns the
    corresponding 3D points for PnP solving.

    Args:
        points_2d_query: Matched points in query image (N, 2)
        points_2d_render: Matched points in render image (N, 2)
        render_data: Render metadata from database (contains 'points_2d' or projection params)
        model_data: Model metadata from database (contains 'surface_points_3d')
        snap_threshold: Maximum pixel distance to snap to surface point

    Returns:
        Tuple of (query_2d_points, world_3d_points) for PnP
    """
    surface_3d = np.array(model_data['surface_points_3d'])

    # Get 2D projections of 3D surface points in the render
    if 'points_2d' in render_data:
        points_2d_surface = render_data['points_2d']
    else:
        # Fallback: project 3D points using stored camera parameters
        K = np.array(render_data['intrinsic_K'])
        R = np.array(render_data['extrinsic_R'])
        t = np.array(render_data['extrinsic_t']).reshape(3, 1)

        points_2d_surface = []
        for pt_3d in surface_3d:
            pt_cam = R @ np.array(pt_3d).reshape(3, 1) + t
            if pt_cam[2] > 0:  # In front of camera
                pt_2d = K @ pt_cam
                pt_2d = pt_2d[:2] / pt_2d[2]
                points_2d_surface.append({'x': float(pt_2d[0]), 'y': float(pt_2d[1]), 'visible': 1})
            else:
                points_2d_surface.append({'x': 0, 'y': 0, 'visible': 0})

    # Extract visible 2D locations and their 3D indices
    visible_2d_locs = []
    visible_3d_indices = []

    for i, pt_info in enumerate(points_2d_surface):
        if pt_info['visible'] == 1:
            visible_2d_locs.append([pt_info['x'], pt_info['y']])
            visible_3d_indices.append(i)

    if len(visible_2d_locs) == 0:
        return np.array([]), np.array([])

    visible_2d_locs = np.array(visible_2d_locs)

    # Snap each match to the nearest visible surface point
    matched_3d = []
    matched_2d_query = []

    for i, pt_render in enumerate(points_2d_render):
        distances = np.linalg.norm(visible_2d_locs - pt_render, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist < snap_threshold:
            surface_idx = visible_3d_indices[min_idx]
            matched_3d.append(surface_3d[surface_idx])
            matched_2d_query.append(points_2d_query[i])

    return np.array(matched_2d_query), np.array(matched_3d)


def solve_pnp_ransac(points_2d: np.ndarray,
                     points_3d: np.ndarray,
                     K: np.ndarray,
                     ransac_iterations: int = 2000,
                     reprojection_threshold: float = 6.0) -> Tuple[bool, np.ndarray, np.ndarray, int, float]:
    """
    Solve Perspective-n-Point problem using RANSAC.

    Args:
        points_2d: 2D points in query image (N, 2)
        points_3d: Corresponding 3D points in world (N, 3)
        K: Camera intrinsic matrix (3, 3)
        ransac_iterations: Number of RANSAC iterations
        reprojection_threshold: RANSAC inlier threshold in pixels

    Returns:
        Tuple of (success, R, t, num_inliers, reprojection_error)
    """
    if len(points_2d) < 4:
        return False, None, None, 0, float('inf')

    dist_coeffs = np.zeros((4, 1))

    try:
        # RANSAC-based PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d.astype(np.float64),
            points_2d.astype(np.float64),
            K,
            dist_coeffs,
            iterationsCount=ransac_iterations,
            reprojectionError=reprojection_threshold,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP
        )
    except Exception as e:
        return False, None, None, 0, float('inf')

    if not success or inliers is None:
        return False, None, None, 0, float('inf')

    num_inliers = len(inliers)

    # Refine with inliers only
    if num_inliers >= 4:
        inlier_2d = points_2d[inliers.flatten()]
        inlier_3d = points_3d[inliers.flatten()]
        success, rvec, tvec = cv2.solvePnP(
            inlier_3d, inlier_2d, K, dist_coeffs,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    # Convert to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    # Compute reprojection error on inliers
    projected, _ = cv2.projectPoints(points_3d[inliers.flatten()], rvec, tvec, K, dist_coeffs)
    error = np.mean(np.linalg.norm(
        projected.reshape(-1, 2) - points_2d[inliers.flatten()].reshape(-1, 2),
        axis=1
    ))

    return True, R, t, num_inliers, error
