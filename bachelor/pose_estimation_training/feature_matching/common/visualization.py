"""
Shared visualization utilities for pose estimation.
Provides 3D bounding box visualization and match visualization.
"""

import numpy as np
import cv2
import random
from typing import Dict, List
from .pose_utils import PoseResult


def get_bbox_corners_3d(bounds: Dict) -> np.ndarray:
    """
    Create 8 corners of a 3D bounding box.

    Args:
        bounds: Dict with 'min' and 'max' corners

    Returns:
        (8, 3) array of corner coordinates
    """
    min_pt = np.array(bounds['min'])
    max_pt = np.array(bounds['max'])

    return np.array([
        [min_pt[0], min_pt[1], min_pt[2]],  # 0: min corner
        [max_pt[0], min_pt[1], min_pt[2]],  # 1
        [max_pt[0], max_pt[1], min_pt[2]],  # 2
        [min_pt[0], max_pt[1], min_pt[2]],  # 3
        [min_pt[0], min_pt[1], max_pt[2]],  # 4
        [max_pt[0], min_pt[1], max_pt[2]],  # 5
        [max_pt[0], max_pt[1], max_pt[2]],  # 6: max corner
        [min_pt[0], max_pt[1], max_pt[2]],  # 7
    ], dtype=np.float64)


def get_bbox_edges() -> List[tuple]:
    """
    Get edge connectivity for a 3D bounding box.

    Returns:
        List of (start_idx, end_idx) tuples for each edge
    """
    return [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]


def visualize_pose_with_bbox(image: np.ndarray,
                             result: PoseResult,
                             model_bounds: Dict,
                             output_path: str) -> None:
    """
    Visualize estimated pose by drawing 3D bounding box on image.

    Args:
        image: Input image (BGR)
        result: PoseResult containing pose estimation
        model_bounds: Model bounding box from database
        output_path: Where to save visualization
    """
    if not result.success:
        cv2.imwrite(output_path, image)
        return

    # Get 3D bounding box corners
    corners_3d = get_bbox_corners_3d(model_bounds)

    # Project to 2D
    rvec, _ = cv2.Rodrigues(result.rotation_R)
    corners_2d, _ = cv2.projectPoints(
        corners_3d, rvec, result.translation_t,
        result.intrinsic_K, np.zeros(4)
    )
    corners_2d = corners_2d.reshape(-1, 2).astype(int)

    # Draw on image
    vis_image = image.copy()

    # Draw edges
    edges = get_bbox_edges()
    for start, end in edges:
        cv2.line(vis_image,
                tuple(corners_2d[start]),
                tuple(corners_2d[end]),
                (0, 255, 0), 2)

    # Add text info
    cv2.putText(vis_image,
               f"{result.model_name} ({result.method})",
               (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(vis_image,
               f"Inliers: {result.num_inliers}, Error: {result.reprojection_error:.1f}px",
               (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(output_path, vis_image)


def draw_match_visualization(query_image: np.ndarray,
                            render_image: np.ndarray,
                            points_query: np.ndarray,
                            points_render: np.ndarray,
                            methods: List[str],
                            output_path: str,
                            max_matches: int = 100) -> None:
    """
    Create side-by-side visualization showing matched keypoints.

    Args:
        query_image: Query image (BGR)
        render_image: Render image (BGR)
        points_query: Matched points in query (N, 2)
        points_render: Matched points in render (N, 2)
        methods: List of method names for each match
        output_path: Where to save visualization
        max_matches: Maximum number of matches to display
    """
    h1, w1 = query_image.shape[:2]
    h2, w2 = render_image.shape[:2]

    # Resize to target height
    target_height = 800
    scale1 = target_height / h1
    scale2 = target_height / h2

    query_resized = cv2.resize(query_image, (int(w1 * scale1), target_height))
    render_resized = cv2.resize(render_image, (int(w2 * scale2), target_height))

    # Create combined image
    w_total = query_resized.shape[1] + render_resized.shape[1]
    vis_image = np.zeros((target_height, w_total, 3), dtype=np.uint8)

    vis_image[:, :query_resized.shape[1]] = query_resized
    vis_image[:, query_resized.shape[1]:] = render_resized

    offset_x = query_resized.shape[1]

    # Subsample matches if too many
    num_matches = len(points_query)
    if num_matches > max_matches:
        indices = random.sample(range(num_matches), max_matches)
        points_query = points_query[indices]
        points_render = points_render[indices]
        methods = [methods[i] for i in indices]

    # Define colors for each method
    method_colors = {
        'loftr': (0, 0, 255),       # Red
        'superpoint': (0, 255, 255),  # Yellow
        'sift': (255, 0, 255)       # Magenta
    }

    # Draw matches
    for pt_q, pt_r, method in zip(points_query, points_render, methods):
        pt1 = (int(pt_q[0] * scale1), int(pt_q[1] * scale1))
        pt2 = (int(pt_r[0] * scale2) + offset_x, int(pt_r[1] * scale2))

        color = method_colors.get(method, (0, 255, 0))

        cv2.line(vis_image, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(vis_image, pt1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(vis_image, pt2, 3, color, -1, cv2.LINE_AA)

    # Add labels
    cv2.putText(vis_image, "Query Image", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_image, "Best Match", (offset_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Add method counts
    method_counts = {}
    for method in set(methods):
        method_counts[method] = methods.count(method)

    y_pos = 60
    for method, color in method_colors.items():
        count = method_counts.get(method, 0)
        if count > 0:
            cv2.putText(vis_image, f"{method}: {count}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            y_pos += 25

    total = sum(method_counts.values())
    cv2.putText(vis_image, f"Total: {total}", (10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Add legend
    legend_y = target_height - 100
    cv2.putText(vis_image, "Legend:", (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (method, color) in enumerate(method_colors.items()):
        cv2.putText(vis_image, f"{method}", (10, legend_y + 20 + i * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, vis_image)
