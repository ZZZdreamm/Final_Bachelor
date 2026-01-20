"""
Shared mesh loading and processing utilities.
Used for rendering, visualization, and pose refinement.
"""

import numpy as np
import trimesh
import cv2


def load_and_normalize_mesh(mesh_path, target_scale=10.0):
    """
    Load a mesh file and normalize it to a standard scale.

    Args:
        mesh_path: Path to mesh file (.glb, .gltf, .fbx, etc.)
        target_scale: Target size for normalization (default: 10.0)

    Returns:
        Normalized trimesh object
    """
    loaded = trimesh.load(mesh_path)

    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"Scene has no geometry: {mesh_path}")
        meshes = [geom for geom in loaded.geometry.values()
                 if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No valid meshes in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded

    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    mesh.vertices -= center

    scale = np.max(bounds[1] - bounds[0])
    if scale > 0:
        mesh.vertices /= (scale / target_scale)

    return mesh


def get_mesh_bounds(mesh):
    """
    Get bounding box information from a trimesh object.

    Args:
        mesh: trimesh object

    Returns:
        dict with min, max, center, size, radius
    """
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = np.linalg.norm(bounds[1] - bounds[0])
    radius = np.max(bounds[1] - bounds[0]) / 2

    return {
        "min": bounds[0],
        "max": bounds[1],
        "center": center,
        "size": size,
        "radius": radius
    }


def project_bbox_to_2d(bbox_corners_3d, rotation_matrix, translation,
                       camera_intrinsics):
    """
    Project 3D bounding box corners to 2D image coordinates.

    Args:
        bbox_corners_3d: (8, 3) array of 3D corners
        rotation_matrix: (3, 3) camera rotation
        translation: (3,) camera translation
        camera_intrinsics: (3, 3) camera intrinsic matrix

    Returns:
        (8, 2) array of 2D projected points
    """
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    tvec = translation.reshape(3, 1)

    bbox_2d, _ = cv2.projectPoints(
        bbox_corners_3d,
        rvec,
        tvec,
        camera_intrinsics,
        np.zeros(5)
    )

    return bbox_2d.reshape(-1, 2)


def get_bbox_edges():
    """
    Get the edge connectivity of a 3D bounding box.

    Returns:
        List of (i, j) tuples representing edges
    """
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]


def create_bbox_corners(bounds):
    """
    Create 8 corners of a 3D bounding box.

    Args:
        bounds: Mesh bounds (min and max corners)

    Returns:
        (8, 3) array of corner coordinates
    """
    min_corner, max_corner = bounds[0], bounds[1]

    return np.array([
        [min_corner[0], min_corner[1], min_corner[2]],
        [max_corner[0], min_corner[1], min_corner[2]],
        [max_corner[0], max_corner[1], min_corner[2]],
        [min_corner[0], max_corner[1], min_corner[2]],
        [min_corner[0], min_corner[1], max_corner[2]],
        [max_corner[0], min_corner[1], max_corner[2]],
        [max_corner[0], max_corner[1], max_corner[2]],
        [min_corner[0], max_corner[1], max_corner[2]],
    ], dtype=np.float32)
