"""
Camera-related utilities for Blender rendering and pose calculations.
"""

import bpy
import random
from math import radians
from mathutils import Vector, Euler


def get_camera_intrinsic_matrix(cam, render):
    """
    Compute camera intrinsic matrix K with correct focal length calculation.

    The intrinsic matrix converts 3D camera coordinates to 2D pixel coordinates.

    Args:
        cam: Blender camera object
        render: Blender render settings

    Returns:
        K: 3x3 intrinsic matrix as list of lists
    """
    scale = render.resolution_percentage / 100.0
    width = render.resolution_x * scale
    height = render.resolution_y * scale

    sensor_width_mm = cam.data.sensor_width
    sensor_height_mm = cam.data.sensor_height

    focal_length_mm = cam.data.lens

    if cam.data.sensor_fit == 'VERTICAL':
        fx = (focal_length_mm / sensor_height_mm) * height
        fy = (focal_length_mm / sensor_height_mm) * height
    elif cam.data.sensor_fit == 'HORIZONTAL':
        fx = (focal_length_mm / sensor_width_mm) * width
        fy = (focal_length_mm / sensor_width_mm) * width
    else:
        if width >= height:
            fx = (focal_length_mm / sensor_width_mm) * width
            fy = (focal_length_mm / sensor_width_mm) * width
        else:
            fx = (focal_length_mm / sensor_height_mm) * height
            fy = (focal_length_mm / sensor_height_mm) * height

    cx = width / 2.0
    cy = height / 2.0

    K = [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ]

    return K


def setup_camera_for_render(cam, center, size, cam_config, direction,
                            height_offset, framing_strategy=None):
    """
    Position camera for realistic phone photography.

    Args:
        cam: Blender camera object
        center: Model center position (Vector)
        size: Model size (float)
        cam_config: Camera configuration dict
        direction: View direction (Vector)
        height_offset: Height offset for camera (float)
        framing_strategy: Optional framing strategy dict
    """
    cam.data.lens = cam_config["lens"]

    base_distance = size * cam_config["distance_factor"]

    if framing_strategy:
        final_distance = base_distance * framing_strategy["zoom_mult"]
    else:
        final_distance = base_distance

    final_distance *= random.uniform(0.97, 1.03)

    view_dir = direction.normalized()
    view_dir.z = height_offset
    view_dir.normalize()

    cam.location = center + view_dir * final_distance

    if hasattr(cam, 'look_at'):
        cam.look_at(center)

    bpy.context.view_layer.update()

    if framing_strategy:
        cam_right = cam.matrix_world.to_3x3().col[0]
        cam_up = cam.matrix_world.to_3x3().col[1]

        offset_x = framing_strategy["offset_x"] * size * random.uniform(0.9, 1.1)
        offset_y = framing_strategy["offset_y"] * size * random.uniform(0.9, 1.1)

        shifted_target = center + (cam_right * offset_x) + (cam_up * offset_y)

        target_jitter = Vector((
            random.uniform(-0.05, 0.05),
            random.uniform(-0.05, 0.05),
            random.uniform(-0.02, 0.02)
        )) * size

        cam.look_at(shifted_target + target_jitter)
    else:
        jitter = cam_config["position_jitter"] * size
        offset = Vector((
            random.uniform(-jitter, jitter),
            random.uniform(-jitter, jitter),
            random.uniform(-jitter * 0.5, jitter * 0.5)
        ))
        cam.location += offset

        target_offset = Vector((
            random.uniform(-0.05, 0.05),
            random.uniform(-0.05, 0.05),
            random.uniform(-0.02, 0.02)
        )) * size
        cam.look_at(center + target_offset)

    tilt_range = cam_config["tilt_range"]
    cam.rotation_euler.rotate(Euler((
        radians(random.uniform(tilt_range[0], tilt_range[1])),
        0,
        radians(random.uniform(tilt_range[0] * 0.5, tilt_range[1] * 0.5))
    ), 'XYZ'))

    bpy.context.view_layer.update()


def get_camera_pose_for_training(cam, model_center):
    """
    Extract camera extrinsic parameters as used by Blender for rendering.

    CRITICAL: This must be called AFTER bpy.context.view_layer.update()

    Args:
        cam: Blender camera object
        model_center: Center of the model (Vector)

    Returns:
        dict: Camera-to-world transformation containing:
            - rotation_matrix: 3x3 rotation of camera in world space
            - translation: [x, y, z] camera position in world space
            - rotation_quaternion: [w, x, y, z] rotation as quaternion
            - camera_forward: Forward direction vector
            - distance_to_center: Distance from camera to model center
            - looking_at_center: Boolean indicating if camera is pointing at center
            - model_center: Model center coordinates
    """
    bpy.context.view_layer.update()

    cam_world_matrix = cam.matrix_world.copy()

    rotation_mat_3x3 = cam_world_matrix.to_3x3()

    translation_vec = cam_world_matrix.to_translation()

    quat = rotation_mat_3x3.to_quaternion()
    rotation_quaternion = [quat.w, quat.x, quat.y, quat.z]

    translation = [translation_vec.x, translation_vec.y, translation_vec.z]

    rotation_matrix = [
        [rotation_mat_3x3[0][0], rotation_mat_3x3[0][1], rotation_mat_3x3[0][2]],
        [rotation_mat_3x3[1][0], rotation_mat_3x3[1][1], rotation_mat_3x3[1][2]],
        [rotation_mat_3x3[2][0], rotation_mat_3x3[2][1], rotation_mat_3x3[2][2]]
    ]

    forward_local = Vector((0, 0, -1))
    forward_world = rotation_mat_3x3 @ forward_local

    cam_to_center = model_center - translation_vec
    distance_to_center = cam_to_center.length

    forward_normalized = forward_world.normalized()
    to_center_normalized = cam_to_center.normalized()
    dot_product = forward_normalized.dot(to_center_normalized)

    return {
        "rotation_quaternion": rotation_quaternion,
        "translation": translation,
        "rotation_matrix": rotation_matrix,
        "camera_forward": [forward_world.x, forward_world.y, forward_world.z],
        "distance_to_center": distance_to_center,
        "looking_at_center": dot_product > 0.98,
        "model_center": [model_center.x, model_center.y, model_center.z]
    }
