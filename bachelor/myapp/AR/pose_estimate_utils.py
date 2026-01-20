from typing import Dict, List, Tuple
import os
import io

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import cv2
import numpy as np
import trimesh
import pyrender

from myapp.AR.model_template import TRAINED_CLASSIFICATION_MODEL


def get_camera_matrix(img_w, img_h):
    """
    Estimates the Intrinsic Camera Matrix (K) based on image size and focal length.
    """
    FOCAL_LENGTH_MM = 28.0 
    SENSOR_WIDTH_MM = 36.0
    
    fx = (FOCAL_LENGTH_MM * img_w) / SENSOR_WIDTH_MM
    fy = fx
    cx = img_w / 2.0
    cy = img_h / 2.0
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)
    
    
def get_common_camera_intrinsics(img_width: int, img_height: int) -> List[Dict]:
    """
    Generate common camera intrinsic configurations for testing.

    Returns list of camera configs to try, ordered from most to least common.
    """
    configs = []

    for fov in [60, 65, 70, 75]:
        focal_length = (img_width / 2) / np.tan(np.radians(fov / 2))
        configs.append({
            'name': f'Smartphone (FOV {fov}°)',
            'fx': focal_length,
            'fy': focal_length,
            'cx': img_width / 2,
            'cy': img_height / 2,
            'priority': 1
        })

    for fov in [80, 85, 90]:
        focal_length = (img_width / 2) / np.tan(np.radians(fov / 2))
        configs.append({
            'name': f'Wide-angle (FOV {fov}°)',
            'fx': focal_length,
            'fy': focal_length,
            'cx': img_width / 2,
            'cy': img_height / 2,
            'priority': 2
        })

    for fov in [45, 50, 55]:
        focal_length = (img_width / 2) / np.tan(np.radians(fov / 2))
        configs.append({
            'name': f'DSLR Standard (FOV {fov}°)',
            'fx': focal_length,
            'fy': focal_length,
            'cx': img_width / 2,
            'cy': img_height / 2,
            'priority': 3
        })

    training_fx = 796.4444444444445
    training_width = 1024
    scale_x = img_width / training_width
    scale_y = img_height / img_height

    configs.append({
        'name': 'Training Camera (scaled)',
        'fx': training_fx * scale_x,
        'fy': training_fx * scale_y,
        'cx': img_width / 2,
        'cy': img_height / 2,
        'priority': 0
    })

    for aspect_ratio in [1.1, 0.9]:
        focal_length = (img_width / 2) / np.tan(np.radians(65 / 2))
        configs.append({
            'name': f'Non-square pixels (ratio {aspect_ratio:.1f})',
            'fx': focal_length,
            'fy': focal_length * aspect_ratio,
            'cx': img_width / 2,
            'cy': img_height / 2,
            'priority': 4
        })

    configs.sort(key=lambda x: x['priority'])

    return configs


def compute_rendering_score(rendered: np.ndarray, depth: np.ndarray, confidence: float) -> float:
    """
    Compute a score for how "good" a rendering is.
    Higher score = better match.

    Heuristics:
    - Depth coverage (how much of image has rendered content)
    - Depth variance (not all at same distance)
    - Model confidence
    """
    h, w = depth.shape
    total_pixels = h * w

    rendered_pixels = np.sum(depth > 0)
    coverage = rendered_pixels / total_pixels

    if coverage < 0.05:
        coverage_score = coverage * 2
    elif coverage > 0.6:
        coverage_score = 1.0 - coverage
    else:
        coverage_score = min(coverage * 2.5, 1.0)

    if rendered_pixels > 100:
        valid_depth = depth[depth > 0]
        depth_std = np.std(valid_depth)
        depth_mean = np.mean(valid_depth)
        depth_variance_score = min(depth_std / (depth_mean + 1e-6), 1.0)
    else:
        depth_variance_score = 0.0

    if rendered_pixels > 100:
        min_depth = np.min(valid_depth)
        max_depth = np.max(valid_depth)

        if 0.5 < min_depth < 10 and max_depth < 100:
            depth_range_score = 1.0
        else:
            depth_range_score = 0.5
    else:
        depth_range_score = 0.0

    confidence_score = confidence

    total_score = (
        coverage_score * 0.3 +
        depth_variance_score * 0.2 +
        depth_range_score * 0.2 +
        confidence_score * 0.3
    )

    return total_score


def render_with_intrinsics(
    img: np.ndarray,
    mesh: trimesh.Trimesh,
    R_opencv: np.ndarray,
    t_opencv: np.ndarray,
    camera_config: Dict
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Render mesh with specific camera intrinsics.

    Returns:
        (color, depth, score)
    """
    h_img, w_img = img.shape[:2]

    fx = camera_config['fx']
    fy = camera_config['fy']
    cx = camera_config['cx']
    cy = camera_config['cy']

    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R_opencv
    view_matrix[:3, 3] = t_opencv.squeeze()

    cam_pose_cv = np.linalg.inv(view_matrix)

    cv_to_gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
    final_pose = cam_pose_cv @ cv_to_gl

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.8, 0.8, 0.8])
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    cam = pyrender.IntrinsicsCamera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        znear=0.05,
        zfar=1000.0
    )

    scene.add(cam, pose=final_pose)

    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1000.0)
    scene.add(light, pose=final_pose)

    r = pyrender.OffscreenRenderer(w_img, h_img)
    try:
        color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        r.delete()

    score = compute_rendering_score(color, depth, 1.0)

    return color, depth, score


def process_image_heuristic(img_bytes: bytes, model_bytes: bytes, model_name: str) -> bytes:
    """
    Process image with DL pose prediction using the SAME coordinate transforms as PnP.
    This ensures consistency between the old PnP pipeline and new DL pipeline.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h_img, w_img = img.shape[:2]

    debug_img = img.copy()

    model_file = io.BytesIO(model_bytes)

    print("Running DL Pose Prediction...")

    pose_estimator = TRAINED_CLASSIFICATION_MODEL['pose_model']
    pose_result = pose_estimator.predict_pose(img)

    R_blender = pose_result['rotation_matrix']
    t_blender = pose_result['translation']
    confidence = pose_result['confidence']

    if confidence < 0.2:
        return img_bytes

    blender_to_opencv = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])

    R_opencv = blender_to_opencv @ R_blender @ blender_to_opencv.T

    t_opencv = blender_to_opencv @ t_blender

    cv2.putText(debug_img, f"Conf: {confidence:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite("debug_dl_prediction.jpg", debug_img)

    model_file.seek(0)
    mesh_final = trimesh.load(model_file, file_type='glb', force='mesh')

    mesh_size = np.linalg.norm(mesh_final.bounds[1] - mesh_final.bounds[0])
    mesh_center = (mesh_final.bounds[0] + mesh_final.bounds[1]) / 2

    target_size = 10.0
    scale_factor = target_size / mesh_size

    mesh_final.apply_translation(-mesh_center)
    mesh_final.apply_scale(scale_factor)

    rot_matrix = np.eye(4)
    rot_matrix[0:3, 0:3] = blender_to_opencv
    mesh_final.apply_transform(rot_matrix)

    camera_configs = get_common_camera_intrinsics(w_img, h_img)[:3]

    best_score = -1
    best_rendering = None

    for config in camera_configs:
        try:
            color, depth, score = render_with_intrinsics(
                img, mesh_final, R_opencv, t_opencv, config
            )
            if score * confidence > best_score:
                best_score = score * confidence
                best_rendering = (color, depth)
        except:
            continue

    if best_rendering is None:
        return img_bytes

    color, depth = best_rendering

    if color.shape[2] == 3:
        mask = (depth > 0).astype(np.uint8) * 255
        color = np.dstack((color, mask))

    cv2.imwrite("debug_base.jpg", img)

    if color.shape[2] == 4:
        alpha = color[:, :, 3] / 255.0
        render_bgr = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)

        for c in range(3):
            img[:, :, c] = (1.0 - alpha) * img[:, :, c] + alpha * render_bgr[:, :, c]

    final_debug_img = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_result.jpg", img)
    cv2.imwrite("debug_render_only.jpg", final_debug_img)

    _, encoded_img = cv2.imencode('.jpg', img)
    return encoded_img.tobytes()