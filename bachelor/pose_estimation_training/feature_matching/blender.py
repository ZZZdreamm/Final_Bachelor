import bpy
import bpy_extras
from math import radians
from mathutils import Vector, Matrix, Euler
import os
import random
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.blender_utils import (
    clear_all_objects, import_model, setup_look_at_method,
    get_model_bounds, setup_hdri_environment, preload_hdris
)
from common.camera_config import (
    FRONTAL_VIEWS, SIDE_VIEWS, CAMERA_HEIGHTS, PHONE_CAMERA_CONFIGS, get_all_views
)
from common.camera_utils import get_camera_intrinsic_matrix

scene = bpy.context.scene

RENDERS_PER_MODEL = 200

NUM_SURFACE_POINTS = 500

RENDER_SETTINGS = {
    "resolution_x": 1920,
    "resolution_y": 1080,
    "samples": 64,  
}

LIGHTING_CONFIG = {
    "num_hdris": 4,
    "brightness_range": (0.85, 1.15)
}


def sample_surface_points(objs, num_points=500):
    """
    Sample points uniformly from mesh surfaces.
    """
    import bmesh
    
    all_points = []
    all_normals = []
    total_area = 0
    mesh_data = []
    
    # First pass: calculate total surface area
    for obj in objs:
        if obj.type != 'MESH':
            continue
        
        # Create BMesh for area calculation
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.transform(obj.matrix_world)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        
        for face in bm.faces:
            area = face.calc_area()
            total_area += area
            mesh_data.append({
                'verts': [v.co.copy() for v in face.verts],
                'normal': face.normal.copy(),
                'area': area
            })
        bm.free()
    
    if total_area == 0:
        return [], []
    
    for face_data in mesh_data:
        n_samples = int((face_data['area'] / total_area) * num_points)
        
        for _ in range(n_samples):
            r1, r2 = random.random(), random.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            r3 = 1 - r1 - r2
            
            v0, v1, v2 = face_data['verts']
            point = v0 * r1 + v1 * r2 + v2 * r3
            
            all_points.append([point.x, point.y, point.z])
            all_normals.append([face_data['normal'].x, face_data['normal'].y, face_data['normal'].z])
    
    while len(all_points) < num_points and mesh_data:
        face_data = random.choice(mesh_data)
        r1, r2 = random.random(), random.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        r3 = 1 - r1 - r2
        
        v0, v1, v2 = face_data['verts']
        point = v0 * r1 + v1 * r2 + v2 * r3
        
        all_points.append([point.x, point.y, point.z])
        all_normals.append([face_data['normal'].x, face_data['normal'].y, face_data['normal'].z])
    
    return all_points[:num_points], all_normals[:num_points]


def get_camera_extrinsic_matrix(cam):
    """Compute camera extrinsic matrix [R|t]."""
    cam_matrix_world = cam.matrix_world
    world_to_cam = cam_matrix_world.inverted()
    
    blender_to_opencv = Matrix([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    
    extrinsic = blender_to_opencv @ world_to_cam
    
    R = [[extrinsic[i][j] for j in range(3)] for i in range(3)]
    t = [extrinsic[i][3] for i in range(3)]
    
    return {
        "R": R,
        "t": t,
        "matrix_4x4": [[extrinsic[i][j] for j in range(4)] for i in range(4)]
    }


def project_points_to_2d(cam, points_3d, render):
    """Project 3D points to 2D image coordinates."""
    scale = render.resolution_percentage / 100.0
    width = int(render.resolution_x * scale)
    height = int(render.resolution_y * scale)
    
    projected = []
    
    for point in points_3d:
        pt_3d = Vector(point)
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, pt_3d)
        
        pixel_x = co_2d.x * width
        pixel_y = height - (co_2d.y * height)
        
        in_frame = 0 <= pixel_x < width and 0 <= pixel_y < height
        in_front = co_2d.z > 0
        
        visibility = 1 if (in_frame and in_front) else 0
        
        projected.append({
            "x": round(pixel_x, 2),
            "y": round(pixel_y, 2),
            "visible": visibility,
            "depth": round(co_2d.z, 4)
        })
    
    return projected


def setup_camera_for_render(cam, center, size, cam_config, direction, height_offset):
    """Position camera for realistic phone photography."""
    cam.data.lens = cam_config["lens"]
    distance = size * cam_config["distance_factor"]
    distance *= random.uniform(0.97, 1.03)
    
    view_dir = direction.normalized()
    view_dir.z = height_offset
    view_dir.normalize()
    
    cam.location = center + view_dir * distance
    
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


def generate_render_plan(num_renders=200):
    """Generate a weighted random render plan."""
    plan = []
    all_views = get_all_views()
    
    for i in range(num_renders):
        cam_weights = [c["weight"] for c in PHONE_CAMERA_CONFIGS]
        cam_config = random.choices(PHONE_CAMERA_CONFIGS, weights=cam_weights)[0]
        
        view_weights = [v["weight"] for v in all_views]
        view = random.choices(all_views, weights=view_weights)[0]
        
        height_weights = [h["weight"] for h in CAMERA_HEIGHTS]
        height = random.choices(CAMERA_HEIGHTS, weights=height_weights)[0]
        
        hdri_idx = i % LIGHTING_CONFIG["num_hdris"]
        
        plan.append({
            "render_idx": i,
            "hdri_idx": hdri_idx,
            "cam_config": cam_config,
            "direction": view["dir"].copy(),
            "height_z": height["z"],
            "height_name": height["name"]
        })
    
    return plan


def render_model(model_path: str, output_dir: str, hdri_files: list):
    """Render all views of a single model and export pose data."""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")
    
    clear_all_objects()
    imported = import_model(model_path, collection_name="BuildingModel")
    
    mesh_objs = [obj for obj in imported if obj.type == 'MESH']
    if not mesh_objs:
        print("ERROR: No mesh objects found")
        return
    
    bounds = get_model_bounds(mesh_objs)
    center = bounds["center"]
    size = bounds["size"]
    radius = bounds["radius"]
    
    print(f"  Model bounds: center={center}, size={size:.2f}")
    
    print(f"  Sampling {NUM_SURFACE_POINTS} surface points...")
    surface_points, surface_normals = sample_surface_points(mesh_objs, NUM_SURFACE_POINTS)
    print(f"  Sampled {len(surface_points)} points")
    
    if scene.camera is None:
        bpy.ops.object.camera_add()
        scene.camera = bpy.context.object
    cam = scene.camera
    
    render_plan = generate_render_plan(RENDERS_PER_MODEL)
    print(f"  Render plan: {len(render_plan)} images")
    
    images_dir = os.path.join(output_dir, model_name, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    model_metadata = {
        "model_name": model_name,
        "model_path": model_path,
        "bounds": {
            "min": list(bounds["min"]),
            "max": list(bounds["max"]),
            "center": list(center),
            "size": size,
            "radius": radius
        },
        "surface_points_3d": surface_points,
        "surface_normals": surface_normals,
        "num_points": len(surface_points),
        "renders": []
    }
    
    for plan_item in render_plan:
        render_idx = plan_item["render_idx"]
        
        if hdri_files:
            hdri_file = hdri_files[plan_item["hdri_idx"] % len(hdri_files)]
            brightness = random.uniform(*LIGHTING_CONFIG["brightness_range"])
            setup_hdri_environment(hdri_file, brightness)
        
        setup_camera_for_render(
            cam, center, size,
            plan_item["cam_config"],
            plan_item["direction"],
            plan_item["height_z"]
        )
        
        bpy.context.view_layer.update()
        
        K = get_camera_intrinsic_matrix(cam, scene.render)
        extrinsic = get_camera_extrinsic_matrix(cam)
        points_2d = project_points_to_2d(cam, surface_points, scene.render)
        
        render_name = f"render_{render_idx:04d}.png"
        scene.render.filepath = os.path.join(images_dir, render_name)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.film_transparent = False
        bpy.ops.render.render(write_still=True)
        
        render_data = {
            "filename": render_name,
            "render_idx": render_idx,
            "camera_config": plan_item["cam_config"]["name"],
            "focal_length_mm": plan_item["cam_config"]["lens"],
            "direction": [plan_item["direction"].x, plan_item["direction"].y, plan_item["direction"].z],
            "height_name": plan_item["height_name"],
            "intrinsic_K": K,
            "extrinsic_R": extrinsic["R"],
            "extrinsic_t": extrinsic["t"],
            "extrinsic_matrix_4x4": extrinsic["matrix_4x4"],
            "camera_location": list(cam.location),
            "camera_rotation_euler": list(cam.rotation_euler),
            "points_2d": points_2d
        }
        model_metadata["renders"].append(render_data)
        
        if (render_idx + 1) % 20 == 0:
            print(f"  Progress: {render_idx + 1}/{len(render_plan)}")
    
    metadata_path = os.path.join(output_dir, model_name, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"✓ Completed: {len(render_plan)} renders")
    print(f"  Saved to: {output_dir}/{model_name}/")


def render_all_models(models_dir: str, output_dir: str, assets_dir: str = None):
    """
    Render all models in directory.
    """
    scene.render.resolution_x = RENDER_SETTINGS["resolution_x"]
    scene.render.resolution_y = RENDER_SETTINGS["resolution_y"]
    scene.render.resolution_percentage = 100
    
    scene.eevee.taa_render_samples = RENDER_SETTINGS["samples"]
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 1.0
    scene.eevee.use_shadows = True
    
    if hdri_files:
        hdri_files = hdri_files[:LIGHTING_CONFIG["num_hdris"]]
    print(f"Loaded {len(hdri_files)} HDRI files")
    
    # Find all model files
    model_extensions = (".glb", ".gltf", ".fbx")
    model_files = [f for f in os.listdir(models_dir) 
                   if f.lower().endswith(model_extensions)]
    
    print(f"\nFound {len(model_files)} models to process\n")
    
    for idx, filename in enumerate(model_files, 1):
        print(f"\n[{idx}/{len(model_files)}]")
        model_path = os.path.join(models_dir, filename)
        render_model(model_path, output_dir, hdri_files)
    
    print(f"\n{'='*60}")
    print("All models processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    setup_look_at_method()

    MODELS_DIR = "C:/Users/KMult/Desktop/Praca_inzynierska/models/modele_test"
    OUTPUT_DIR = "C:/Users/KMult/Desktop/Praca_inzynierska/models/AR_Renders"
    ASSETS_DIR = "C:/Users/KMult/Desktop/Praca_inzynierska/models/assets"

    render_all_models(MODELS_DIR, OUTPUT_DIR, ASSETS_DIR)