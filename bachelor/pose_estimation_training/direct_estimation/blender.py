import bpy
from math import radians
from mathutils import Vector, Euler
import os
import random
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.blender_utils import (
    clear_all_objects, import_model, setup_look_at_method,
    get_model_bounds, setup_hdri_environment, preload_hdris, get_clay_material
)
from common.camera_config import (
    FRONTAL_VIEWS, SIDE_VIEWS, CAMERA_HEIGHTS, PHONE_CAMERA_CONFIGS
)
from common.camera_utils import (
    get_camera_intrinsic_matrix, setup_camera_for_render,
    get_camera_pose_for_training
)

scene = bpy.context.scene

FRAMING_STRATEGIES = [
    {"name": "CENTERED", "weight": 140, "offset_x": 0.0, "offset_y": 0.0, "zoom_mult": 1.0},
    {"name": "TOP_LEFT", "weight": 10, "offset_x": 0.35, "offset_y": -0.35, "zoom_mult": 1.1},
    {"name": "TOP_RIGHT", "weight": 10, "offset_x": -0.35, "offset_y": -0.35, "zoom_mult": 1.1},
    {"name": "BOTTOM_LEFT", "weight": 10, "offset_x": 0.35, "offset_y": 0.35, "zoom_mult": 1.1},
    {"name": "BOTTOM_RIGHT", "weight": 10, "offset_x": -0.35, "offset_y": 0.35, "zoom_mult": 1.1},
    {"name": "SHIFT_LEFT", "weight": 5, "offset_x": 0.4, "offset_y": 0.0, "zoom_mult": 1.0},
    {"name": "SHIFT_RIGHT", "weight": 5, "offset_x": -0.4, "offset_y": 0.0, "zoom_mult": 1.0},
    {"name": "MACRO_FILL", "weight": 10, "offset_x": 0.0, "offset_y": 0.0, "zoom_mult": 0.6}, 
]

FLAT_COLOR_CHANCE = 0.5 
FLAT_COLOR_RANGE = (0.2, 0.8)

RENDERS_PER_MODEL = 500

TRAIN_SPLIT = 0.8

SEPARATE_BUILDING_FOLDERS = False 

RENDER_SETTINGS = {
    "resolution_x": 1024,
    "resolution_y": 1024,
    "samples": 64,
}

LIGHTING_CONFIG = {
    "num_hdris": 4,
    "brightness_range": (0.85, 1.15)
}


def normalize_model(objs, bounds):
    """
    Center and normalize model to standard coordinate system.
    Important for training: all models should be in same scale/position.
    """
    center = bounds["center"]
    scale_factor = 10.0 / bounds["size"] 
    
    for obj in objs:
        if obj.type == 'MESH':
            obj.location -= center
            obj.scale *= scale_factor
            
    bpy.context.view_layer.update()
    
    return get_model_bounds(objs)


def generate_render_plan(num_renders=1000):
    """Generate a weighted random render plan with train/val split"""
    from common.camera_config import get_all_views

    plan = []
    all_views = get_all_views()
    
    # Calculate split point
    num_train = int(num_renders * TRAIN_SPLIT)
    
    for i in range(num_renders):
        cam_weights = [c["weight"] for c in PHONE_CAMERA_CONFIGS]
        cam_config = random.choices(PHONE_CAMERA_CONFIGS, weights=cam_weights)[0]
        
        view_weights = [v["weight"] for v in all_views]
        view = random.choices(all_views, weights=view_weights)[0]
        
        height_weights = [h["weight"] for h in CAMERA_HEIGHTS]
        height = random.choices(CAMERA_HEIGHTS, weights=height_weights)[0]
        
        framing_weights = [f["weight"] for f in FRAMING_STRATEGIES]
        framing = random.choices(FRAMING_STRATEGIES, weights=framing_weights)[0]
        
        hdri_idx = i % LIGHTING_CONFIG["num_hdris"]
        
        use_flat_color = random.random() < FLAT_COLOR_CHANCE
        
        split = 'train' if i < num_train else 'val'
        
        plan.append({
            "render_idx": i,
            "hdri_idx": hdri_idx,
            "cam_config": cam_config,
            "direction": view["dir"].copy(),
            "height_z": height["z"],
            "height_name": height["name"],
            "framing": framing,
            "use_flat_color": use_flat_color,
            "split": split
        })
    
    return plan


def render_model(model_path: str, output_dir: str, hdri_files: list, building_id: int):
    """Render all views of a single model and export pose data for training"""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {model_name} (Building ID: {building_id})")
    print(f"{'='*60}")
    
    clear_all_objects()
    imported = import_model(model_path, collection_name="BuildingModel")
    
    mesh_objs = [obj for obj in imported if obj.type == 'MESH']
    if not mesh_objs:
        print("ERROR: No mesh objects found")
        return
    
    original_bounds = get_model_bounds(mesh_objs)
    print(f"  Original bounds: center={original_bounds['center']}, size={original_bounds['size']:.2f}")
    
    bounds = normalize_model(mesh_objs, original_bounds)
    center = bounds["center"]  
    size = bounds["size"]      
    
    print(f"  Normalized bounds: center={center}, size={size:.2f}")
    
    if scene.camera is None:
        bpy.ops.object.camera_add()
        scene.camera = bpy.context.object
    cam = scene.camera
    
    render_plan = generate_render_plan(RENDERS_PER_MODEL)
    print(f"  Render plan: {len(render_plan)} images ({int(TRAIN_SPLIT*100)}% train)")
    
    if SEPARATE_BUILDING_FOLDERS:
        images_dir = os.path.join(output_dir, "images", model_name)
    else:
        images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    clay_mat = get_clay_material()
    
    annotations = []
    sample_idx = 0
    
    for plan_item in render_plan:
        render_idx = plan_item["render_idx"]
        
        if hdri_files:
            hdri_file = hdri_files[plan_item["hdri_idx"] % len(hdri_files)]
            brightness = random.uniform(*LIGHTING_CONFIG["brightness_range"])
            setup_hdri_environment(hdri_file, brightness)
        
        if plan_item["use_flat_color"]:
            grey_val = random.uniform(FLAT_COLOR_RANGE[0], FLAT_COLOR_RANGE[1])
            clay_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (grey_val, grey_val, grey_val, 1)
            scene.view_layers[0].material_override = clay_mat
        else:
            scene.view_layers[0].material_override = None

        setup_camera_for_render(
            cam, center, size,
            plan_item["cam_config"],
            plan_item["direction"],
            plan_item["height_z"],
            framing_strategy=plan_item["framing"]
        )
        
        bpy.context.view_layer.update()
        
        pose_data = get_camera_pose_for_training(cam, center)
        K = get_camera_intrinsic_matrix(cam, scene.render)
        
        if not pose_data.get("looking_at_center", True):
            print(f"  WARNING: Camera may not be looking at center for render {render_idx}")
        
        image_filename = f"{model_name}_{sample_idx:06d}.jpg"
        
        if SEPARATE_BUILDING_FOLDERS:
            image_path = os.path.join(images_dir, image_filename)
            relative_image_path = os.path.join(model_name, image_filename)
        else:
            image_path = os.path.join(images_dir, image_filename)
            relative_image_path = image_filename
            
        scene.render.filepath = image_path
        scene.render.image_settings.file_format = 'JPEG'
        scene.render.image_settings.quality = 95
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.film_transparent = False
        bpy.ops.render.render(write_still=True)
        
        annotation = {
            'image_id': sample_idx,
            'image_filename': relative_image_path,  
            'building_id': building_id,
            'building_name': model_name,
            'rotation_quaternion': pose_data['rotation_quaternion'],  # [w, x, y, z]
            'translation': pose_data['translation'],  # [x, y, z]
            'rotation_matrix': pose_data['rotation_matrix'],  # 3x3 matrix
            'camera_intrinsics': K,
            'focal_length_mm': plan_item["cam_config"]["lens"],
            'split': plan_item['split'],  # 'train' or 'val'
            'is_flat_color': plan_item["use_flat_color"],
            'camera_config': plan_item["cam_config"]["name"],
            'framing_strategy': plan_item["framing"]["name"],
            'camera_forward': pose_data['camera_forward'],
            'distance_to_center': pose_data['distance_to_center'],
            'model_center': pose_data['model_center']
        }
        
        annotations.append(annotation)
        sample_idx += 1
        
        if (render_idx + 1) % 50 == 0:
            print(f"  Progress: {render_idx + 1}/{len(render_plan)}")
    
    scene.view_layers[0].material_override = None
    
    print(f"✓ Completed: {len(annotations)} renders")
    
    return annotations, bounds


def render_all_models(models_dir: str, output_dir: str, assets_dir: str = None):
    """
    Render all models and create unified training dataset.
    """
    scene.render.resolution_x = RENDER_SETTINGS["resolution_x"]
    scene.render.resolution_y = RENDER_SETTINGS["resolution_y"]
    scene.render.resolution_percentage = 100
   
    scene.eevee.taa_render_samples = RENDER_SETTINGS["samples"]
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 1.0
    scene.eevee.use_shadows = True
    
    hdri_files = preload_hdris(assets_dir)
    if hdri_files:
        hdri_files = hdri_files[:LIGHTING_CONFIG["num_hdris"]]
    print(f"Loaded {len(hdri_files)} HDRI files")
    
    model_extensions = (".glb", ".gltf", ".fbx")
    model_files = [f for f in os.listdir(models_dir) 
                   if f.lower().endswith(model_extensions)]
    
    print(f"\nFound {len(model_files)} models to process\n")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    building_names = []
    all_annotations = []
    model_bounds_info = {}
    
    for building_id, filename in enumerate(model_files):
        print(f"\n[{building_id + 1}/{len(model_files)}]")
        model_path = os.path.join(models_dir, filename)
        model_name = os.path.splitext(filename)[0]
        building_names.append(model_name)
        
        annotations, bounds = render_model(model_path, output_dir, hdri_files, building_id)
        all_annotations.extend(annotations)
        
        model_bounds_info[model_name] = {
            "center": list(bounds["center"]),
            "size": bounds["size"],
            "radius": bounds["radius"]
        }
    
    dataset_info = {
        'buildings': building_names,
        'num_buildings': len(building_names),
        'samples_per_building': RENDERS_PER_MODEL,
        'image_size': [RENDER_SETTINGS["resolution_x"], RENDER_SETTINGS["resolution_y"]],
        'train_split': TRAIN_SPLIT,
        'model_bounds': model_bounds_info,
        'annotations': all_annotations
    }
    
    dataset_path = os.path.join(output_dir, "annotations", "dataset.json")
    with open(dataset_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n{'='*60}")
    print("All models processed!")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_annotations)}")
    print(f"Buildings: {len(building_names)}")
    print(f"Dataset saved to: {dataset_path}")
    print(f"\nReady for training with train_pose_network.py!")


if __name__ == "__main__":
    setup_look_at_method()

    MODELS_DIR = "C:/Users/KMult/Desktop/Praca_inzynierska/models/modele_test"
    OUTPUT_DIR = "C:/Users/KMult/Desktop/Praca_inzynierska/models/AR_pose2"
    ASSETS_DIR = "C:/Users/KMult/Desktop/Praca_inzynierska/models/assets"

    render_all_models(MODELS_DIR, OUTPUT_DIR, ASSETS_DIR)