import bpy
import bpy_extras
from math import radians, sin, cos, pi
from mathutils import Vector, Matrix, Euler
import os
import random
import json
import math
import glob

scene = bpy.context.scene

DIRECTIONS = [] 
CLOSEUP_DIRECTIONS = [
    Vector((0, 1, 0)), 
    
    Vector((0.2, 1, 0)), Vector((-0.2, 1, 0)),
    
    Vector((0.35, 0.95, 0)), Vector((-0.35, 0.95, 0)),
    Vector((0.5, 1, 0)), Vector((-0.5, 1, 0)),
    
    Vector((0.8, 0.8, 0)), Vector((-0.8, 0.8, 0)),
]

SIDE_DIRECTIONS = [
    Vector((1.0, 0.1, 0)),   # Right Side
    Vector((-1.0, 0.1, 0)),  # Left Side
    Vector((0.95, 0.3, 0)),  # Right Side
    Vector((-0.95, 0.3, 0)), # Left Side 
]

HUMAN_PERSPECTIVE_DIRECTIONS = [
    Vector((1, 1, 0.1)), Vector((-1, 1, 0.1)), Vector((0, 1, 0.15)), 
    Vector((0.5, 0.8, 0.1)), Vector((-0.5, 0.8, 0.1)),
]
Z_POSITIONS = [0, -0.1, 0.15]

PHONE_VIEW_CONFIGS = [
    {"name": "Normal", "distance_factor": 1.0, "lens": 28, "random_tilt": True},
    {"name": "Distant", "distance_factor": 1.4, "lens": 28, "random_tilt": True},
    {"name": "HumanPerspective", "distance_factor": 0.85, "lens": 24, "random_tilt": True},
    {"name": "LowAngle", "distance_factor": 0.6, "lens": 18, "random_tilt": True},
    {"name": "SideProfile", "distance_factor": 1.1, "lens": 35, "random_tilt": True},
]

SCATTER_CONFIG = {
    "num_nature": 2,              
    "num_distractors": 2,           
    "nature_scale": (0.4, 0.8),      
    "distractor_scale": (0.7, 1.0), 
    "safe_margin_nature": (0.5, 1.0), 
    "safe_margin_build": (0.7, 1.2)
}


def create_debug_material(name, color):
    if name in bpy.data.materials: return bpy.data.materials[name]
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = False
    mat.diffuse_color = color
    return mat

def clear_all_objects():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA': continue
        bpy.data.objects.remove(obj, do_unlink=True)
    for col in bpy.data.collections:
        bpy.data.collections.remove(col)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0: bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if mat.users == 0: bpy.data.materials.remove(mat)

def clear_noise_collections():
    for col_name in ["NatureCollection", "DistractorsCollection"]:
        if col_name in bpy.data.collections:
            col = bpy.data.collections[col_name]
            for obj in list(col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)

def import_model(model_path: str, collection_name="ImportedGLTF"):
    ext = os.path.splitext(model_path)[1].lower()
    if collection_name not in bpy.data.collections:
        new_col = bpy.data.collections.new(collection_name)
        scene.collection.children.link(new_col)
    target_col = bpy.data.collections[collection_name]
    
    try:
        if ext in [".glb", ".gltf"]: 
            bpy.ops.import_scene.gltf(filepath=model_path)
        elif ext == ".fbx": 
            bpy.ops.import_scene.fbx(filepath=model_path)
        else: return []
    except Exception as e:
        print(f"Failed to import {model_path}: {e}")
        return []

    imported = bpy.context.selected_objects
    for obj in imported:
        if obj.name not in target_col.objects:
            target_col.objects.link(obj)
        for col in obj.users_collection:
            if col != target_col:
                col.objects.unlink(obj)
    return imported

def look_at(cam, target):
    direction = target - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
bpy.types.Object.look_at = look_at

def get_max_dimension(objs):
    min_corner = Vector((float('inf'), float('inf'), float('inf')))
    max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))
    has_mesh = False
    for obj in objs:
        if obj.type == 'MESH':
            has_mesh = True
            for v in obj.bound_box:
                world_v = obj.matrix_world @ Vector(v)
                min_corner = Vector(map(min, zip(min_corner, world_v)))
                max_corner = Vector(map(max, zip(max_corner, world_v)))
    if not has_mesh: return 1.0 
    size_vector = (max_corner - min_corner)
    return max(size_vector.x, size_vector.y, size_vector.z) / 2.0

def get_scene_keypoints(objs, custom_keypoints_list=None):
    min_corner = Vector((float('inf'), float('inf'), float('inf')))
    max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))
    has_objects = False
    for obj in objs:
        has_objects = True
        for v in obj.bound_box:
            world_v = obj.matrix_world @ Vector(v)
            min_corner = Vector(map(min, zip(min_corner, world_v)))
            max_corner = Vector(map(max, zip(max_corner, world_v)))
    if not has_objects: return [], Vector((0,0,0)), 0, 0, None, None

    center = (min_corner + max_corner) / 2
    size_vector = (max_corner - min_corner)
    size = size_vector.length
    size_z = size_vector.z
    points = []
    
    if custom_keypoints_list and len(custom_keypoints_list) > 0:
        target_obj = objs[0] 
        for kp in custom_keypoints_list:
            local_vec = Vector(kp[:3]) 
            world_vec = target_obj.matrix_world @ local_vec
            points.append(world_vec)
    return points, center, size, size_z, min_corner, max_corner

def calculate_visibility_raycast(scene, cam, points_3d, target_objects):
    vis_flags = []
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_loc = cam.matrix_world.translation
    
    for pt in points_3d:
        direction = pt - cam_loc
        dist_to_pt = direction.length
        direction.normalize()
        is_hit, hit_loc, _, _, hit_obj, _ = scene.ray_cast(depsgraph, cam_loc, direction)
        
        if not is_hit:
            vis_flags.append(2)
            continue
        dist_to_hit = (hit_loc - cam_loc).length
        if dist_to_hit >= (dist_to_pt - 0.1):
            vis_flags.append(2) 
        else:
            physical_gap = (hit_loc - pt).length
            if physical_gap < 0.2:
                vis_flags.append(2) 
            else:
                vis_flags.append(1) 
    return vis_flags

def get_2d_corners_with_visibility(scene, cam, corners_3d, vis_flags):
    render = scene.render
    scale = render.resolution_percentage / 100.0
    width = int(render.resolution_x * scale)
    height = int(render.resolution_y * scale)
    corners_2d = []
    for i, corner in enumerate(corners_3d):
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, corner)
        pixel_x = co_2d.x * width
        pixel_y = height - (co_2d.y * height)
        final_vis = vis_flags[i]
        corners_2d.append([round(pixel_x, 2), round(pixel_y, 2), final_vis])
    return corners_2d

def setup_camera(custom_keypoints=None):
    if "HeroCollection" not in bpy.data.collections: return None
    objs = bpy.data.collections["HeroCollection"].objects
    if not objs: return None
    
    corners, center, size, size_z, min_c, max_c = get_scene_keypoints(objs, custom_keypoints)

    if scene.camera is None:
        bpy.ops.object.camera_add()
        scene.camera = bpy.context.object
    cam = scene.camera
    cam.location = center + Vector((0, -size * 1.2, size * 0.25))
    cam.look_at(center)
    cam.data.lens = 28
    return center, size, size_z, cam, corners, min_c, max_c

def set_camera_phone_view(cam, center, size, config, direction):
    cam.data.lens = config["lens"]
    distance = size * config["distance_factor"]
    cam.location = center + direction.normalized() * distance
    if config["random_tilt"]:
        offset_factor = size * random.uniform(0.01, 0.05) 
        cam.location += Vector((random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1))).normalized() * offset_factor
    cam.look_at(center)
    if config["random_tilt"]:
        cam.rotation_euler.rotate(Euler((radians(random.uniform(-3, 3)), 0, radians(random.uniform(-2, 2))), 'XYZ'))

def setup_hdri_environment(hdri_path):
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    
    node_bg = nodes.new(type='ShaderNodeBackground')
    node_env = nodes.new(type='ShaderNodeTexEnvironment')
    node_out = nodes.new(type='ShaderNodeOutputWorld')
    
    try:
        img = bpy.data.images.load(hdri_path)
        node_env.image = img
        links.new(node_env.outputs["Color"], node_bg.inputs["Color"])
        links.new(node_bg.outputs["Background"], node_out.inputs["Surface"])
        node_env.texture_mapping.rotation[2] = random.uniform(0, 6.28)
        print(f"Environment set to: {os.path.basename(hdri_path)}")
    except Exception as e:
        print(f"Failed to load HDRI {hdri_path}: {e}")
        
    for obj in bpy.data.objects:
         if obj.name.startswith("ENV_LIGHT_SUN"): bpy.data.objects.remove(obj)

def get_safe_angle_for_front_view():
    angle_deg = random.uniform(160, 380)
    return radians(angle_deg % 360)

def scatter_assets(assets_list, center, min_radius, max_radius, count, scale_range, col_name, reference_size):
    if not assets_list: return
    
    for _ in range(count):
        chosen_file = random.choice(assets_list)
        objs = import_model(chosen_file, collection_name=col_name)
        if not objs: continue
        
        noise_size = get_max_dimension(objs)
        if noise_size <= 0.0001: noise_size = 1.0 
        
        config_scale_factor = random.uniform(scale_range[0], scale_range[1])
        final_scale_multiplier = (reference_size / noise_size) * config_scale_factor

        angle = get_safe_angle_for_front_view()
        
        dist = random.uniform(min_radius, max_radius)
        x = center.x + math.cos(angle) * dist
        y = center.y + math.sin(angle) * dist
        rot_z = random.uniform(0, 2 * math.pi)
        
        for obj in objs:
            if obj.parent is None:
                obj.location = Vector((x, y, 0))
                obj.rotation_euler.z = rot_z
                obj.scale = obj.scale * final_scale_multiplier

def render_single_view(output_dir, render_name, camera_details, direction, metadata_list, phone_config=None, distance_factor=2.0, random_offset=False, add_tilt=False):
    center, size, size_z, cam, corners_3d, min_c, max_c = camera_details
    center_loc = center.copy()
    
    if phone_config:
        set_camera_phone_view(cam, center_loc, size, phone_config, direction)
    else:
        cam.location = center_loc + direction.normalized() * size * distance_factor
        cam.look_at(center_loc)

    bpy.context.view_layer.update()

    target_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    vis_flags = calculate_visibility_raycast(bpy.context.scene, cam, corners_3d, target_objs)
    corners_2d = get_2d_corners_with_visibility(bpy.context.scene, cam, corners_3d, vis_flags)

    metadata_list.append({
        "filename": render_name,
        "corners_2d": corners_2d 
    })

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = False 
    
    os.makedirs(output_dir, exist_ok=True)
    scene.render.filepath = os.path.join(output_dir, render_name)
    bpy.ops.render.render(write_still=True)

def preload_assets(assets_dir, models_dir):
    data = {
        "hdris": ["default"],
        "nature": [],
        "distractors": []
    }
    
    if assets_dir and os.path.exists(assets_dir):
        hdri_root = os.path.join(assets_dir, "hdri")
        if os.path.exists(hdri_root):
            files = glob.glob(os.path.join(hdri_root, "*.exr")) + glob.glob(os.path.join(hdri_root, "*.hdr"))
            if files:
                data["hdris"] = files
        
        nature_root = os.path.join(assets_dir, "nature")
        if os.path.exists(nature_root):
            data["nature"] = glob.glob(os.path.join(nature_root, "*.glb"))

    if models_dir and os.path.exists(models_dir):
        data["distractors"] = glob.glob(os.path.join(models_dir, "*.glb"))
        
    print(f"Preloaded Assets: {len(data['hdris'])} HDRIs, {len(data['nature'])} Nature, {len(data['distractors'])} Distractors")
    return data

def render_model_all_views(model_path: str, output_dir: str, rotate_y: bool, preloaded_assets, model_keypoints=None):
    clear_all_objects()
    import_model(model_path, collection_name="HeroCollection") 
    
    hero_objs = [obj for obj in bpy.data.collections["HeroCollection"].objects if obj.type == 'MESH']
    if not hero_objs: return
    
    min_c = Vector((float('inf'), float('inf'), float('inf')))
    max_c = Vector((float('-inf'), float('-inf'), float('-inf')))
    for obj in hero_objs:
        for v in obj.bound_box:
            world_v = obj.matrix_world @ Vector(v)
            min_c = Vector(map(min, zip(min_c, world_v)))
            max_c = Vector(map(max, zip(max_c, world_v)))
    
    hero_radius = max((max_c.x - min_c.x), (max_c.y - min_c.y), (max_c.z - min_c.z)) / 2.0
    center_pt = (min_c + max_c) / 2.0
    
    camera_details = setup_camera(custom_keypoints=model_keypoints)
    if not camera_details: return
    
    model_metadata = []

    for hdri_file in preloaded_assets["hdris"]:
        hdri_name = "Default"
        if hdri_file != "default":
            setup_hdri_environment(hdri_file)
            hdri_name = os.path.splitext(os.path.basename(hdri_file))[0]
        
        if preloaded_assets["nature"] or preloaded_assets["distractors"]:
            clear_noise_collections()
            curr_margin_nature = random.uniform(SCATTER_CONFIG["safe_margin_nature"][0], SCATTER_CONFIG["safe_margin_nature"][1])
            curr_margin_build = random.uniform(SCATTER_CONFIG["safe_margin_build"][0], SCATTER_CONFIG["safe_margin_build"][1])
            
            scatter_assets(preloaded_assets["nature"], center_pt, 
                           hero_radius * curr_margin_nature, 
                           hero_radius * 1.6, 
                           SCATTER_CONFIG["num_nature"], 
                           SCATTER_CONFIG["nature_scale"], "NatureCollection", hero_radius)
                           
            valid_distractors = [f for f in preloaded_assets["distractors"] if os.path.abspath(f) != os.path.abspath(model_path)]
            scatter_assets(valid_distractors, center_pt,
                           hero_radius * curr_margin_build, 
                           hero_radius * 2.5, 
                           SCATTER_CONFIG["num_distractors"], 
                           SCATTER_CONFIG["distractor_scale"], "DistractorsCollection", hero_radius)

        for phone_config in PHONE_VIEW_CONFIGS:
            
            if phone_config["name"] == "SideProfile":
                dirs = SIDE_DIRECTIONS
            elif phone_config["name"] == "HumanPerspective":
                dirs = HUMAN_PERSPECTIVE_DIRECTIONS
            else:
                dirs = CLOSEUP_DIRECTIONS
            
            if phone_config["name"] == "LowAngle":
                z_pos_list = [-0.3, -0.4, -0.5]
            elif phone_config["name"] == "HumanPerspective":
                z_pos_list = [0]
            else:
                z_pos_list = Z_POSITIONS
            
            for z_p in z_pos_list:
                for d_idx, d in enumerate(dirs):
                    curr_d = d.copy()
                    curr_d.z = z_p
                    if rotate_y: curr_d.y *= -1
                    r_name = f"render_{hdri_name}_{phone_config['name']}_{z_p}_{d_idx}.png"
                    render_single_view(output_dir, r_name, camera_details, curr_d, model_metadata, phone_config=phone_config)

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(model_metadata, f, indent=4)

def render_folder(models_dir, output_dir, assets_dir):
    assets_data = preload_assets(assets_dir, models_dir)
    semantic_data = {}
    json_path = os.path.join(models_dir, "keypoints/keypoints_all.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                semantic_data = json.load(f)
        except Exception as e:
            print(f"Error loading metadata.json: {e}")

    for f in os.listdir(models_dir):
        if f.lower().endswith((".glb", ".gltf", ".fbx")):
            model_name = os.path.splitext(f)[0]
            model_kps = None
            if model_name in semantic_data:
                model_kps = semantic_data[model_name].get("points", None)
            
            render_model_all_views(
                os.path.join(models_dir, f), 
                os.path.join(output_dir, model_name), 
                f.lower().startswith("y2"),
                preloaded_assets=assets_data, 
                model_keypoints=model_kps
            )

if __name__ == "__main__":
    MODELS_DIR = "C:/Users/KMult/Desktop/Praca_inzynierska/models/modele_test"
    OUTPUT_DIR = "C:/Users/KMult/Desktop/Praca_inzynierska/models/BlenderRenders300"
    ASSETS_DIR = "C:/Users/KMult/Desktop/Praca_inzynierska/models/assets"
    
    render_folder(MODELS_DIR, OUTPUT_DIR, ASSETS_DIR)