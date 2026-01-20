import bpy
import bpy_extras
from math import radians, sin, cos, pi
from mathutils import Vector, Matrix, Euler
import os
import random
import json
import math
import glob
import numpy as np

scene = bpy.context.scene

FRONTAL_VIEWS = [
    {"dir": Vector((0, 1, 0)), "weight": 15},
    {"dir": Vector((0.15, 1, 0)), "weight": 12},
    {"dir": Vector((-0.15, 1, 0)), "weight": 12},
    {"dir": Vector((0.3, 0.95, 0)), "weight": 10},
    {"dir": Vector((-0.3, 0.95, 0)), "weight": 10},
    {"dir": Vector((0.5, 0.85, 0)), "weight": 8},
    {"dir": Vector((-0.5, 0.85, 0)), "weight": 8},
    {"dir": Vector((0.7, 0.7, 0)), "weight": 5},
    {"dir": Vector((-0.7, 0.7, 0)), "weight": 5},
]

SIDE_VIEWS = [
    {"dir": Vector((0.9, 0.4, 0)), "weight": 3},
    {"dir": Vector((-0.9, 0.4, 0)), "weight": 3},
]

CAMERA_HEIGHTS = [
    {"z": 0.0, "name": "eye_level", "weight": 50},
    {"z": 0.12, "name": "slightly_up", "weight": 30},
    {"z": -0.08, "name": "slightly_down", "weight": 20},
    {"z": 0.25, "name": "low_angle", "weight": 10},
]

PHONE_CAMERA_CONFIGS = [
    {
        "name": "Normal",
        "lens": 28,
        "distance_factor": 1.4,
        "weight": 70,
        "tilt_range": (-2, 2),
        "position_jitter": 0.02
    },
    {
        "name": "Standard_Close",
        "lens": 28,
        "distance_factor": 1.1,
        "weight": 40,
        "tilt_range": (-3, 3),
        "position_jitter": 0.025
    },
    {
        "name": "Wide",
        "lens": 24,
        "distance_factor": 1.0,
        "weight": 30,
        "tilt_range": (-3, 3),
        "position_jitter": 0.03
    },
    {
        "name": "Telephoto",
        "lens": 52,
        "distance_factor": 2.2,
        "weight": 20,
        "tilt_range": (-1, 1),
        "position_jitter": 0.015
    },
]

FRAMING_STRATEGIES = [
    {"name": "CENTERED", "weight": 200, "offset_x": 0.0, "offset_y": 0.0, "zoom_mult": 1.0},
    
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

def clear_all_objects():
    """Clear all objects except camera"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            continue
        bpy.data.objects.remove(obj, do_unlink=True)
    for col in bpy.data.collections:
        bpy.data.collections.remove(col)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)

def import_model(model_path: str, collection_name="ImportedModel"):
    """Import 3D model (GLB, GLTF, or FBX)"""
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
        else:
            print(f"Unsupported format: {ext}")
            return []
    except Exception as e:
        print(f"Import error: {e}")
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
    """Point camera at target location"""
    direction = target - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

bpy.types.Object.look_at = look_at

def get_model_bounds(objs):
    """Get bounding box of all mesh objects"""
    min_corner = Vector((float('inf'), float('inf'), float('inf')))
    max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    for obj in objs:
        if obj.type == 'MESH':
            for v in obj.bound_box:
                world_v = obj.matrix_world @ Vector(v)
                min_corner = Vector(map(min, zip(min_corner, world_v)))
                max_corner = Vector(map(max, zip(max_corner, world_v)))
    
    center = (min_corner + max_corner) / 2
    size = (max_corner - min_corner).length
    radius = max((max_corner - min_corner)) / 2
    
    return {
        "min": min_corner,
        "max": max_corner,
        "center": center,
        "size": size,
        "radius": radius
    }

def sample_surface_points(objs, num_points=500):
    """Sample points uniformly from mesh surfaces."""
    import bmesh
    
    all_points = []
    all_normals = []
    total_area = 0
    mesh_data = []
    
    for obj in objs:
        if obj.type != 'MESH':
            continue
        
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

def get_camera_intrinsic_matrix(cam, render):
    """Compute camera intrinsic matrix K."""
    scale = render.resolution_percentage / 100.0
    width = render.resolution_x * scale
    height = render.resolution_y * scale
    
    if cam.data.sensor_fit == 'VERTICAL':
        sensor_height = cam.data.sensor_height
        fx = fy = (height * cam.data.lens) / sensor_height
    else:
        sensor_width = cam.data.sensor_width
        fx = fy = (width * cam.data.lens) / sensor_width
    
    cx = width / 2.0
    cy = height / 2.0
    
    K = [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ]
    
    return K

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

def get_clay_material():
    """Get or create a generic flat/clay material."""
    mat_name = "Override_Clay_Material"
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]
    
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.9
    bsdf.inputs['Specular IOR Level'].default_value = 0.1
    
    out = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def setup_camera_for_render(cam, center, size, cam_config, direction, height_offset, framing_strategy):
    """Position camera for realistic phone photography with framing variations."""
    cam.data.lens = cam_config["lens"]
    
    base_distance = size * cam_config["distance_factor"]
    final_distance = base_distance * framing_strategy["zoom_mult"]
    final_distance *= random.uniform(0.97, 1.03)
    
    view_dir = direction.normalized()
    view_dir.z = height_offset
    view_dir.normalize()
    
    cam.location = center + view_dir * final_distance
    cam.look_at(center)
    
    bpy.context.view_layer.update()
    
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
    
    pos_jitter = cam_config["position_jitter"] * size
    offset = Vector((
        random.uniform(-pos_jitter, pos_jitter),
        random.uniform(-pos_jitter, pos_jitter),
        random.uniform(-pos_jitter * 0.5, pos_jitter * 0.5)
    ))
    cam.location += offset
    
    tilt_range = cam_config["tilt_range"]
    cam.rotation_euler.rotate(Euler((
        radians(random.uniform(tilt_range[0], tilt_range[1])),
        0,
        radians(random.uniform(tilt_range[0] * 0.5, tilt_range[1] * 0.5))
    ), 'XYZ'))

def setup_hdri_environment(hdri_path, brightness=1.0):
    """Setup HDRI environment lighting"""
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
        
        node_env.texture_mapping.rotation[2] = random.uniform(0, 2 * math.pi)
        node_bg.inputs["Strength"].default_value = brightness
        
    except Exception as e:
        pass

def generate_render_plan(num_renders=300):
    """Generate a weighted random render plan."""
    plan = []
    all_views = FRONTAL_VIEWS + SIDE_VIEWS
    
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
        
        plan.append({
            "render_idx": i,
            "hdri_idx": hdri_idx,
            "cam_config": cam_config,
            "direction": view["dir"].copy(),
            "height_z": height["z"],
            "height_name": height["name"],
            "framing": framing,
            "use_flat_color": use_flat_color
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
        return

    bounds = get_model_bounds(mesh_objs)
    center = bounds["center"]
    size = bounds["size"]
    radius = bounds["radius"]

    surface_points, surface_normals = sample_surface_points(mesh_objs, NUM_SURFACE_POINTS)
    
    if scene.camera is None:
        bpy.ops.object.camera_add()
        scene.camera = bpy.context.object
    cam = scene.camera

    render_plan = generate_render_plan(RENDERS_PER_MODEL)
    
    images_dir = os.path.join(output_dir, model_name)
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
    
    clay_mat = get_clay_material()

    total_renders = len(render_plan)

    for plan_item in render_plan:
        render_idx = plan_item["render_idx"]

        print(f"  Rendering {render_idx + 1}/{total_renders}...", end='\r')

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
            plan_item["framing"]
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
            "framing_strategy": plan_item["framing"]["name"],
            "is_flat_color": plan_item["use_flat_color"],
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

    scene.view_layers[0].material_override = None

    metadata_path = os.path.join(output_dir, model_name, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)

    print(f"\n  Completed: {total_renders} renders saved to {images_dir}")

def preload_hdris(assets_dir):
    """Load HDRI files from assets directory"""
    hdris = []
    if assets_dir and os.path.exists(assets_dir):
        hdri_root = os.path.join(assets_dir, "hdri")
        if os.path.exists(hdri_root):
            hdris = glob.glob(os.path.join(hdri_root, "*.exr")) + \
                    glob.glob(os.path.join(hdri_root, "*.hdr"))
    return hdris

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
    
    hdri_files = preload_hdris(assets_dir)
    if hdri_files:
        hdri_files = hdri_files[:LIGHTING_CONFIG["num_hdris"]]

    model_extensions = (".glb", ".gltf", ".fbx")
    model_files = [f for f in os.listdir(models_dir)
                   if f.lower().endswith(model_extensions)]

    for idx, filename in enumerate(model_files, 1):
        print(f"\nModel {idx}/{len(model_files)}: {filename}")
        model_path = os.path.join(models_dir, filename)
        render_model(model_path, output_dir, hdri_files)

    print(f"\n{'='*60}")
    print(f"All {len(model_files)} models completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    MODELS_DIR = "C:/Users/KMult/Desktop/Praca_inzynierska/models/modele_rdy"
    OUTPUT_DIR = "C:/Users/KMult/Desktop/Praca_inzynierska/models/AR_Renders5"
    ASSETS_DIR = "C:/Users/KMult/Desktop/Praca_inzynierska/models/assets"
    
    render_all_models(MODELS_DIR, OUTPUT_DIR, ASSETS_DIR)