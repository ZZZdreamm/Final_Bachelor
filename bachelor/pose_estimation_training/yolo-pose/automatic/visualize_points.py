import bpy
import json
import os
import math
from mathutils import Vector

MODEL_NAME = "Ilmet"  
MODELS_FOLDER = "C:/Users/KMult/Desktop/Praca_inzynierska/models/modele_test/"
JSON_FILE = "C:/Users/KMult/Desktop/Praca_inzynierska/models/modele_test/keypoints/keypoints_all.json"

TARGET_SIZE = 4.0        
DOT_SIZE_RATIO = 0.005      
DOT_COLOR = (1.0, 0.0, 0.0, 1.0) 

def cleanup_scene():
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for col in bpy.data.collections: bpy.data.collections.remove(col)
    for block in bpy.data.meshes:
        if block.users == 0: bpy.data.meshes.remove(block)

def get_red_material():
    mat_name = "Keypoint_Red_Mat"
    if mat_name in bpy.data.materials: return bpy.data.materials[mat_name]
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    shader = nodes.new(type='ShaderNodeEmission')
    shader.inputs[0].default_value = DOT_COLOR
    shader.inputs[1].default_value = 1.0 
    output = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(shader.outputs[0], output.inputs[0])
    mat.diffuse_color = DOT_COLOR
    return mat

def load_and_visualize():
    # 1. Load Data
    if not os.path.exists(JSON_FILE):
        print("Error: JSON file not found.")
        return

    with open(JSON_FILE, 'r') as f: full_data = json.load(f)
    if MODEL_NAME not in full_data:
        print(f"Error: Model '{MODEL_NAME}' not found in JSON.")
        return
    points = full_data[MODEL_NAME]["points"]

    cleanup_scene()

    # 2. Find and Import File
    target_path = None
    for ext in ['.glb', '.gltf']:
        p = os.path.join(MODELS_FOLDER, MODEL_NAME + ext)
        if os.path.exists(p): target_path = p; break
    
    if not target_path:
        print(f"File not found for: {MODEL_NAME}")
        return

    print(f"Importing {MODEL_NAME}...")
    bpy.ops.import_scene.gltf(filepath=target_path)

    # 3. Join Mesh Parts 
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objs: return
    bpy.ops.object.select_all(action='DESELECT')
    for m in mesh_objs: m.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1: bpy.ops.object.join()
    obj = bpy.context.active_object

    # 4. Sync Coordinate Space
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # 5. Normalize Scale
    dim = obj.dimensions
    max_dim = max(dim.x, dim.y, dim.z)
    if max_dim > 0:
        scale_val = TARGET_SIZE / max_dim
        obj.scale = (scale_val, scale_val, scale_val)
        print(f"Scaled model by {scale_val:.4f} to fit {TARGET_SIZE}m view.")

    # 6. Spawn Dots
    viz_col = bpy.data.collections.new("Keypoints_Viz")
    bpy.context.scene.collection.children.link(viz_col)
    
    bpy.ops.mesh.primitive_uv_sphere_add(radius=TARGET_SIZE * DOT_SIZE_RATIO)
    temp_dot = bpy.context.active_object
    dot_mesh = temp_dot.data
    dot_mesh.materials.append(get_red_material())
    bpy.data.objects.remove(temp_dot)

    # Place dots using Matrix World
    mw = obj.matrix_world
    
    for i, p in enumerate(points):
        world_loc = mw @ Vector(p)
        
        dot = bpy.data.objects.new(f"KP_{i}", dot_mesh)
        dot.location = world_loc
        
        dot.parent = obj
        dot.matrix_parent_inverse = mw.inverted()
        
        viz_col.objects.link(dot)

    print(f"Success: Visualized {len(points)} keypoints.")

load_and_visualize()