"""
Shared Blender utilities for rendering and scene management.
Used by both direct_estimation and feature_matching approaches.
"""

import bpy
import os
import random
import math
import glob
from mathutils import Vector


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
    scene = bpy.context.scene
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
    """Point camera at target location"""
    direction = target - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()


def setup_look_at_method():
    """Add look_at method to Blender Object class"""
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
        print(f"Failed to load HDRI {hdri_path}: {e}")


def preload_hdris(assets_dir):
    """Load HDRI files from assets directory"""
    hdris = []
    if assets_dir and os.path.exists(assets_dir):
        hdri_root = os.path.join(assets_dir, "hdri")
        if os.path.exists(hdri_root):
            hdris = glob.glob(os.path.join(hdri_root, "*.exr")) + \
                    glob.glob(os.path.join(hdri_root, "*.hdr"))
    return hdris


def get_clay_material():
    """Get or create a generic flat/clay material for domain randomization"""
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
