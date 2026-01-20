import bpy
import bmesh
import json
import random
import os
from mathutils import Vector

INPUT_FOLDER = "C:/Users/KMult/Desktop/Praca_inzynierska/models/modele_test/"
OUTPUT_FILE = "C:/Users/KMult/Desktop/Praca_inzynierska/models/modele_test/keypoints/keypoints_all.json"
NUM_POINTS = 60
USE_SAMPLING = True
SAMPLE_LIMIT = 10000
HARRIS_RATIO = 0.2     
MIN_DIST_RATIO = 0.05  

def cleanup_scene():
    """Removes all objects from the scene to prepare for the next import."""
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
        
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)


def get_model_scale(verts):
    if not verts: return 1.0
    min_v = Vector((min(v.x for v in verts), min(v.y for v in verts), min(v.z for v in verts)))
    max_v = Vector((max(v.x for v in verts), max(v.y for v in verts), max(v.z for v in verts)))
    diagonal = (max_v - min_v).length
    return diagonal if diagonal > 0 else 1.0

def get_directional_extremes(verts):
    if not verts: return []
    seeds = []
    directions = [
        Vector((1, 0, 0)), Vector((-1, 0, 0)),
        Vector((0, 1, 0)), Vector((0, -1, 0)),
        Vector((0, 0, 1)), Vector((0, 0, -1))
    ]
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                directions.append(Vector((x, y, z)).normalized())

    for direction in directions:
        extreme_v = max(verts, key=lambda v: v.dot(direction))
        seeds.append(extreme_v)
    return seeds

def get_curvature_points(bm, count, exclude_points, min_dist):
    candidates = []
    min_dist_sq = min_dist * min_dist
    
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    
    for v in bm.verts:
        if len(v.link_faces) < 2: continue
        avg_normal = Vector((0,0,0))
        for f in v.link_faces:
            avg_normal += f.normal
        avg_normal /= len(v.link_faces)
        score = 1.0 - avg_normal.length
        if score > 0.02: 
            candidates.append((v.co.copy(), score))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = []
    
    for cand_co, score in candidates:
        if len(selected) >= count: break
        is_far_enough = True
        for p in exclude_points:
            if (cand_co - p).length_squared < min_dist_sq:
                is_far_enough = False; break
        if not is_far_enough: continue
        for p in selected:
            if (cand_co - p).length_squared < min_dist_sq:
                is_far_enough = False; break
        if is_far_enough:
            selected.append(cand_co)
    return selected

def get_hybrid_point_sampling(bm, total_points, sample_limit=5000):
    verts = [v.co.copy() for v in bm.verts]
    if not verts: return []

    diag = get_model_scale(verts)
    dynamic_min_dist = diag * MIN_DIST_RATIO

    # 1. Directional
    dir_seeds = get_directional_extremes(verts)
    unique_seeds = list({tuple(v) for v in dir_seeds})
    selected_co = [Vector(v) for v in unique_seeds]

    # 2. Curvature
    target_harris_count = int(total_points * HARRIS_RATIO)
    if len(selected_co) < total_points:
        harris_points = get_curvature_points(bm, target_harris_count, selected_co, dynamic_min_dist)
        selected_co.extend(harris_points)

    # 3. FPS
    points_needed = total_points - len(selected_co)
    if points_needed > 0:
        candidate_pool = verts
        if USE_SAMPLING and len(candidate_pool) > sample_limit:
            random.seed(42)
            candidate_pool = random.sample(candidate_pool, sample_limit)
        
        distances = [float('inf')] * len(candidate_pool)
        for seed in selected_co:
            for i, v in enumerate(candidate_pool):
                d = (v - seed).length_squared
                if d < distances[i]: distances[i] = d

        for _ in range(points_needed):
            if max(distances) == 0: break 
            farthest_idx = distances.index(max(distances))
            farthest_co = candidate_pool[farthest_idx]
            selected_co.append(farthest_co)
            for i, v in enumerate(candidate_pool):
                d_new = (v - farthest_co).length_squared
                if d_new < distances[i]: distances[i] = d_new

    return selected_co[:total_points]

def process_single_model():
    obj = bpy.context.active_object
    if not obj or obj.type != 'MESH': return None

    bm = bmesh.new()
    bm.from_mesh(obj.data) 
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    
    points_local = get_hybrid_point_sampling(bm, NUM_POINTS, SAMPLE_LIMIT)
    bm.free()
    
    return points_local

def process_folder():
    print("-" * 50)
    print(f"Starting Batch Processing on: {INPUT_FOLDER}")
    print("-" * 50)

    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder not found: {INPUT_FOLDER}")
        return

    all_models_dict = {}
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.glb', '.gltf'))]
    
    for filename in files:
        file_path = os.path.join(INPUT_FOLDER, filename)
        model_name_key = os.path.splitext(filename)[0]
        
        print(f"Processing: {model_name_key}...")
        
        cleanup_scene()
        
        try:
            bpy.ops.import_scene.gltf(filepath=file_path)
        except Exception as e:
            print(f"  Error importing {filename}: {e}")
            continue

        mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        
        if not mesh_objects:
            print(f"  Warning: No meshes found in {filename}. Skipping.")
            continue
            
        bpy.ops.object.select_all(action='DESELECT')
        for mesh in mesh_objects:
            mesh.select_set(True)
        
        bpy.context.view_layer.objects.active = mesh_objects[0]
        if len(mesh_objects) > 1:
            bpy.ops.object.join()
            
        points = process_single_model()
        
        if points:
            all_models_dict[model_name_key] = {
                "count": len(points),
                "points": [[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in points]
            }
            print(f"  > Success: Generated {len(points)} local points.")
        else:
            print("  > Failed to generate points.")

    if all_models_dict:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_models_dict, f, indent=4)
        print("-" * 50)
        print(f"FINISHED. Processed {len(all_models_dict)} models.")
        print(f"Saved to: {OUTPUT_FILE}")
    else:
        print("No data generated.")

process_folder()