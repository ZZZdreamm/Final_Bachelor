import bpy
import os
import random
from math import radians
from mathutils import Vector, Euler


scene = bpy.context.scene

COLLECTION_NAME = "ImportedGLTF"
if COLLECTION_NAME not in bpy.data.collections:
    collection = bpy.data.collections.new(COLLECTION_NAME)
    scene.collection.children.link(collection)
else:
    collection = bpy.data.collections[COLLECTION_NAME]
    
    
DIRECTIONS = [
    Vector((x, y, z))
    for x in (-1, 0, 1)
    for y in (-1, 0, 1)
    for z in (0, 1)
    if (x, y, z) != (0, 0, 0)
]

CLOSEUP_DIRECTIONS = [
    # From front
    Vector((0, 1, 0)),    
    Vector((0.5, 1, 0)),  
    Vector((-0.5, 1, 0)), 
    Vector((0.2, 1, 0)),  
    Vector((-0.2, 1, 0)), 
    
    # From back
    Vector((0, -1, 0)),    
    Vector((0.5, -1, 0)),  
    Vector((-0.5, -1, 0)), 
    Vector((0.2, -1, 0)),  
    Vector((-0.2, -1, 0)), 
    
    Vector((0, 0.7, -0.3)),
    Vector((0, 0.5, -0.6)),
]

HUMAN_PERSPECTIVE_DIRECTIONS = [
    Vector((1, 1, 0.1)),    # Front-Right, low
    Vector((-1, 1, 0.1)),   # Front-Left, low
    Vector((0, 1, 0.15)),   # Front, slightly higher
    Vector((1, 0.5, 0.1)),  # Side-Front, low
    Vector((-1, 0.5, 0.1)), # Side-Front, low
]

Z_POSITIONS = [0, -0.1, 0.15]


ENVIRONMENT_CONFIGS = [
    {
        "name": "HardLight",
        "use_hdri": False,
        "hdri_strength": 0.5,
        "background_color": (0.2, 0.2, 0.2, 1.0),
        "use_sun_light": True,
        "sun_direction": Vector((1, -1, 1)),
        "sun_strength": 5.0
    },
    {
        "name": "SoftLight",
        "use_hdri": True,
        "hdri_strength": 1.0,
        "background_color": (0.05, 0.05, 0.05, 1.0), 
        "use_sun_light": False
    },
    {
        "name": "FlatColor",
        "use_hdri": False,
        "hdri_strength": 1.0,
        "background_color": (0.7, 0.7, 0.7, 1.0), 
        "use_sun_light": False
    },
]


PHONE_VIEW_CONFIGS = [
    {
        "name": "Normal",
        "distance_factor": 1.1,
        "lens": 28, 
        "random_tilt": True
    },
    {
        "name": "Distant",
        "distance_factor": 1.5,
        "lens": 28, 
        "random_tilt": True
    },
    {
        "name": "CloseUp",
        "distance_factor": 0.6,
        "lens": 35, 
        "random_tilt": True
    },
        {
        "name": "SideShot",
        "distance_factor": 0.8,
        "lens": 25, 
        "random_tilt": True
    },
    {
        "name": "HumanPerspective",
        "distance_factor": 0.5,
        "lens": 20,       
        "random_tilt": True
    },
    {
        "name": "VeryCloseUpAndLow",
        "distance_factor": 0.35,
        "lens": 16,
        "random_tilt": True
    },
]


def clear_scene_objects():
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
        
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.name.startswith("ENV_LIGHT_"):
            bpy.data.objects.remove(obj, do_unlink=True)


def import_model(model_path: str):
    clear_scene_objects()
    ext = os.path.splitext(model_path)[1].lower()

    try:
        if ext in [".glb", ".gltf"]:
            bpy.ops.import_scene.gltf(filepath=model_path)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=model_path)
        else:
            print(f"Unsupported format: {ext}")
            return
    except Exception as e:
        print(f"Failed to import {model_path}: {e}")
        return

    for obj in bpy.context.selected_objects:
        if obj.name not in collection.objects:
            collection.objects.link(obj)
        for coll in obj.users_collection:
            if coll != collection:
                coll.objects.unlink(obj)


def look_at(cam, target):
    direction = target - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()


bpy.types.Object.look_at = look_at


def setup_camera():
    objs = collection.objects
    if not objs:
        return None
    

    min_corner = Vector((float('inf'), float('inf'), float('inf')))
    max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))
    for obj in objs:
        for v in obj.bound_box:
            world_v = obj.matrix_world @ Vector(v)
            min_corner = Vector(map(min, zip(min_corner, world_v)))
            max_corner = Vector(map(max, zip(max_corner, world_v)))
    center = (min_corner + max_corner) / 2
    size_vector = (max_corner - min_corner)
    size = size_vector.length
    size_z = size_vector.z


    if scene.camera is None:
        bpy.ops.object.camera_add()
        scene.camera = bpy.context.object
    cam = scene.camera


    cam.location = center + Vector((0, -size * 1.2, size * 0.25))
    cam.look_at(center)
    
    cam.data.lens = 28


    return center, size, size_z, cam


def set_camera_phone_view(cam, center, size, config, direction):
    """
    Sets the camera based on a specific 'phone view' configuration.
    """
    cam.data.lens = config["lens"]

    distance = size * config["distance_factor"]
    cam.location = center + direction.normalized() * distance

    if config["random_tilt"]:
        offset_factor = size * random.uniform(0.01, 0.05)

        x_offset = Vector((1, 0, 0)) * random.uniform(-offset_factor, offset_factor)
        y_offset = Vector((0, 1, 0)) * random.uniform(-offset_factor, offset_factor)
        z_offset = Vector((0, 0, 1)) * random.uniform(-offset_factor, offset_factor)

        cam.location += x_offset + y_offset + z_offset

    cam.look_at(center)

    if config["random_tilt"]:
        tilt_angle = radians(random.uniform(-3, 3))
        roll_angle = radians(random.uniform(-2, 2))
        cam.rotation_euler.rotate(Euler((tilt_angle, 0, roll_angle), 'XYZ'))
        

def setup_environment(config, center):
    """
    Configures the scene's lighting, background, and world settings.
    """
    world = scene.world

    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.name.startswith("ENV_LIGHT_SUN"):
            bpy.data.objects.remove(obj, do_unlink=True)

    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]

    bg_node.inputs["Color"].default_value = config["background_color"]
    bg_node.inputs["Strength"].default_value = config["hdri_strength"]

    if config["use_sun_light"]:
        sun_data = bpy.data.lights.new(name="ENV_LIGHT_SUN_DATA", type='SUN')
        sun_data.energy = config["sun_strength"]
        sun_data.color = (1.0, 0.95, 0.9)

        sun_obj = bpy.data.objects.new(name="ENV_LIGHT_SUN", object_data=sun_data)
        scene.collection.objects.link(sun_obj)

        sun_obj.location = center - config["sun_direction"].normalized() * 100
        sun_obj.rotation_euler = config["sun_direction"].to_track_quat('-Z', 'Y').to_euler()
        

def render_single_view(output_dir: str, render_name: str, camera_details, direction: Vector, phone_config = None, distance_factor: float = 2.0, random_offset: bool = False, add_tilt: bool = False):
    center, size, size_z, cam = camera_details

    center = center.copy()
    if phone_config and phone_config["name"] == "VeryCloseUpAndLow":
        target_offset_factor = 0.20
        center.z += size_z * target_offset_factor

    if phone_config:
        set_camera_phone_view(cam, center, size, phone_config, direction)
    else:
        cam.location = center + direction.normalized() * size * distance_factor

        if random_offset:
            offset_factor = size * random.uniform(0.01, 0.05)

            x_offset = Vector((1, 0, 0)) * random.uniform(-offset_factor, offset_factor)
            y_offset = Vector((0, 1, 0)) * random.uniform(-offset_factor, offset_factor)
            z_offset = Vector((0, 0, 1)) * random.uniform(-offset_factor, offset_factor)

            cam.location += x_offset + y_offset + z_offset

        cam.look_at(center)

        if add_tilt:
            tilt_angle = radians(random.uniform(-3, 3))
            roll_angle = radians(random.uniform(-2, 2))
            cam.rotation_euler.rotate(Euler((tilt_angle, 0, roll_angle), 'XYZ'))

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, render_name)
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)



def render_model_all_views(model_path: str, output_dir: str, rotate_y: bool):
    """
    Render from overview and close-up perspectives across all environments
    """
    import_model(model_path)
    camera_details = setup_camera()

    center, size, size_z, cam = camera_details

    for env_config in ENVIRONMENT_CONFIGS:
        env_name = env_config["name"]
        setup_environment(env_config, center)

        for idx, direction in enumerate(DIRECTIONS):
            render_name = f"render_overview_{env_name}_dir{idx}.png"
            render_single_view(output_dir, render_name, camera_details, direction, distance_factor=1.1, random_offset=True, add_tilt=True)

        for phone_config in PHONE_VIEW_CONFIGS:
            phone_config_name = phone_config["name"]

            if phone_config_name == "HumanPerspective":
                current_directions = HUMAN_PERSPECTIVE_DIRECTIONS
                z_positions = [0]
            else:
                current_directions = CLOSEUP_DIRECTIONS
                z_positions = Z_POSITIONS

            for z_position in z_positions:
                for direction_idx, direction in enumerate(current_directions):
                    current_direction = direction.copy()
                    current_direction.z = z_position

                    if rotate_y:
                        current_direction.y *= -1

                    render_name = f"render_close_{env_name}_{phone_config_name}_z{z_position}_dir{direction_idx}.png"

                    render_single_view(output_dir, render_name, camera_details, current_direction, phone_config=phone_config, random_offset=True, add_tilt=True)


def render_folder(models_dir: str, output_dir: str):
    """
    Process every model in a folder
    """
    for filename in os.listdir(models_dir):
        if filename.lower().endswith((".glb", ".gltf", ".fbx")):
            model_path = os.path.join(models_dir, filename)
            model_name = os.path.splitext(filename)[0]
            model_render_dir = os.path.join(output_dir, model_name)

            rotate_flag = model_name.lower().startswith("y2")

            render_model_all_views(model_path, model_render_dir, rotate_flag)
            clear_scene_objects()


if __name__ == "__main__":
    models_dir = "C:/Users/KMult/Desktop/Praca_inzynierska/models/modele_rdy" 
    output_dir = "C:/Users/KMult/Desktop/Praca_inzynierska/models/BlenderRenders6"
    
    render_folder(models_dir, output_dir)