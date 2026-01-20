import os
import json
import requests
import argparse

API_URL = "http://localhost:8000/buildings/add_model/" 
MODELS_FOLDER = "/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/modele_rdy" 
METADATA_FILE = "./metadata.json"

def load_buildings(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        buildings_list = json.load(f)
    return {b["name"]: b for b in buildings_list.values()}

def save_buildings(json_file, buildings_dict):
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(buildings_dict, f, indent=4, ensure_ascii=False)

def send_building_data(building_data, model_file_path):
    """
    Sends building metadata and a single 3D model file to the API endpoint.

    Args:
        building_data (dict): Dictionary containing building metadata.
        model_file_path (str): File path to the 3D model (.glb or .fbx).
    """
    
    model_filename = os.path.basename(model_file_path)

    mime_type = "application/octet-stream"
    if model_filename.lower().endswith(".glb"):
        mime_type = "model/gltf-binary"
    elif model_filename.lower().endswith(".fbx"):
        mime_type = "application/octet-stream"

    try:
        file_handle = open(model_file_path, "rb")
        files = {
            "model_file": (model_filename, file_handle, mime_type)
        }

        data = {
            "building": json.dumps(building_data)
        }

        response = requests.post(API_URL, data=data, files=files)
        return response

    except FileNotFoundError:
        return None
    except Exception:
        return None
    finally:
        if 'file_handle' in locals() and not file_handle.closed:
            file_handle.close()

def main(force=False):
    buildings = load_buildings(METADATA_FILE)
    index = 0
    MAXIMUM = 10000

    MODEL_EXTENSIONS = (".glb", ".fbx")
    for name, building in buildings.items():
        name = building.get("name")
        location = building.get("location")
        height = building.get("height")
        width = building.get("width")
        depth = building.get("depth")
        should_skip = building.get("should_skip", False)

        if should_skip and not force:
            continue

        model_file = None
        for ext in MODEL_EXTENSIONS:
            expected_file_path = os.path.join(MODELS_FOLDER, name + ext)

            if os.path.exists(expected_file_path):
                model_file = expected_file_path
                break

        if not model_file:
            continue

        building_data = {
            "name": name,
            "location": location,
            "height": height,
            "width": width,
            "depth": depth
        }
        response = send_building_data(building_data, model_file)
        index += 1

        if response.status_code == 200:
            building["should_skip"] = True

    save_buildings(METADATA_FILE, buildings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send building data and renders.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore should_skip flag and process all buildings"
    )
    args = parser.parse_args()

    main(force=args.force)
