import os
import json
import requests

API_URL = "http://localhost:8000/buildings"
METADATA_FILE = "./metadata.json"

def load_buildings(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def save_buildings(json_file, buildings_dict):
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(buildings_dict, f, indent=4, ensure_ascii=False)

def delete_building(building_name):
    url = f"{API_URL}/{building_name}/"
    response = requests.delete(url)
    return response.status_code == 200

def main():
    buildings = load_buildings(METADATA_FILE)

    for name, building in buildings.items():
        if building.get("should_skip", False):
            success = delete_building(name)
            if success:
                building["should_skip"] = False

    save_buildings(METADATA_FILE, buildings)

if __name__ == "__main__":
    main()