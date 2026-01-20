import os
import json
import requests

API_URL = "http://localhost:8000/buildings/update_metadata/" 
METADATA_FILE = "./update_metadata.json"

def load_buildings(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        buildings_list = json.load(f)
        
    return {b["name"]: b for b in buildings_list.values()}

def send_building_data(name, location, height, width, depth):
    data = {
        "name": name,
        "location": location,
        "height": height,
        "width": width,
        "depth": depth
    }

    response = requests.post(API_URL, json=data)

def main():
    buildings = load_buildings(METADATA_FILE)

    for name, building in buildings.items():
        name = building.get("name")
        location = building.get("location")
        height = building.get("height")
        width = building.get("width")
        depth = building.get("depth")
        send_building_data(name, location, height, width, depth)
        
if __name__ == "__main__":
    main()
