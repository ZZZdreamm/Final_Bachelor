import json
import cv2
import os
import numpy as np

MODEL_DIR = r"/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/AR_Renders105/Ilmet"

RENDER_INDEX = 0 

SHOW_OCCLUDED = True        
VISIBLE_COLOR = (0, 255, 0) 
OCCLUDED_COLOR = (0, 0, 255)
POINT_RADIUS = 4

def visualize_points():
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        data = json.load(f)

    renders = data.get("renders", [])
    if RENDER_INDEX >= len(renders):
        print(f"Error: Render index {RENDER_INDEX} out of range (max {len(renders)-1})")
        return

    render_data = renders[RENDER_INDEX]
    image_filename = render_data["filename"]
    points_2d = render_data["points_2d"]

    image_path = os.path.join(MODEL_DIR, "images", image_filename)
    if not os.path.exists(image_path):
        image_path = os.path.join(MODEL_DIR, image_filename)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Failed to load image.")
        return

    print(f"Visualizing render {RENDER_INDEX}: {image_filename}")
    print(f"Total points: {len(points_2d)}")
    
    visible_count = 0
    occluded_count = 0

    for pt in points_2d:
        x = int(pt["x"])
        y = int(pt["y"])
        is_visible = pt["visible"] == 1

        h, w = img.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            continue

        if is_visible:
            cv2.circle(img, (x, y), POINT_RADIUS, VISIBLE_COLOR, -1) 
            visible_count += 1
        elif SHOW_OCCLUDED:
            cv2.circle(img, (x, y), POINT_RADIUS, OCCLUDED_COLOR, -1)
            occluded_count += 1

    cv2.putText(img, f"Visible: {visible_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, VISIBLE_COLOR, 2)
    if SHOW_OCCLUDED:
        cv2.putText(img, f"Occluded: {occluded_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, OCCLUDED_COLOR, 2)

    display_height = 800
    aspect_ratio = img.shape[1] / img.shape[0]
    display_width = int(display_height * aspect_ratio)
    img_display = cv2.resize(img, (display_width, display_height))

    output_path = os.path.join(MODEL_DIR, f"debug_vis_{RENDER_INDEX}.jpg")
    cv2.imwrite(output_path, img)
    print(f"Saved visualization to: {output_path}")

if __name__ == "__main__":
    visualize_points()