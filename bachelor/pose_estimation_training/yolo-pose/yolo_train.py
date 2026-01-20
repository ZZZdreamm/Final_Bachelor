import os
import json
import shutil
import cv2
import yaml
import random
from ultralytics import YOLO

PREPARE_DATASET = False
INPUT_DATA_DIR = "/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/BlenderRenders420" 
YOLO_DATASET_DIR = "training_yolo/yolo_building_pose_dataset_new3"
PROJECT_NAME = "yolo_model_pose_new3"
EPOCHS = 400
BATCH_SIZE = 16
IMG_SIZE = 640

NUM_KEYPOINTS = 150
VAL_SET_KEYWORD = "modern_evening_street"
# ==============================================

def setup_directories():
    if os.path.exists(YOLO_DATASET_DIR): shutil.rmtree(YOLO_DATASET_DIR)
    for t in ['train', 'val']:
        os.makedirs(f"{YOLO_DATASET_DIR}/images/{t}", exist_ok=True)
        os.makedirs(f"{YOLO_DATASET_DIR}/labels/{t}", exist_ok=True)

def normalize_coords(value, max_value):
    return max(0.0, min(1.0, value / max_value))

def create_yolo_label(class_id, image_shape, keypoints_data):
    """
    Generates YOLO label line.
    keypoints_data is expected to be list of NUM_KEYPOINTS (60) points: [x, y, v]
    """
    img_h, img_w = image_shape[:2]

    if len(keypoints_data) != NUM_KEYPOINTS:
        print(f"WARNING: Found {len(keypoints_data)} points, expected {NUM_KEYPOINTS}. Padding with zeros.")
        while len(keypoints_data) < NUM_KEYPOINTS:
            keypoints_data.append([0, 0, 0])
        if len(keypoints_data) > NUM_KEYPOINTS:
            keypoints_data = keypoints_data[:NUM_KEYPOINTS]

    cleaned_keypoints = []
    for pt in keypoints_data:
        x, y, v = pt
        if int(v) == 1:
            cleaned_keypoints.append([0.0, 0.0, 0])
        else:
            cleaned_keypoints.append([x, y, v])
    
    keypoints_data = cleaned_keypoints

    xs = [pt[0] for pt in keypoints_data if pt[2] > 0]
    ys = [pt[1] for pt in keypoints_data if pt[2] > 0]
    
    if not xs: 
        xs = [0]
        ys = [0]
    
    pad = 10
    min_x, max_x = max(0, min(xs)-pad), min(img_w, max(xs)+pad)
    min_y, max_y = max(0, min(ys)-pad), min(img_h, max(ys)+pad)

    bw, bh = max_x - min_x, max_y - min_y
    cx, cy = min_x + bw/2, min_y + bh/2

    kpts_str = []
    for pt in keypoints_data:
        px = normalize_coords(pt[0], img_w)
        py = normalize_coords(pt[1], img_h)
        visibility = int(pt[2]) 
        kpts_str.extend([f"{px:.6f}", f"{py:.6f}", f"{visibility}"])

    label_line = (
        f"{class_id} "
        f"{normalize_coords(cx, img_w):.6f} {normalize_coords(cy, img_h):.6f} "
        f"{normalize_coords(bw, img_w):.6f} {normalize_coords(bh, img_h):.6f} "
        + " ".join(kpts_str)
    )
    return label_line

def process_dataset():
    all_samples = []
    model_dirs = sorted([d for d in os.listdir(INPUT_DATA_DIR) if os.path.isdir(os.path.join(INPUT_DATA_DIR, d))])
    
    model_class_map = {name: idx for idx, name in enumerate(model_dirs)}
    names_map = {idx: name for name, idx in model_class_map.items()}

    print(f"Processing {len(model_dirs)} models...")

    for model_name in model_dirs:
        model_dir = os.path.join(INPUT_DATA_DIR, model_name)
        current_class_id = model_class_map[model_name]
        
        json_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(json_path): continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        for entry in data:
            all_samples.append({
                'src_path': os.path.join(model_dir, entry['filename']),
                'unique_name': f"{model_name}_{entry['filename']}",
                'filename': entry['filename'], 
                'keypoints': entry['keypoints_2d'], 
                'class_id': current_class_id
            })

    train_set = []
    val_set = []

    for s in all_samples:
        if VAL_SET_KEYWORD in s['filename']:
            val_set.append(s)
        else:
            train_set.append(s)

    random.shuffle(train_set)
    random.shuffle(val_set)

    print(f"Dataset Split: {len(train_set)} Training images, {len(val_set)} Validation images.")

    def save_data(samples, split_name):
        for s in samples:
            shutil.copy(s['src_path'], os.path.join(YOLO_DATASET_DIR, f"images/{split_name}", s['unique_name']))
            
            img = cv2.imread(s['src_path'])
            if img is None:
                print(f"Error reading image: {s['src_path']}")
                continue
            h, w = img.shape[:2]

            label = create_yolo_label(s['class_id'], (h,w), s['keypoints'])
            txt_name = os.path.splitext(s['unique_name'])[0] + ".txt"
            
            with open(os.path.join(YOLO_DATASET_DIR, f"labels/{split_name}", txt_name), 'w') as f:
                f.write(label)

    print("Saving Training Data...")
    save_data(train_set, "train")
    print("Saving Validation Data...")
    save_data(val_set, "val")

    yaml_content = {
        'path': os.path.abspath(YOLO_DATASET_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'kpt_shape': [NUM_KEYPOINTS, 3], 
        'names': names_map
    }

    yaml_path = os.path.join(YOLO_DATASET_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    return yaml_path

def train_model(yaml_path):
    model = YOLO('yolo11l-pose.pt') 
    
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=f"run_{NUM_KEYPOINTS}pt_pose_aug", 
        workers=4,

        # GEOMETRIC AUGMENTATIONS
        mosaic=1.0,       
        degrees=15.0,     
        scale=0.5,        
        translate=0.1,    
        perspective=0.001,
        
        # COLOR AUGMENTATIONS
        hsv_h=0.015,      
        hsv_s=0.7,        
        hsv_v=0.4,        
        flipud=0.0,
        fliplr=0.0,    
    )

if __name__ == "__main__":
    if PREPARE_DATASET:
        setup_directories()
        yaml_path = process_dataset()
        train_model(yaml_path)
    else:
        train_model(os.path.join(YOLO_DATASET_DIR, "data.yaml"))