import torch
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel, CLIPConfig
import h5py


CLIP_MODEL_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"


def preprocess_dataset(annotations_file, output_file, device='cuda', batch_size=32):
    """
    Preprocess all images with CLIP and save features to HDF5 file
    
    Args:
        annotations_file: Path to dataset.json
        output_file: Path to save .h5 file with features
        device: 'cuda' or 'cpu'
        batch_size: Number of images to process at once
    """
    
    print("Loading CLIP model and processor...")
    clip_config = CLIPConfig.from_pretrained(CLIP_MODEL_ID)
    clip_model = CLIPVisionModel.from_pretrained(
        CLIP_MODEL_ID,
        config=clip_config.vision_config,
    )
    clip_model.to(device)
    clip_model.eval()  
    
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    
    print(f"✓ CLIP model loaded (feature dim: {clip_model.config.hidden_size})")
    
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    images_dir = Path(annotations_file).parent.parent / 'images'
    
    print(f"Total samples to process: {len(annotations)}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as h5f:
        feature_dim = clip_model.config.hidden_size
        
        image_ids = []
        all_features = []
        all_rotations = []
        all_translations = []
        all_building_ids = []
        all_splits = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(annotations), batch_size), desc="Extracting features"):
                batch_anns = annotations[i:i+batch_size]
                batch_images = []
                
                for ann in batch_anns:
                    img_path = images_dir / ann['image_filename']
                    image = cv2.imread(str(img_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    batch_images.append(image)
                
                # Process with CLIP processor
                inputs = processor(images=batch_images, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(device)
                
                # Extract features
                outputs = clip_model(pixel_values=pixel_values)
                features = outputs.pooler_output.cpu().numpy()  
                
                # Store data
                for j, ann in enumerate(batch_anns):
                    image_ids.append(str(ann['image_id']))
                    all_features.append(features[j])
                    all_rotations.append(ann['rotation_quaternion'])
                    all_translations.append(ann['translation'])
                    all_building_ids.append(str(ann['building_id']))
                    all_splits.append(ann['split'])
        
        all_features = np.array(all_features, dtype=np.float32)
        all_rotations = np.array(all_rotations, dtype=np.float32)
        all_translations = np.array(all_translations, dtype=np.float32)
        all_building_ids = np.array(all_building_ids, dtype=np.int32)
        
        h5f.create_dataset('features', data=all_features, compression='gzip')
        h5f.create_dataset('rotations', data=all_rotations, compression='gzip')
        h5f.create_dataset('translations', data=all_translations, compression='gzip')
        h5f.create_dataset('building_ids', data=all_building_ids, compression='gzip')
        
        dt = h5py.string_dtype(encoding='utf-8')
        h5f.create_dataset('image_ids', data=np.array(image_ids, dtype=dt))
        h5f.create_dataset('splits', data=np.array(all_splits, dtype=dt))
        
        h5f.attrs['num_samples'] = len(image_ids)
        h5f.attrs['feature_dim'] = feature_dim
        h5f.attrs['clip_model_id'] = CLIP_MODEL_ID
        
    print(f"\n✓ Features saved to {output_file}")
    print(f"  Total samples: {len(image_ids)}")
    print(f"  Feature shape: {all_features.shape}")
    print(f"  File size: {output_path.stat().st_size / 1024**2:.2f} MB")
    
    splits_unique, splits_counts = np.unique(all_splits, return_counts=True)
    print(f"\n  Split distribution:")
    for split, count in zip(splits_unique, splits_counts):
        print(f"    {split}: {count} samples")


def verify_preprocessed_data(h5_file):
    """Verify the preprocessed data"""
    print(f"\nVerifying {h5_file}...")
    
    with h5py.File(h5_file, 'r') as h5f:
        print(f"Datasets in file: {list(h5f.keys())}")
        print(f"\nMetadata:")
        for key, value in h5f.attrs.items():
            print(f"  {key}: {value}")
        
        print(f"\nDataset shapes:")
        print(f"  features: {h5f['features'].shape}")
        print(f"  rotations: {h5f['rotations'].shape}")
        print(f"  translations: {h5f['translations'].shape}")
        print(f"  building_ids: {h5f['building_ids'].shape}")
        
        print(f"\nSample data (first entry):")
        print(f"  Image ID: {h5f['image_ids'][0]}")
        print(f"  Split: {h5f['splits'][0]}")
        print(f"  Feature (first 5): {h5f['features'][0][:5]}")
        print(f"  Rotation: {h5f['rotations'][0]}")
        print(f"  Translation: {h5f['translations'][0]}")
        print(f"  Building ID: {h5f['building_ids'][0]}")


if __name__ == "__main__":
    annotations_file = '/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/AR_pose1/annotations/dataset.json'
    output_file = '/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/AR_pose1/annotations/clip_features.h5'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("⚠️  WARNING: Processing on CPU will be slower!")
    
    preprocess_dataset(
        annotations_file=annotations_file,
        output_file=output_file,
        device=device,
        batch_size=32  
    )
    
    verify_preprocessed_data(output_file)
    
    print("\n✓ Preprocessing complete! You can now use train_pose_cached.py for training.")