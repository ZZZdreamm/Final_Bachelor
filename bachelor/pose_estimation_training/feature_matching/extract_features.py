import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import pickle
from typing import Dict

try:
    import torch
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using SIFT only.")

try:
    from lightglue import SuperPoint
    SUPERPOINT_AVAILABLE = True
except ImportError:
    SUPERPOINT_AVAILABLE = False
    print("Warning: Kornia not available. SuperPoint disabled.")


class FeatureExtractor:
    """
    Multi-method feature extractor for building recognition.
    """
    
    def __init__(self, 
                 use_superpoint: bool = True,
                 use_sift: bool = True,
                 use_global: bool = True,
                 device: str = 'cuda'):
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_superpoint = use_superpoint and SUPERPOINT_AVAILABLE and TORCH_AVAILABLE
        self.use_sift = use_sift
        self.use_global = use_global and TORCH_AVAILABLE
        
        print(f"Device: {self.device}")
        print(f"SuperPoint: {'enabled' if self.use_superpoint else 'disabled'}")
        print(f"SIFT: {'enabled' if self.use_sift else 'disabled'}")
        print(f"Global features: {'enabled' if self.use_global else 'disabled'}")
        
        if self.use_superpoint:
            self._init_superpoint()
        
        if self.use_sift:
            self._init_sift()
        
        if self.use_global:
            self._init_global_extractor()
    
    def _init_superpoint(self):
        """Initialize SuperPoint detector"""
        self.superpoint = SuperPoint(
            max_num_keypoints=4096,
            nms_radius=3,
            detection_threshold=0.005
        ).to(self.device)
        self.superpoint.eval()
        print("  SuperPoint initialized")
    
    def _init_sift(self):
        """Initialize SIFT detector"""
        self.sift = cv2.SIFT_create(
            nfeatures=2000,
            contrastThreshold=0.04,
            edgeThreshold=10
        )
        print("  SIFT initialized")
    
    def _init_global_extractor(self):
        """Initialize global feature extractor (ResNet)"""
        self.global_model = models.resnet50(pretrained=True)
        self.global_model = torch.nn.Sequential(
            *list(self.global_model.children())[:-1]  
        ).to(self.device)
        self.global_model.eval()
        
        self.global_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("  Global extractor (ResNet50) initialized")
    
    def extract_superpoint(self, image: np.ndarray) -> Dict:
        """
        Extract SuperPoint keypoints and descriptors.
        
        Args:
            image: BGR image (H, W, 3)
        
        Returns:
            Dict with 'keypoints' (N, 2), 'descriptors' (N, 256), 'scores' (N,)
        """
        if not self.use_superpoint:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.superpoint({'image': tensor})
        
        keypoints = output['keypoints'][0].cpu().numpy()  
        descriptors = output['descriptors'][0].cpu().numpy() 
        scores = output['keypoint_scores'][0].cpu().numpy()  
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scores': scores,
            'method': 'superpoint'
        }
    
    def extract_sift(self, image: np.ndarray) -> Dict:
        """
        Extract SIFT keypoints and descriptors.
        
        Args:
            image: BGR image (H, W, 3)
        
        Returns:
            Dict with 'keypoints' (N, 2), 'descriptors' (N, 128)
        """
        if not self.use_sift:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if keypoints is None or len(keypoints) == 0:
            return {
                'keypoints': np.array([]),
                'descriptors': np.array([]),
                'method': 'sift'
            }
        
        kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        return {
            'keypoints': kp_array,
            'descriptors': descriptors,
            'sizes': np.array([kp.size for kp in keypoints]),
            'angles': np.array([kp.angle for kp in keypoints]),
            'responses': np.array([kp.response for kp in keypoints]),
            'method': 'sift'
        }
    
    def extract_global(self, image: np.ndarray) -> Dict:
        """
        Extract global image descriptor using CNN.
        
        Args:
            image: BGR image (H, W, 3)
        
        Returns:
            Dict with 'descriptor' (2048,)
        """
        if not self.use_global:
            return None
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        tensor = self.global_transform(rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.global_model(tensor)
        
        descriptor = features.squeeze().cpu().numpy()
        descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-8)
        
        return {
            'descriptor': descriptor,
            'method': 'resnet50'
        }
    
    def extract_all(self, image: np.ndarray) -> Dict:
        """
        Extract all feature types from an image.
        
        Args:
            image: BGR image (H, W, 3)
        
        Returns:
            Dict containing all extracted features
        """
        result = {
            'image_shape': image.shape[:2]
        }
        
        if self.use_superpoint:
            result['superpoint'] = self.extract_superpoint(image)
        
        if self.use_sift:
            result['sift'] = self.extract_sift(image)
        
        if self.use_global:
            result['global'] = self.extract_global(image)
        
        return result


class FeatureDatabaseBuilder:
    """
    Build feature database from rendered images.
    """
    
    def __init__(self, extractor: FeatureExtractor):
        self.extractor = extractor
    
    def process_model(self, model_dir: str) -> Dict:
        """
        Process all renders for a single model.
        
        Args:
            model_dir: Path to model directory containing images/ and metadata.json
        
        Returns:
            Dict containing model features and metadata
        """
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            print(f"Warning: No metadata.json in {model_dir}")
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_name = metadata['model_name']
        images_dir = model_dir
        
        if not os.path.exists(images_dir):
            print(f"Warning: No images directory in {model_dir}")
            return None
        
        render_features = []
        
        for render_info in tqdm(metadata['renders'], desc=f"  {model_name}"):
            image_path = os.path.join(images_dir, render_info['filename'])
            
            if not os.path.exists(image_path):
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            features = self.extractor.extract_all(image)
            
            render_features.append({
                'filename': render_info['filename'],
                'intrinsic_K': render_info['intrinsic_K'],
                'extrinsic_R': render_info['extrinsic_R'],
                'extrinsic_t': render_info['extrinsic_t'],
                'camera_config': render_info['camera_config'],
                'focal_length_mm': render_info['focal_length_mm'],
                'points_2d': render_info['points_2d'],
                'features': features
            })
        
        model_data = {
            'model_name': model_name,
            'bounds': metadata['bounds'],
            'surface_points_3d': metadata['surface_points_3d'],
            'num_renders': len(render_features),
            'renders': render_features
        }
        
        if self.extractor.use_global:
            global_descriptors = []
            for r in render_features:
                if r['features'].get('global'):
                    global_descriptors.append(r['features']['global']['descriptor'])
            
            if global_descriptors:
                model_data['global_descriptors'] = np.array(global_descriptors)
                # Mean descriptor for model-level matching
                model_data['mean_global_descriptor'] = np.mean(
                    model_data['global_descriptors'], axis=0
                )
        
        return model_data
    
    def build_database(self, renders_dir: str, output_dir: str):
        """
        Build complete feature database from all models.
        
        Args:
            renders_dir: Directory containing model subdirectories
            output_dir: Where to save the feature database
        """
        os.makedirs(output_dir, exist_ok=True)
        
        model_dirs = [
            d for d in os.listdir(renders_dir)
            if os.path.isdir(os.path.join(renders_dir, d))
        ]
        
        print(f"Found {len(model_dirs)} models to process")
        
        database = {
            'models': {},
            'model_names': [],
            'global_index': None 
        }
        
        all_global_descriptors = []
        global_to_model = []  
        
        for model_name in model_dirs:
            model_dir = os.path.join(renders_dir, model_name)
            print(f"\nProcessing: {model_name}")
            
            model_data = self.process_model(model_dir)
            
            if model_data is None:
                continue
            
            database['models'][model_name] = model_data
            database['model_names'].append(model_name)
            
            # Build global index
            if 'global_descriptors' in model_data:
                for i, desc in enumerate(model_data['global_descriptors']):
                    all_global_descriptors.append(desc)
                    global_to_model.append((model_name, i))
        
        # Build global descriptor matrix for fast nearest neighbor search
        if all_global_descriptors:
            database['global_index'] = {
                'descriptors': np.array(all_global_descriptors),
                'mapping': global_to_model
            }
        
        print(f"\nSaving database to {output_dir}")
        
        with open(os.path.join(output_dir, 'feature_database.pkl'), 'wb') as f:
            pickle.dump(database, f)
        
        summary = {
            'num_models': len(database['model_names']),
            'model_names': database['model_names'],
            'total_renders': sum(
                m['num_renders'] for m in database['models'].values()
            ),
            'global_index_size': len(all_global_descriptors) if all_global_descriptors else 0
        }
        
        with open(os.path.join(output_dir, 'database_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Database built successfully!")
        print(f"  Models: {summary['num_models']}")
        print(f"  Total renders: {summary['total_renders']}")
        print(f"  Global index size: {summary['global_index_size']}")


def extract_single_image(image_path: str, output_path: str = None):
    """
    Extract features from a single image (for testing).
    """
    extractor = FeatureExtractor()
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    features = extractor.extract_all(image)
    
    print(f"Image shape: {features['image_shape']}")
    
    if 'superpoint' in features and features['superpoint']:
        sp = features['superpoint']
        print(f"SuperPoint: {len(sp['keypoints'])} keypoints")
    
    if 'sift' in features and features['sift']:
        sift = features['sift']
        print(f"SIFT: {len(sift['keypoints'])} keypoints")
    
    if 'global' in features and features['global']:
        print(f"Global: {features['global']['descriptor'].shape}")
    
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Saved to: {output_path}")
    
    return features


def visualize_features(image_path: str, output_path: str):
    """
    Visualize extracted features on image.
    """
    extractor = FeatureExtractor()
    
    image = cv2.imread(image_path)
    features = extractor.extract_all(image)
    
    vis_image = image.copy()
    
    # Draw SuperPoint keypoints (red)
    if 'superpoint' in features and features['superpoint']:
        for kp in features['superpoint']['keypoints']:
            cv2.circle(vis_image, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)
    
    # Draw SIFT keypoints (green)
    if 'sift' in features and features['sift']:
        for kp in features['sift']['keypoints']:
            cv2.circle(vis_image, (int(kp[0]), int(kp[1])), 2, (0, 255, 0), -1)
    
    cv2.imwrite(output_path, vis_image)
    print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract features for AR building recognition')
    parser.add_argument('--renders_dir', type=str, required=True,
                        help='Directory containing rendered model images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for feature database')
    parser.add_argument('--no_superpoint', action='store_true',
                        help='Disable SuperPoint extraction')
    parser.add_argument('--no_sift', action='store_true',
                        help='Disable SIFT extraction')
    parser.add_argument('--no_global', action='store_true',
                        help='Disable global feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for neural network inference')
    
    args = parser.parse_args()
    
    extractor = FeatureExtractor(
        use_superpoint=not args.no_superpoint,
        use_sift=not args.no_sift,
        use_global=not args.no_global,
        device=args.device
    )
    
    builder = FeatureDatabaseBuilder(extractor)
    builder.build_database(args.renders_dir, args.output_dir)


if __name__ == "__main__":
    main()