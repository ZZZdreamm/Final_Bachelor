import torch
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import trimesh
from torchvision import transforms
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from pose_network import BuildingPoseNetCached, quaternion_to_rotation_matrix
from pose_refinement import PoseRefiner
from common.mesh_utils import (
    load_and_normalize_mesh, get_bbox_edges, create_bbox_corners, project_bbox_to_2d
)


class BatchBuildingPoseEstimator:
    """
    Process folders of building images and generate pose predictions
    """
    
    def __init__(self, model_path, config_path, mesh_dir, device='cuda', image_size=512):
        self.device = device
        self.image_size = image_size
        self.mesh_dir = Path(mesh_dir)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model = BuildingPoseNetCached(
            num_buildings=self.config['num_buildings']
        )
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Buildings: {', '.join(self.config['buildings'])}")
        print(f"✓ Image size: {image_size}x{image_size}")
        print(f"✓ Device: {device}\n")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.refiners = {}
    
    def predict_pose(self, image):
        """Predict pose from single image"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
        
        building_id = torch.argmax(output['building_logits'], dim=1).item()
        building_probs = torch.softmax(output['building_logits'], dim=1)[0].cpu().numpy()
        rotation_quat = output['rotation'][0].cpu().numpy()
        translation = output['translation'][0].cpu().numpy()
        confidence = output['confidence'][0].item()
        
        rotation_matrix = quaternion_to_rotation_matrix(
            torch.from_numpy(rotation_quat)
        ).numpy()
        
        return {
            'building_id': building_id,
            'building_name': self.config['buildings'][building_id],
            'building_probabilities': {
                self.config['buildings'][i]: float(building_probs[i])
                for i in range(len(building_probs))
            },
            'rotation_quaternion': rotation_quat.tolist(),
            'rotation_matrix': rotation_matrix.tolist(),
            'translation': translation.tolist(),
            'confidence': float(confidence)
        }
    
    def refine_pose(self, image, prediction, mesh_path, method='scipy', max_iterations=50):
        """Refine pose using render-and-compare"""
        building_id = prediction['building_id']
        
        if building_id not in self.refiners:
            self.refiners[building_id] = PoseRefiner(
                mesh_path=str(mesh_path),
                image_size=(self.image_size, self.image_size),
                device=self.device
            )
        
        refiner = self.refiners[building_id]
        
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        
        if method == 'scipy':
            refined_quat, refined_trans = refiner.refine_pose_scipy(
                initial_rotation=np.array(prediction['rotation_quaternion']),
                initial_translation=np.array(prediction['translation']),
                target_image=image_resized,
                max_iterations=max_iterations
            )
        else:
            refined_quat, refined_trans, _ = refiner.refine_pose_grid_search(
                initial_rotation=np.array(prediction['rotation_quaternion']),
                initial_translation=np.array(prediction['translation']),
                target_image=image_resized,
                num_samples=5
            )
        
        refined_R = quaternion_to_rotation_matrix(
            torch.from_numpy(refined_quat)
        ).numpy()
        
        return {
            'rotation_quaternion': refined_quat.tolist(),
            'rotation_matrix': refined_R.tolist(),
            'translation': refined_trans.tolist()
        }
    
    def render_overlay(self, image, rotation_matrix, translation, mesh_path,
                      alpha=0.5, output_size=None):
        """
        Render 3D model bounding box overlay on image (fast visualization)
        """
        mesh = load_and_normalize_mesh(mesh_path, target_scale=10.0)
        
        if output_size is None:
            render_h, render_w = image.shape[:2]
        else:
            render_w, render_h = output_size
        
        if output_size is not None:
            image = cv2.resize(image, output_size)
        
        overlay = image.copy()
        rendered = np.zeros_like(image)
        
        focal_length = 800
        K = np.array([
            [focal_length, 0, render_w/2],
            [0, focal_length, render_h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        bbox_corners_3d = create_bbox_corners(mesh.bounds)
        bbox_2d = project_bbox_to_2d(bbox_corners_3d, rotation_matrix, translation, K)
        bbox_2d = bbox_2d.astype(int)

        bbox_edges = get_bbox_edges()
        
        for i, j in bbox_edges:
            pt1 = tuple(bbox_2d[i])
            pt2 = tuple(bbox_2d[j])
            cv2.line(overlay, pt1, pt2, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.line(rendered, pt1, pt2, (0, 255, 255), 3, cv2.LINE_AA)
        
        for corner in bbox_2d:
            cv2.circle(overlay, tuple(corner), 5, (255, 0, 255), -1)
            cv2.circle(rendered, tuple(corner), 5, (255, 0, 255), -1)
        
        axis_length = 3.0
        axes_3d = np.float32([
            [0, 0, 0], 
            [axis_length, 0, 0], 
            [0, axis_length, 0], 
            [0, 0, axis_length]  
        ])
        
        axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, np.zeros(5))
        axes_2d = axes_2d.reshape(-1, 2).astype(int)
        
        origin = tuple(axes_2d[0])
        cv2.arrowedLine(overlay, origin, tuple(axes_2d[1]), (0, 0, 255), 3, tipLength=0.3)
        cv2.arrowedLine(rendered, origin, tuple(axes_2d[1]), (0, 0, 255), 3, tipLength=0.3)
        cv2.arrowedLine(overlay, origin, tuple(axes_2d[2]), (0, 255, 0), 3, tipLength=0.3)
        cv2.arrowedLine(rendered, origin, tuple(axes_2d[2]), (0, 255, 0), 3, tipLength=0.3)
        cv2.arrowedLine(overlay, origin, tuple(axes_2d[3]), (255, 0, 0), 3, tipLength=0.3)
        cv2.arrowedLine(rendered, origin, tuple(axes_2d[3]), (255, 0, 0), 3, tipLength=0.3)
        
        cv2.putText(overlay, 'BBOX', tuple(bbox_2d[6] + np.array([10, -10])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        mask = np.any(rendered > 0, axis=2).astype(np.uint8) * 255
        
        comparison = np.hstack([image, rendered, overlay])
        
        return overlay, rendered, comparison, mask
    
    def create_visualization_grid(self, image, overlay, rendered, prediction, 
                                 refined_prediction=None):
        """
        Create informative visualization grid with annotations
        """
        h, w = image.shape[:2]
        
        display_size = (512, 512)
        img_display = cv2.resize(image, display_size)
        overlay_display = cv2.resize(overlay, display_size)
        rendered_display = cv2.resize(rendered, display_size)
        
        canvas_h = display_size[1] * 2 + 100 
        canvas_w = display_size[0] * 2
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        
        canvas[0:display_size[1], 0:display_size[0]] = img_display
        canvas[0:display_size[1], display_size[0]:] = rendered_display
        canvas[display_size[1]:display_size[1]*2, 0:display_size[0]] = overlay_display
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (0, 0, 0)
        
        cv2.putText(canvas, "Original Image", (10, 30), font, font_scale, color, thickness)
        cv2.putText(canvas, "3D Model Render", (display_size[0] + 10, 30), font, font_scale, color, thickness)
        cv2.putText(canvas, "Overlay", (10, display_size[1] + 30), font, font_scale, color, thickness)
        
        info_y = display_size[1] + 60
        info_x = display_size[0] + 10
        line_height = 30
        
        building_text = f"Building: {prediction['building_name']}"
        cv2.putText(canvas, building_text, (info_x, info_y), 
                   font, 0.5, color, 1)
        
        conf_text = f"Confidence: {prediction['confidence']:.3f}"
        cv2.putText(canvas, conf_text, (info_x, info_y + line_height), 
                   font, 0.5, color, 1)
        
        trans = prediction['translation']
        trans_text = f"Position: [{trans[0]:.1f}, {trans[1]:.1f}, {trans[2]:.1f}]"
        cv2.putText(canvas, trans_text, (info_x, info_y + line_height * 2), 
                   font, 0.4, color, 1)
        
        if refined_prediction:
            cv2.putText(canvas, "Refined: Yes", (info_x, info_y + line_height * 3), 
                       font, 0.5, (0, 150, 0), 2)
        
        return canvas
    
    def process_building_folder(self, folder_path, building_name, output_dir, 
                               use_refinement=False, create_visualizations=True):
        """
        Process all images in a building folder
        """
        folder_path = Path(folder_path)
        output_dir = Path(output_dir) / building_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in folder_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"⚠️  No images found in {folder_path}")
            return []
        
        print(f"\n{'='*70}")
        print(f"Processing: {building_name}")
        print(f"Images: {len(image_files)}")
        print(f"{'='*70}")
        
        mesh_path = self.mesh_dir / f"{building_name}.glb"
        if not mesh_path.exists():
            for ext in ['.gltf', '.fbx']:
                alt_path = self.mesh_dir / f"{building_name}{ext}"
                if alt_path.exists():
                    mesh_path = alt_path
                    break
        
        if not mesh_path.exists():
            print(f"⚠️  Mesh not found: {mesh_path}")
            print("   Skipping visualization (predictions only)")
            mesh_path = None
        
        results = []
        
        for image_file in tqdm(image_files, desc=f"Processing {building_name}"):
            try:
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"⚠️  Failed to load {image_file}")
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                prediction = self.predict_pose(image_rgb)
                
                refined_prediction = None
                if use_refinement and mesh_path:
                    try:
                        refined_prediction = self.refine_pose(
                            image_rgb, prediction, mesh_path
                        )
                    except Exception as e:
                        print(f"⚠️  Refinement failed for {image_file.name}: {e}")
                
                final_prediction = refined_prediction if refined_prediction else {
                    'rotation_quaternion': prediction['rotation_quaternion'],
                    'rotation_matrix': prediction['rotation_matrix'],
                    'translation': prediction['translation']
                }
                
                if create_visualizations and mesh_path:
                    try:
                        overlay, rendered, comparison, mask = self.render_overlay(
                            image_rgb,
                            np.array(final_prediction['rotation_matrix']),
                            np.array(final_prediction['translation']),
                            mesh_path,
                            alpha=0.5
                        )
                        
                        grid = self.create_visualization_grid(
                            image_rgb, overlay, rendered, prediction, 
                            refined_prediction
                        )
                        
                        base_name = image_file.stem
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_overlay.jpg"),
                            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                        )
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_comparison.jpg"),
                            cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                        )
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_grid.jpg"),
                            cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
                        )
                        
                    except Exception as e:
                        print(f"⚠️  Visualization failed for {image_file.name}: {e}")
                
                result = {
                    'image_file': image_file.name,
                    'building_folder': building_name,
                    'initial_prediction': prediction,
                    'refined_prediction': refined_prediction,
                    'final_prediction': final_prediction
                }
                
                with open(output_dir / f"{image_file.stem}_prediction.json", 'w') as f:
                    json.dump(result, f, indent=2)
                
                results.append(result)
                
            except Exception as e:
                print(f"✗ Error processing {image_file.name}: {e}")
                continue
        
        print(f"✓ Processed {len(results)}/{len(image_files)} images")
        return results
    
    def process_all_folders(self, input_dir, output_dir, use_refinement=False,
                           create_visualizations=True):
        """
        Process all building folders in input directory
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        building_folders = [d for d in input_dir.iterdir() if d.is_dir()]
        
        if not building_folders:
            print(f"⚠️  No building folders found in {input_dir}")
            return
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING")
        print(f"{'='*70}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Building folders: {len(building_folders)}")
        print(f"Refinement: {'Enabled' if use_refinement else 'Disabled'}")
        print(f"{'='*70}\n")
        
        all_results = {}
        
        for folder in building_folders:
            building_name = folder.name
            results = self.process_building_folder(
                folder, building_name, output_dir,
                use_refinement, create_visualizations
            )
            all_results[building_name] = results
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_buildings': len(building_folders),
            'total_images': sum(len(r) for r in all_results.values()),
            'model_path': str(self.model_path) if hasattr(self, 'model_path') else 'unknown',
            'refinement_used': use_refinement,
            'results': all_results
        }
        
        summary_path = output_dir / 'predictions.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total buildings: {len(building_folders)}")
        print(f"Total images: {summary['total_images']}")
        print(f"Results saved to: {output_dir}")
        print(f"Summary: {summary_path}")
        print(f"{'='*70}\n")
        

def main():
    parser = argparse.ArgumentParser(
        description='Batch inference for building pose estimation'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing building folders')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON')
    parser.add_argument('--mesh-dir', type=str, required=True,
                       help='Directory containing GLB files')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--refine', action='store_true',
                       help='Apply pose refinement (slower but more accurate)')
    parser.add_argument('--no-vis', action='store_true',
                       help='Skip visualization (predictions only)')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Model input size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create estimator
    estimator = BatchBuildingPoseEstimator(
        model_path=args.model,
        config_path=args.config,
        mesh_dir=args.mesh_dir,
        device=args.device,
        image_size=args.image_size
    )
    estimator.model_path = args.model  # Store for summary
    
    # Process all folders
    estimator.process_all_folders(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_refinement=args.refine,
        create_visualizations=not args.no_vis
    )


if __name__ == "__main__":
    main()