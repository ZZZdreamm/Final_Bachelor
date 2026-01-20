import torch
import torch.nn.functional as F
import numpy as np
import cv2
import trimesh
import pyrender
from scipy.spatial.transform import Rotation
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pose_network import quaternion_to_rotation_matrix
from common.mesh_utils import load_and_normalize_mesh


class PoseRefiner:
    """
    Refine pose estimation using render-and-compare optimization
    """
    
    def __init__(self, mesh_path, image_size=(1024, 1024), device='cuda'):
        self.device = device
        self.image_size = image_size

        self.mesh = load_and_normalize_mesh(mesh_path, target_scale=10.0)

        self.focal_length = 800
        self.camera_intrinsics = self._build_camera_intrinsics()
        
    def _build_camera_intrinsics(self):
        """Build camera intrinsic matrix"""
        fx = fy = self.focal_length
        cx, cy = self.image_size[0] / 2, self.image_size[1] / 2
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def render_mesh(self, R, t):
        """
        Render mesh from given pose
        
        Args:
            R: rotation matrix (3x3) numpy array
            t: translation vector (3,) numpy array
        
        Returns:
            Rendered RGB image
        """
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        
        mesh_pyrender = pyrender.Mesh.from_trimesh(self.mesh)
        scene.add(mesh_pyrender)
        
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=np.eye(4))
        
        camera = pyrender.IntrinsicsCamera(
            fx=self.camera_intrinsics[0, 0],
            fy=self.camera_intrinsics[1, 1],
            cx=self.camera_intrinsics[0, 2],
            cy=self.camera_intrinsics[1, 2]
        )
        
        # Camera pose
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R.T
        camera_pose[:3, 3] = -R.T @ t
        
        scene.add(camera, pose=camera_pose)
        
        # Render
        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.image_size[0],
            viewport_height=self.image_size[1]
        )
        
        color, depth = renderer.render(scene)
        renderer.delete()
        
        return color, depth
    
    def extract_edge_map(self, image):
        """Extract edges from image using Canny"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    def compute_perceptual_features(self, image):
        """
        Extract deep features for perceptual loss
        Uses edge detection and gradient-based features
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine features
        features = np.stack([
            gray / 255.0,
            np.abs(grad_x) / np.abs(grad_x).max(),
            np.abs(grad_y) / np.abs(grad_y).max()
        ], axis=0)
        
        return torch.from_numpy(features).float()
    
    def edge_based_loss(self, rendered_edges, target_edges):
        """
        Compute edge-based loss
        """
        rendered_edges = torch.from_numpy(rendered_edges).float() / 255.0
        target_edges = torch.from_numpy(target_edges).float() / 255.0
        
        loss = F.mse_loss(rendered_edges, target_edges)
        
        return loss
    
    def refine_pose_scipy(self, initial_rotation, initial_translation, 
                          target_image, max_iterations=100):
        """
        Refine pose using scipy optimization (gradient-free)
        
        Args:
            initial_rotation: Initial rotation quaternion (4,) [w,x,y,z]
            initial_translation: Initial translation (3,)
            target_image: Target RGB image
            max_iterations: Maximum optimization iterations
        
        Returns:
            Refined rotation (quaternion), refined translation
        """
        from scipy.optimize import minimize
        
        R_init = quaternion_to_rotation_matrix(torch.from_numpy(initial_rotation)).numpy()
        rotvec_init = Rotation.from_matrix(R_init).as_rotvec()
        
        x0 = np.concatenate([rotvec_init, initial_translation])
        
        target_edges = self.extract_edge_map(target_image)
        
        def objective(x):
            rotvec = x[:3]
            translation = x[3:]
            
            R = Rotation.from_rotvec(rotvec).as_matrix()
            
            rendered, _ = self.render_mesh(R, translation)
            rendered_edges = self.extract_edge_map(rendered)
            
            loss = np.mean((rendered_edges - target_edges) ** 2)
            
            return loss
        
        print("Refining pose...")
        result = minimize(
            objective,
            x0,
            method='Powell', 
            options={'maxiter': max_iterations, 'disp': True}
        )
        
        rotvec_refined = result.x[:3]
        translation_refined = result.x[3:]
        
        R_refined = Rotation.from_rotvec(rotvec_refined).as_matrix()
        quat_refined = Rotation.from_matrix(R_refined).as_quat()  
        quat_refined = np.array([quat_refined[3], quat_refined[0], 
                                quat_refined[1], quat_refined[2]]) 
        
        return quat_refined, translation_refined
    
    def refine_pose_grid_search(self, initial_rotation, initial_translation,
                                target_image, angle_range=10, trans_range=2.0,
                                num_samples=5):
        """
        Refine pose using grid search around initial estimate
        
        Args:
            initial_rotation: Initial rotation quaternion (4,) [w,x,y,z]
            initial_translation: Initial translation (3,)
            target_image: Target RGB image
            angle_range: Search range in degrees
            trans_range: Translation search range in meters
            num_samples: Number of samples per dimension
        
        Returns:
            Best rotation (quaternion), best translation, best score
        """
        R_init = quaternion_to_rotation_matrix(torch.from_numpy(initial_rotation)).numpy()
        rotvec_init = Rotation.from_matrix(R_init).as_rotvec()
        
        target_edges = self.extract_edge_map(target_image)
        
        best_score = float('inf')
        best_rotvec = rotvec_init
        best_trans = initial_translation
        
        angle_deltas = np.linspace(-angle_range, angle_range, num_samples) * np.pi / 180
        trans_deltas = np.linspace(-trans_range, trans_range, num_samples)
        
        total_samples = num_samples ** 6 
        print(f"Grid search: {total_samples} samples")
        
        sample_count = 0
        for dx in angle_deltas:
            for dy in angle_deltas:
                for dz in angle_deltas:
                    delta_rotvec = np.array([dx, dy, dz])
                    rotvec = rotvec_init + delta_rotvec
                    
                    for dtx in trans_deltas:
                        for dty in trans_deltas:
                            for dtz in trans_deltas:
                                delta_trans = np.array([dtx, dty, dtz])
                                translation = initial_translation + delta_trans
                                
                                R = Rotation.from_rotvec(rotvec).as_matrix()
                                
                                try:
                                    rendered, _ = self.render_mesh(R, translation)
                                    rendered_edges = self.extract_edge_map(rendered)
                                    score = np.mean((rendered_edges - target_edges) ** 2)
                                    
                                    if score < best_score:
                                        best_score = score
                                        best_rotvec = rotvec
                                        best_trans = translation
                                except:
                                    pass
                                
                                sample_count += 1
                                if sample_count % 1000 == 0:
                                    print(f"Processed {sample_count}/{total_samples} samples, best score: {best_score:.4f}")
        
        R_refined = Rotation.from_rotvec(best_rotvec).as_matrix()
        quat_refined = Rotation.from_matrix(R_refined).as_quat()  
        quat_refined = np.array([quat_refined[3], quat_refined[0], 
                                quat_refined[1], quat_refined[2]])  
        
        print(f"Grid search completed. Best score: {best_score:.4f}")
        
        return quat_refined, best_trans, best_score


def test_refinement():
    """Test the pose refinement"""
    
    refiner = PoseRefiner(
        mesh_path='path/to/building.glb',
        image_size=(1024, 1024)
    )
    
    initial_rotation = np.array([1.0, 0.0, 0.0, 0.0])  
    initial_translation = np.array([0.0, 0.0, 20.0])
    
    target_image = cv2.imread('path/to/target_image.jpg')
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    target_image = cv2.resize(target_image, (1024, 1024))
    
    refined_rot, refined_trans = refiner.refine_pose_scipy(
        initial_rotation,
        initial_translation,
        target_image,
        max_iterations=50
    )
    
    print(f"Initial rotation: {initial_rotation}")
    print(f"Refined rotation: {refined_rot}")
    print(f"Initial translation: {initial_translation}")
    print(f"Refined translation: {refined_trans}")
    
    R_init = quaternion_to_rotation_matrix(torch.from_numpy(initial_rotation)).numpy()
    R_refined = quaternion_to_rotation_matrix(torch.from_numpy(refined_rot)).numpy()
    
    rendered_init, _ = refiner.render_mesh(R_init, initial_translation)
    rendered_refined, _ = refiner.render_mesh(R_refined, refined_trans)
    
    comparison = np.hstack([target_image, rendered_init, rendered_refined])
    cv2.imwrite('pose_refinement_comparison.jpg', 
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print("Comparison saved to pose_refinement_comparison.jpg")


if __name__ == "__main__":
    test_refinement()
