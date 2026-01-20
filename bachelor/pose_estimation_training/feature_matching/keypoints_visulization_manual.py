import os
import numpy as np
import cv2
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import random

LOFTR_AVAILABLE = False
try:
    import torch
    from kornia.feature import LoFTR
    LOFTR_AVAILABLE = True
except ImportError:
    print("Warning: Kornia not available. LoFTR disabled.")

try:
    from lightglue import SuperPoint
    SUPERPOINT_AVAILABLE = True
except ImportError:
    SUPERPOINT_AVAILABLE = False
    print("Warning: LightGlue not available. SuperPoint disabled.")



class LoFTRMatcher:
    """LoFTR-based feature matching (best for cross-domain)"""
    def __init__(self, device='cuda'):
        if not LOFTR_AVAILABLE:
            self.model = None
            return
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = LoFTR(pretrained='outdoor').to(self.device).eval()
    
    def match(self, img1: np.ndarray, img2: np.ndarray, conf_threshold: float = 0.01, max_dim: int = 1200):
        if self.model is None: 
            return np.array([]), np.array([]), np.array([])
        
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        if np.mean(g1) < 128: 
            g1 = 255 - g1
        if np.mean(g2) < 128:
            g2 = 255 - g2
        
        g1 = cv2.GaussianBlur(g1, (3, 3), 0.5)
        g2 = cv2.GaussianBlur(g2, (3, 3), 0.5)
        
        h1, w1 = g1.shape
        h2, w2 = g2.shape
        s1 = min(max_dim / max(h1, w1), 1.0)
        s2 = min(max_dim / max(h2, w2), 1.0)
        
        g1_r = cv2.resize(g1, None, fx=s1, fy=s1) if s1 < 1 else g1
        g2_r = cv2.resize(g2, None, fx=s2, fy=s2) if s2 < 1 else g2
        
        t1 = torch.from_numpy(g1_r).float()[None, None].to(self.device) / 255.0
        t2 = torch.from_numpy(g2_r).float()[None, None].to(self.device) / 255.0
        
        with torch.no_grad():
            batch = {'image0': t1, 'image1': t2}
            output = self.model(batch)
            pts1 = output['keypoints0'].cpu().numpy()[0]
            pts2 = output['keypoints1'].cpu().numpy()[0]
            conf = output['confidence'].cpu().numpy()[0]

        mask = conf > conf_threshold
        return pts1[mask] / s1, pts2[mask] / s2, conf[mask]


class SuperPointMatcher:
    def __init__(self, device='cuda', max_keypoints=2048):
        if not SUPERPOINT_AVAILABLE:
            self.model = None
            return
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = SuperPoint(max_num_keypoints=max_keypoints).to(self.device).eval()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def extract(self, image):
        if self.model is None: 
            return None, None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        tensor = torch.from_numpy(gray).float()[None, None].to(self.device) / 255.0
        with torch.no_grad():
            out = self.model({'image': tensor})
        return out['keypoints'][0].cpu().numpy(), out['descriptors'][0].cpu().numpy()

    def match(self, img1, img2, ratio_thresh=0.8):
        if self.model is None: 
            return np.array([]), np.array([])
        k1, d1 = self.extract(img1)
        k2, d2 = self.extract(img2)
        if k1 is None or k2 is None or len(k1) < 2 or len(k2) < 2: 
            return np.array([]), np.array([])
        
        matches = self.bf_matcher.knnMatch(d1.astype(np.float32), d2.astype(np.float32), k=2)
        p1, p2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < ratio_thresh * m[1].distance:
                p1.append(k1[m[0].queryIdx])
                p2.append(k2[m[0].trainIdx])
        return np.array(p1), np.array(p2)


class EnsembleMatcher:
    """Combines LoFTR, SuperPoint, and SIFT"""
    def __init__(self, device='cuda'):
        self.loftr = LoFTRMatcher(device)
        self.superpoint = SuperPointMatcher(device)
        self.sift = cv2.SIFT_create(nfeatures=5000)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        
    def match_ensemble(self, query: np.ndarray, render: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        all_q, all_r, methods = [], [], []
        
        # 1. LoFTR
        l_q, l_r, _ = self.loftr.match(query, render)
        if len(l_q) > 0:
            all_q.extend(l_q)
            all_r.extend(l_r)
            methods.extend(['loftr'] * len(l_q))
            
        # 2. SuperPoint
        sp_q, sp_r = self.superpoint.match(query, render)
        if len(sp_q) > 0:
            all_q.extend(sp_q)
            all_r.extend(sp_r)
            methods.extend(['superpoint'] * len(sp_q))
            
        # 3. SIFT
        gq = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(render, cv2.COLOR_BGR2GRAY)
        kq, dq = self.sift.detectAndCompute(gq, None)
        kr, dr = self.sift.detectAndCompute(gr, None)
        
        if kq is not None and kr is not None and len(kq) > 2 and len(kr) > 2:
            matches = self.flann.knnMatch(dq, dr, k=2)
            s_q, s_r = [], []
            for m in matches:
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                    s_q.append(kq[m[0].queryIdx].pt)
                    s_r.append(kr[m[0].trainIdx].pt)
            if len(s_q) > 0:
                all_q.extend(s_q)
                all_r.extend(s_r)
                methods.extend(['sift'] * len(s_q))

        return np.array(all_q), np.array(all_r), methods


class MatchVisualizer:
    """
    Visualize feature matches between query and specified database render.
    """
    
    def __init__(self, database_path: str, renders_dir: str):
        # Load database
        print(f"Loading database from {database_path}")
        with open(database_path, 'rb') as f:
            self.database = pickle.load(f)
        
        self.renders_dir = Path(renders_dir)
        if not self.renders_dir.exists():
            print(f"Warning: Renders directory {renders_dir} does not exist!")
        
        print(f"Loaded {len(self.database['model_names'])} models")
        
        print("Initializing Ensemble Matcher (LoFTR + SuperPoint + SIFT)...")
        self.ensemble_matcher = EnsembleMatcher()
    
    def load_render_image(self, model_name: str, render_idx: int) -> Optional[np.ndarray]:
        """
        Load the render image from disk.
        """
        render_filename = f"render_{render_idx:04d}.png"
        
        path = self.renders_dir / model_name / "images" / render_filename
        
        if path.exists():
            image = cv2.imread(str(path))
            if image is not None:
                return image
        
        path = self.renders_dir / model_name / render_filename
        if path.exists():
            image = cv2.imread(str(path))
            if image is not None:
                return image
        
        print(f"Warning: Could not find render image: {model_name}/images/{render_filename}")
        return None
    
    def visualize_matches(self,
                         query_image: np.ndarray,
                         render_image: np.ndarray,
                         match_data: Dict,
                         output_path: str,
                         max_matches: int = 100):
        """
        Create visualization showing matched keypoints with colored lines.
        
        Args:
            query_image: Query image (BGR)
            render_image: Matched render image (BGR)
            match_data: Dictionary containing points and methods from ensemble matcher
            output_path: Where to save visualization
            max_matches: Maximum number of matches to display
        """
        points_query = match_data['points_query']
        points_render = match_data['points_render']
        methods = match_data['methods']
        
        h1, w1 = query_image.shape[:2]
        h2, w2 = render_image.shape[:2]
        
        target_height = 800
        scale1 = target_height / h1
        scale2 = target_height / h2
        
        query_resized = cv2.resize(query_image, (int(w1 * scale1), target_height))
        render_resized = cv2.resize(render_image, (int(w2 * scale2), target_height))
        
        w_total = query_resized.shape[1] + render_resized.shape[1]
        vis_image = np.zeros((target_height, w_total, 3), dtype=np.uint8)
        
        vis_image[:, :query_resized.shape[1]] = query_resized
        vis_image[:, query_resized.shape[1]:] = render_resized
        
        offset_x = query_resized.shape[1]
        
        num_matches = len(points_query)
        if num_matches > max_matches:
            indices = random.sample(range(num_matches), max_matches)
            points_query = points_query[indices]
            points_render = points_render[indices]
            methods = [methods[i] for i in indices]
        
        method_colors = {
            'loftr': (0, 0, 255),     
            'superpoint': (0, 255, 255),  
            'sift': (255, 0, 255)     
        }
        
        for i, (pt_q, pt_r, method) in enumerate(zip(points_query, points_render, methods)):
            pt1 = (int(pt_q[0] * scale1), int(pt_q[1] * scale1))
            pt2 = (int(pt_r[0] * scale2) + offset_x, int(pt_r[1] * scale2))
            
            color = method_colors.get(method, (0, 255, 0))
            
            cv2.line(vis_image, pt1, pt2, color, 1, cv2.LINE_AA)
            
            cv2.circle(vis_image, pt1, 3, color, -1, cv2.LINE_AA)
            cv2.circle(vis_image, pt2, 3, color, -1, cv2.LINE_AA)
        
        cv2.putText(vis_image, "Query Image", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_image, "Database Render", (offset_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        method_counts = match_data['method_counts']
        y_pos = 60
        for method, color in method_colors.items():
            count = method_counts.get(method, 0)
            if count > 0:
                cv2.putText(vis_image, f"{method}: {count}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                y_pos += 25
        
        total = sum(method_counts.values())
        cv2.putText(vis_image, f"Total: {total}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        legend_y = target_height - 100
        cv2.putText(vis_image, "Legend:", (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for i, (method, color) in enumerate(method_colors.items()):
            cv2.putText(vis_image, f"{method}", (10, legend_y + 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        cv2.imwrite(output_path, vis_image)
        print(f"\nVisualization saved to: {output_path}")
    
    def process_query(self,
                     query_image_path: str,
                     model_name: str,
                     render_idx: int,
                     output_path: str,
                     max_matches: int = 100):
        """
        Main processing pipeline with manual model/render specification.
        
        Args:
            query_image_path: Path to query image
            model_name: Name of model in database
            render_idx: Index of render to match against
            output_path: Where to save visualization
            max_matches: Maximum number of matches to display
        """
        if model_name not in self.database['model_names']:
            print(f"Error: Model '{model_name}' not found in database!")
            print(f"Available models: {self.database['model_names']}")
            return
        
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            print(f"Error: Could not load image {query_image_path}")
            return
        
        print(f"Query image shape: {query_image.shape}")
        
        print(f"\nLoading render: {model_name}, render {render_idx}")
        render_image = self.load_render_image(model_name, render_idx)
        
        if render_image is None:
            print("Error: Could not load render image")
            return
        
        print(f"Render image shape: {render_image.shape}")
        
        print("\nPerforming ensemble matching...")
        pts_query, pts_render, methods = self.ensemble_matcher.match_ensemble(
            query_image, render_image
        )
        
        total_matches = len(pts_query)
        
        if total_matches == 0:
            print("No matches found!")
            return
        
        method_counts = {}
        for method in set(methods):
            method_counts[method] = methods.count(method)
        
        match_data = {
            'points_query': pts_query,
            'points_render': pts_render,
            'methods': methods,
            'method_counts': method_counts
        }
        
        print(f"\nTotal matches: {total_matches}")
        for method, count in method_counts.items():
            print(f"  - {method}: {count}")
        
        print(f"\nCreating visualization...")
        self.visualize_matches(
            query_image,
            render_image,
            match_data,
            output_path,
            max_matches=max_matches
        )
        
        info_path = output_path.rsplit('.', 1)[0] + '_info.txt'
        with open(info_path, 'w') as f:
            f.write(f"Query Image: {query_image_path}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Render Index: {render_idx}\n")
            f.write(f"Total Matches: {total_matches}\n")
            for method, count in method_counts.items():
                f.write(f"  - {method}: {count}\n")
        
        print(f"Match info saved to: {info_path}")


def main():
    QUERY_IMAGE = "/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/test_data_test3/Ilmet/image1.JPG"
    DATABASE_PATH = "./features_orbital/feature_database.pkl"
    RENDERS_DIR = "/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/AR_Renders_Orbital"
    OUTPUT_PATH = "./tests/match_visualization_manual.jpg"
    
    MODEL_NAME = "Ilmet"
    RENDER_IDX = 23     
    
    MAX_MATCHES = 150 
    
    if not os.path.exists(QUERY_IMAGE):
        print(f"Error: Query image not found: {QUERY_IMAGE}")
        return
    
    if not os.path.exists(DATABASE_PATH):
        print(f"Error: Database not found: {DATABASE_PATH}")
        return
    
    if not os.path.exists(RENDERS_DIR):
        print(f"Error: Renders directory not found: {RENDERS_DIR}")
        return
    
    visualizer = MatchVisualizer(DATABASE_PATH, RENDERS_DIR)
    
    visualizer.process_query(
        QUERY_IMAGE,
        MODEL_NAME,
        RENDER_IDX,
        OUTPUT_PATH,
        max_matches=MAX_MATCHES
    )


if __name__ == "__main__":
    main()