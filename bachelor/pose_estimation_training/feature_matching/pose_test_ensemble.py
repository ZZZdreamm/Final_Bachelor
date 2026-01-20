import os
import json
import argparse
import numpy as np
import cv2
import pickle
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from common.matchers import EnsembleMatcher
from common.pose_utils import (
    PoseResult,
    estimate_camera_intrinsics,
    find_3d_correspondences,
    solve_pnp_ransac,
    load_render_image
)
from common.visualization import visualize_pose_with_bbox
from common.retrieval import GlobalRetriever
from extract_features import FeatureExtractor

class PoseEstimator:
    """
    Estimate camera pose using PnP algorithm with Ensemble Matching.
    """

    def __init__(self, database_path: str, renders_root: str):
        print(f"Loading database from {database_path}")
        with open(database_path, 'rb') as f:
            self.database = pickle.load(f)

        self.renders_root = Path(renders_root)
        if not self.renders_root.exists():
            print(f"Warning: Renders directory {renders_root} does not exist!")

        print(f"Loaded {len(self.database['model_names'])} models")

        if FeatureExtractor:
            self.global_extractor = FeatureExtractor(use_superpoint=False, use_sift=False, use_global=True)
        else:
            self.global_extractor = None

        self.retriever = GlobalRetriever(self.database)

        print("Initializing Ensemble Matcher (LoFTR + SuperPoint + SIFT)...")
        self.ensemble_matcher = EnsembleMatcher()

    def estimate_pose(self, image: np.ndarray, top_k_retrieval: int = 5, min_inliers: int = 10) -> PoseResult:
        """
        Estimate camera pose from query image.

        Args:
            image: Query image (BGR)
            top_k_retrieval: Number of candidate models to try
            min_inliers: Minimum inliers required for valid pose

        Returns:
            PoseResult object
        """
        # 1. Estimate camera intrinsics
        K = estimate_camera_intrinsics(image.shape[:2])

        # 2. Global Retrieval (Find Candidates)
        print("Retrieving candidates...")
        query_desc = None
        if self.global_extractor:
            feats = self.global_extractor.extract_all(image)
            if 'global' in feats:
                query_desc = feats['global']['descriptor']

        candidates = self.retriever.retrieve(query_desc, top_k=top_k_retrieval)
        print(f"Top candidates: {[c[0] for c in candidates]}")

        # 3. Ensemble Matching & PnP
        best_result = None
        best_inliers = 0

        for model_name, render_idx, global_score in candidates:
            render_img = load_render_image(self.renders_root, model_name, render_idx)
            if render_img is None:
                print(f"Skipping {model_name} render {render_idx}: Image not found")
                continue

            # Run Ensemble Matching
            pts_query, pts_render, methods = self.ensemble_matcher.match_ensemble(image, render_img)

            if len(pts_query) < 10:
                continue

            model_data = self.database['models'][model_name]
            render_data = model_data['renders'][render_idx]

            # Convert Render Pixels -> 3D World Points
            pts_2d, pts_3d = find_3d_correspondences(pts_query, pts_render, render_data, model_data)

            if len(pts_2d) < 6:
                continue

            # Solve PnP
            success, R, t, num_inliers, error = solve_pnp_ransac(pts_2d, pts_3d, K)
            print(f"Points 2d: {len(pts_2d)}, Points 3d: {len(pts_3d)}")
            print(f"  Tried {model_name} render {render_idx}: {num_inliers} inliers")

            if success and num_inliers > best_inliers:
                best_inliers = num_inliers
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.flatten()

                used_methods = methods[:len(pts_2d)]
                dom_method = max(set(used_methods), key=used_methods.count) if used_methods else "mixed"

                best_result = PoseResult(
                    success=True, model_name=model_name, confidence=global_score * (num_inliers/100),
                    intrinsic_K=K, rotation_R=R, translation_t=t, transform_4x4=T,
                    num_inliers=num_inliers, reprojection_error=error, matched_render_idx=render_idx,
                    method=dom_method
                )
                print(f" > Match {model_name}: {num_inliers} inliers ({dom_method})")

        if best_result is None or best_result.num_inliers < min_inliers:
            return PoseResult(False, "", 0.0, K, None, None, None, 0, float('inf'), -1)

        return best_result

    def visualize_pose(self, image: np.ndarray, result: PoseResult, output_path: str):
        """Visualize the 3D bounding box on image."""
        if not result.success:
            cv2.imwrite(output_path, image)
            return

        model_data = self.database['models'][result.model_name]
        visualize_pose_with_bbox(image, result, model_data['bounds'], output_path)


@dataclass
class BatchStatistics:
    total_images: int = 0
    successful_poses: int = 0
    correct_identifications: int = 0
    failed: int = 0
    building_stats: Dict[str, Dict] = field(default_factory=dict)


def process_batch(input_root: str, database_path: str, renders_root: str, output_root: Optional[str] = None):
    print("--- Initializing Pipeline ---")
    estimator = PoseEstimator(database_path, renders_root)

    stats = BatchStatistics()
    input_path = Path(input_root)
    building_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not building_dirs:
        print(f"No subdirectories found in {input_root}")
        return

    for building_dir in building_dirs:
        ground_truth_name = building_dir.name
        print(f"\nProcessing Building: {ground_truth_name}")
        stats.building_stats[ground_truth_name] = {'total': 0, 'correct': 0}

        images = [f for f in building_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

        for img_path in tqdm(images, desc=f"Images in {ground_truth_name}"):
            stats.total_images += 1
            stats.building_stats[ground_truth_name]['total'] += 1

            image = cv2.imread(str(img_path))
            if image is None: continue

            result = estimator.estimate_pose(image)

            is_correct_building = False
            if result.success:
                stats.successful_poses += 1
                if result.model_name == ground_truth_name:
                    stats.correct_identifications += 1
                    stats.building_stats[ground_truth_name]['correct'] += 1
                    is_correct_building = True
            else:
                stats.failed += 1

            if output_root:
                out_building_dir = Path(output_root) / ground_truth_name
                out_building_dir.mkdir(parents=True, exist_ok=True)
                base_name = img_path.stem

                vis_path = out_building_dir / f"{base_name}_pose.jpg"
                estimator.visualize_pose(image, result, str(vis_path))

                json_path = out_building_dir / f"{base_name}_data.json"
                out_data = result.to_dict()
                out_data['ground_truth'] = ground_truth_name
                out_data['correct_identification'] = is_correct_building
                with open(json_path, 'w') as f:
                    json.dump(out_data, f, indent=2)

    print("\n" + "="*50)
    print("BATCH PROCESSING SUMMARY")
    print("="*50)
    print(f"Total Images: {stats.total_images}")
    print(f"Successful Poses: {stats.successful_poses}")
    if stats.total_images > 0:
        print(f"Identification Accuracy: {(stats.correct_identifications / stats.total_images) * 100:.2f}%")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='AR Building Pose Estimation - Batch Processing')
    parser.add_argument('--input_dir', type=str, required=True, help='Root directory containing building folders')
    parser.add_argument('--database', type=str, required=True, help='Path to feature database (feature_database.pkl)')
    parser.add_argument('--renders_dir', type=str, required=True, help='Directory containing rendered images for LoFTR/Ensemble matching')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    process_batch(args.input_dir, args.database, args.renders_dir, args.output_dir)


if __name__ == "__main__":
    main()
