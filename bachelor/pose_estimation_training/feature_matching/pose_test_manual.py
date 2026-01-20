import os
import json
import numpy as np
import cv2
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from common.matchers import EnsembleMatcher
from common.pose_utils import (
    PoseResult,
    estimate_camera_intrinsics,
    find_3d_correspondences,
    solve_pnp_ransac,
    load_render_image
)
from common.visualization import visualize_pose_with_bbox


class ManualPoseEstimator:
    """
    Estimate camera pose for manually specified query and render.
    """

    def __init__(self, database_path: str, renders_root: str):
        print(f"Loading database from {database_path}")
        with open(database_path, 'rb') as f:
            self.database = pickle.load(f)

        self.renders_root = Path(renders_root)
        if not self.renders_root.exists():
            print(f"Warning: Renders directory {renders_root} does not exist!")

        print(f"Loaded {len(self.database['model_names'])} models")

        print("Initializing Ensemble Matcher (LoFTR + SuperPoint + SIFT)...")
        self.ensemble_matcher = EnsembleMatcher()

    def estimate_pose(self,
                     query_image: np.ndarray,
                     model_name: str,
                     render_idx: int,
                     min_inliers: int = 10) -> PoseResult:
        """
        Estimate pose for specified query image and render.

        Args:
            query_image: Query image (BGR)
            model_name: Name of model in database
            render_idx: Index of render to use
            min_inliers: Minimum inliers required for valid pose

        Returns:
            PoseResult object
        """
        if model_name not in self.database['model_names']:
            print(f"Error: Model '{model_name}' not found in database!")
            print(f"Available models: {self.database['model_names']}")
            return PoseResult(
                False, "", 0.0, None, None, None, None, 0, float('inf'), -1
            )

        # 1. Estimate intrinsics
        K = estimate_camera_intrinsics(query_image.shape[:2])
        print(f"Estimated intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

        # 2. Load render image
        print(f"\nLoading render: {model_name}, index {render_idx}")
        render_img = load_render_image(self.renders_root, model_name, render_idx)

        if render_img is None:
            print("Error: Could not load render image")
            return PoseResult(
                False, model_name, 0.0, K, None, None, None, 0, float('inf'), render_idx
            )

        # 3. Run ensemble matching
        print("\nPerforming ensemble matching...")
        pts_query, pts_render, methods = self.ensemble_matcher.match_ensemble(
            query_image, render_img
        )

        num_matches = len(pts_query)
        print(f"Found {num_matches} total matches")

        if num_matches < 10:
            print("Error: Insufficient matches for pose estimation")
            return PoseResult(
                False, model_name, 0.0, K, None, None, None, 0, float('inf'),
                render_idx, num_matches=num_matches
            )

        method_counts = {}
        for method in set(methods):
            method_counts[method] = methods.count(method)
        print(f"Match breakdown: {method_counts}")

        # 4. Get 3D correspondences
        print("\nFinding 3D correspondences...")
        model_data = self.database['models'][model_name]
        render_data = model_data['renders'][render_idx]

        pts_2d, pts_3d = find_3d_correspondences(
            pts_query, pts_render, render_data, model_data
        )

        num_3d = len(pts_2d)

        if num_3d < 6:
            print(f"Error: Insufficient 3D correspondences ({num_3d})")
            return PoseResult(
                False, model_name, 0.0, K, None, None, None, 0, float('inf'),
                render_idx, num_matches=num_matches, num_3d_correspondences=num_3d
            )

        # 5. Solve PnP
        print("\nSolving PnP...")
        success, R, t, num_inliers, error = solve_pnp_ransac(pts_2d, pts_3d, K)

        if not success or num_inliers < min_inliers:
            print(f"Error: PnP failed (inliers={num_inliers}, required={min_inliers})")
            return PoseResult(
                False, model_name, 0.0, K, R, t, None, num_inliers, error,
                render_idx, num_matches=num_matches, num_3d_correspondences=num_3d
            )

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        used_methods = methods[:num_3d]
        dom_method = max(set(used_methods), key=used_methods.count) if used_methods else "mixed"

        print(f"\n✓ Pose estimation successful!")
        print(f"  Inliers: {num_inliers}")
        print(f"  Reprojection error: {error:.2f} px")
        print(f"  Dominant method: {dom_method}")

        return PoseResult(
            success=True,
            model_name=model_name,
            confidence=float(num_inliers) / 100.0,
            intrinsic_K=K,
            rotation_R=R,
            translation_t=t,
            transform_4x4=T,
            num_inliers=num_inliers,
            reprojection_error=error,
            matched_render_idx=render_idx,
            method=dom_method,
            num_matches=num_matches,
            num_3d_correspondences=num_3d
        )

    def visualize_pose(self,
                      image: np.ndarray,
                      result: PoseResult,
                      output_path: str):
        """
        Visualize the estimated pose with 3D bounding box.
        """
        if not result.success:
            cv2.imwrite(output_path, image)
            print(f"Saved original image (pose failed) to: {output_path}")
            return

        model_data = self.database['models'][result.model_name]
        visualize_pose_with_bbox(image, result, model_data['bounds'], output_path)
        print(f"Saved visualization to: {output_path}")


def main():
    QUERY_IMAGE = "/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/test_data_test3/Ilmet/image1.JPG"
    DATABASE_PATH = "./features_orbital/feature_database.pkl"
    RENDERS_DIR = "/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/AR_Renders_Orbital"

    OUTPUT_VIS = "./tests/pose_visualization.jpg"
    OUTPUT_JSON = "./tests/pose_data.json"

    MODEL_NAME = "Ilmet"
    RENDER_IDX = 23

    MIN_INLIERS = 4

    if not os.path.exists(QUERY_IMAGE):
        print(f"Error: Query image not found: {QUERY_IMAGE}")
        return

    if not os.path.exists(DATABASE_PATH):
        print(f"Error: Database not found: {DATABASE_PATH}")
        return

    if not os.path.exists(RENDERS_DIR):
        print(f"Error: Renders directory not found: {RENDERS_DIR}")
        return

    os.makedirs(os.path.dirname(OUTPUT_VIS), exist_ok=True)

    print(f"Loading query image: {QUERY_IMAGE}")
    query_image = cv2.imread(QUERY_IMAGE)

    if query_image is None:
        print(f"Error: Could not load query image")
        return

    print(f"Query image shape: {query_image.shape}")

    estimator = ManualPoseEstimator(DATABASE_PATH, RENDERS_DIR)

    print("\n" + "="*60)
    print("STARTING POSE ESTIMATION")
    print("="*60)

    result = estimator.estimate_pose(
        query_image,
        MODEL_NAME,
        RENDER_IDX,
        min_inliers=MIN_INLIERS
    )

    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    estimator.visualize_pose(query_image, result, OUTPUT_VIS)

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Saved pose data to: {OUTPUT_JSON}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Success: {result.success}")
    print(f"Model: {result.model_name}")
    print(f"Render: {result.matched_render_idx}")
    print(f"Matches: {result.num_matches}")
    print(f"3D correspondences: {result.num_3d_correspondences}")
    print(f"Inliers: {result.num_inliers}")
    print(f"Reprojection error: {result.reprojection_error:.2f} px")
    print(f"Method: {result.method}")
    print("="*60)


if __name__ == "__main__":
    main()
