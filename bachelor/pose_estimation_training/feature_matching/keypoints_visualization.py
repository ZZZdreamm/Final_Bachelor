import os
import numpy as np
import cv2
import pickle
from typing import Dict, Tuple
from pathlib import Path
from common.matchers import EnsembleMatcher
from common.pose_utils import load_render_image
from common.visualization import draw_match_visualization
from common.retrieval import GlobalRetriever
from extract_features import FeatureExtractor



class MatchVisualizer:
    """
    Visualize feature matches between query and database images.
    """

    def __init__(self, database_path: str, renders_dir: str):
        print(f"Loading database from {database_path}")
        with open(database_path, 'rb') as f:
            self.database = pickle.load(f)

        self.renders_dir = Path(renders_dir)
        if not self.renders_dir.exists():
            print(f"Warning: Renders directory {renders_dir} does not exist!")

        print(f"Loaded {len(self.database['model_names'])} models")

        if FeatureExtractor:
            self.global_extractor = FeatureExtractor(
                use_superpoint=False,
                use_sift=False,
                use_global=True
            )
        else:
            self.global_extractor = None

        print("Initializing Ensemble Matcher (LoFTR + SuperPoint + SIFT)...")
        self.ensemble_matcher = EnsembleMatcher()

        self.retriever = GlobalRetriever(self.database)

    def find_best_match(self,
                       query_image: np.ndarray,
                       top_k_retrieval: int = 20) -> Tuple[str, int, Dict, int]:
        """
        Find the best matching render for the query image using ensemble matching.

        Returns:
            model_name: Name of matched model
            render_idx: Index of matched render
            match_data: Dictionary containing matches and method info
            total_matches: Total number of matches
        """
        print("Extracting global features from query image...")

        # Extract global descriptor for retrieval
        query_descriptor = None
        if self.global_extractor:
            features = self.global_extractor.extract_all(query_image)
            if 'global' in features and features['global']:
                query_descriptor = features['global']['descriptor']

        # Retrieve candidates using global descriptor
        print("Retrieving candidate renders...")
        candidates = self.retriever.retrieve(query_descriptor, top_k=top_k_retrieval)

        print(f"Top candidates: {[(c[0], c[2]) for c in candidates[:5]]}")

        # Find best match by using ensemble matcher
        best_model = None
        best_render_idx = -1
        best_match_data = None
        best_count = 0

        for model_name, render_hint, global_score in candidates:
            render_image = load_render_image(self.renders_dir, model_name, render_hint)
            if render_image is None:
                print(f"Skipping {model_name} render {render_hint}: Image not found")
                continue

            # Run ensemble matching (LoFTR + SuperPoint + SIFT)
            pts_query, pts_render, methods = self.ensemble_matcher.match_ensemble(
                query_image, render_image
            )

            total = len(pts_query)

            if total > best_count:
                best_count = total
                best_model = model_name
                best_render_idx = render_hint

                method_counts = {}
                for method in set(methods):
                    method_counts[method] = methods.count(method)

                best_match_data = {
                    'points_query': pts_query,
                    'points_render': pts_render,
                    'methods': methods,
                    'method_counts': method_counts
                }

        if best_match_data:
            print(f"\nBest match: {best_model}, render {best_render_idx}")
            print(f"Total matches: {best_count}")
            for method, count in best_match_data['method_counts'].items():
                print(f"  - {method}: {count}")

        return best_model, best_render_idx, best_match_data, best_count

    def process_query(self,
                     query_image_path: str,
                     output_path: str,
                     max_matches: int = 100):
        """
        Main processing pipeline.
        """
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            print(f"Error: Could not load image {query_image_path}")
            return

        print(f"Query image shape: {query_image.shape}")

        model_name, render_idx, match_data, total_matches = self.find_best_match(query_image)

        if match_data is None or total_matches == 0:
            print("No matches found!")
            return

        print(f"\nLoading render image...")
        render_image = load_render_image(self.renders_dir, model_name, render_idx)

        if render_image is None:
            print("Error: Could not load render image")
            return

        print(f"Render image shape: {render_image.shape}")

        print(f"\nCreating visualization...")
        draw_match_visualization(
            query_image,
            render_image,
            match_data['points_query'],
            match_data['points_render'],
            match_data['methods'],
            output_path,
            max_matches=max_matches
        )
        print(f"Visualization saved to: {output_path}")

        info_path = output_path.rsplit('.', 1)[0] + '_info.txt'
        with open(info_path, 'w') as f:
            f.write(f"Query Image: {query_image_path}\n")
            f.write(f"Matched Model: {model_name}\n")
            f.write(f"Matched Render: {render_idx}\n")
            f.write(f"Total Matches: {total_matches}\n")
            for method, count in match_data['method_counts'].items():
                f.write(f"  - {method}: {count}\n")

        print(f"Match info saved to: {info_path}")


def main():
    QUERY_IMAGE = "/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/test_data_test3/Ilmet/image1.JPG"
    DATABASE_PATH = "./features_orbital/feature_database.pkl"
    RENDERS_DIR = "/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/AR_Renders_Orbital"
    OUTPUT_PATH = "./tests/match_visualization_db_new.jpg"

    MAX_MATCHES = 50

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
        OUTPUT_PATH,
        max_matches=MAX_MATCHES
    )


if __name__ == "__main__":
    main()
