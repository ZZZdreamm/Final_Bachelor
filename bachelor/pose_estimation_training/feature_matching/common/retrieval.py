"""
Global descriptor-based retrieval for fast candidate selection.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class GlobalRetriever:
    """
    Fast retrieval of candidate models using global image descriptors.

    Uses cosine similarity between CNN-based global descriptors to
    quickly identify the most likely models before expensive feature matching.
    """

    def __init__(self, database: Dict):
        """
        Initialize retriever from feature database.

        Args:
            database: Feature database dict containing 'global_index' and 'model_names'
        """
        self.database = database

        if database.get('global_index'):
            self.global_descriptors = database['global_index']['descriptors']
            self.mapping = database['global_index']['mapping']
            self.has_index = True
        else:
            self.has_index = False
            print("Warning: No global index in database")

    def retrieve(self, query_descriptor: Optional[np.ndarray], top_k: int = 5) -> List[Tuple[str, int, float]]:
        """
        Retrieve top-k most similar renders using global descriptor.

        Args:
            query_descriptor: Global descriptor from query image (e.g., 2048-dim from ResNet)
            top_k: Number of candidates to return

        Returns:
            List of (model_name, render_idx, similarity_score) tuples, sorted by score
        """
        if not self.has_index or query_descriptor is None:
            # Fallback: return first models without scoring
            return [(name, 0, 1.0) for name in self.database['model_names'][:top_k]]

        # Normalize query descriptor
        query = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)

        # Compute cosine similarities
        similarities = np.dot(self.global_descriptors, query)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            model_name, render_idx = self.mapping[idx]
            score = similarities[idx]
            results.append((model_name, render_idx, float(score)))

        return results
