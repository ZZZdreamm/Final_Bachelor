"""
Shared feature matching classes for pose estimation.
Provides LoFTR, SuperPoint, and SIFT ensemble matching.
"""

import numpy as np
import cv2
from typing import Tuple, List

# Optional imports with availability flags
LOFTR_AVAILABLE = False
try:
    import torch
    from kornia.feature import LoFTR
    LOFTR_AVAILABLE = True
except ImportError:
    print("Warning: Kornia not available. LoFTR disabled.")

SUPERPOINT_AVAILABLE = False
try:
    from lightglue import SuperPoint
    SUPERPOINT_AVAILABLE = True
except ImportError:
    print("Warning: LightGlue not available. SuperPoint disabled.")


class LoFTRMatcher:
    """LoFTR-based feature matching (best for cross-domain matching)"""

    def __init__(self, device='cuda'):
        if not LOFTR_AVAILABLE:
            self.model = None
            return
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = LoFTR(pretrained='outdoor').to(self.device).eval()

    def match(self, img1: np.ndarray, img2: np.ndarray,
              conf_threshold: float = 0.2, max_dim: int = 840) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match features between two images using LoFTR.

        Args:
            img1: First image (BGR)
            img2: Second image (BGR)
            conf_threshold: Confidence threshold for matches
            max_dim: Maximum image dimension (larger images are resized)

        Returns:
            Tuple of (points1, points2, confidences)
        """
        if self.model is None:
            return np.array([]), np.array([]), np.array([])

        # Convert to grayscale
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        # Compute scaling factors
        h1, w1 = g1.shape
        h2, w2 = g2.shape
        s1 = min(max_dim / max(h1, w1), 1.0)
        s2 = min(max_dim / max(h2, w2), 1.0)

        # Resize if needed
        g1_r = cv2.resize(g1, None, fx=s1, fy=s1) if s1 < 1 else g1
        g2_r = cv2.resize(g2, None, fx=s2, fy=s2) if s2 < 1 else g2

        # Convert to tensors
        t1 = torch.from_numpy(g1_r).float()[None, None].to(self.device) / 255.0
        t2 = torch.from_numpy(g2_r).float()[None, None].to(self.device) / 255.0

        # Run LoFTR
        with torch.no_grad():
            batch = {'image0': t1, 'image1': t2}
            output = self.model(batch)
            pts1 = output['keypoints0'].cpu().numpy()[0]
            pts2 = output['keypoints1'].cpu().numpy()[0]
            conf = output['confidence'].cpu().numpy()[0]

        # Filter by confidence and scale back to original size
        mask = conf > conf_threshold
        return pts1[mask] / s1, pts2[mask] / s2, conf[mask]


class SuperPointMatcher:
    """SuperPoint-based feature matching"""

    def __init__(self, device='cuda', max_keypoints=2048):
        if not SUPERPOINT_AVAILABLE:
            self.model = None
            return
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = SuperPoint(max_num_keypoints=max_keypoints).to(self.device).eval()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract SuperPoint keypoints and descriptors.

        Args:
            image: Input image (BGR)

        Returns:
            Tuple of (keypoints, descriptors)
        """
        if self.model is None:
            return None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        tensor = torch.from_numpy(gray).float()[None, None].to(self.device) / 255.0

        with torch.no_grad():
            out = self.model({'image': tensor})

        return out['keypoints'][0].cpu().numpy(), out['descriptors'][0].cpu().numpy()

    def match(self, img1: np.ndarray, img2: np.ndarray,
              ratio_thresh: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match features between two images using SuperPoint.

        Args:
            img1: First image (BGR)
            img2: Second image (BGR)
            ratio_thresh: Lowe's ratio test threshold

        Returns:
            Tuple of (points1, points2)
        """
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
    """Combines LoFTR, SuperPoint, and SIFT for robust matching"""

    def __init__(self, device='cuda'):
        self.loftr = LoFTRMatcher(device)
        self.superpoint = SuperPointMatcher(device)
        self.sift = cv2.SIFT_create(nfeatures=5000)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    def match_ensemble(self, query: np.ndarray, render: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Match features using ensemble of LoFTR, SuperPoint, and SIFT.

        Args:
            query: Query image (BGR)
            render: Render image (BGR)

        Returns:
            Tuple of (query_points, render_points, methods_list)
            methods_list indicates which method produced each match
        """
        all_q, all_r, methods = [], [], []

        # 1. LoFTR matches
        l_q, l_r, _ = self.loftr.match(query, render)
        if len(l_q) > 0:
            all_q.extend(l_q)
            all_r.extend(l_r)
            methods.extend(['loftr'] * len(l_q))

        # 2. SuperPoint matches
        sp_q, sp_r = self.superpoint.match(query, render)
        if len(sp_q) > 0:
            all_q.extend(sp_q)
            all_r.extend(sp_r)
            methods.extend(['superpoint'] * len(sp_q))

        # 3. SIFT matches
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
