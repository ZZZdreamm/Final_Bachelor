import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.pose_utils import quaternion_to_rotation_matrix as quat_to_rot_matrix


class BuildingPoseNetCached(nn.Module):
    """
    Neural network for estimating 6DoF pose from pre-extracted CLIP features.
    
    This is much faster than the full model since CLIP feature extraction
    is done offline once, and training only updates the pose heads.
    """
    
    def __init__(self, num_buildings=10, feature_dim=1280):
        super(BuildingPoseNetCached, self).__init__()
        
        self.feature_dim = feature_dim
        
        print(f"✓ Building pose network with feature dimension: {feature_dim}")
        
        self.building_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_buildings)
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4) 
        )
        
        self.translation_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """
        Args:
            features: Pre-extracted CLIP features [batch_size, feature_dim]
        
        Returns:
            Dictionary with predictions
        """
        building_logits = self.building_classifier(features)
        rotation_quat = self.rotation_head(features)
        translation = self.translation_head(features)
        confidence = self.confidence_head(features)
        
        rotation_quat = rotation_quat / (torch.norm(rotation_quat, dim=1, keepdim=True) + 1e-8)
        
        return {
            'building_logits': building_logits,
            'rotation': rotation_quat, 
            'translation': translation,
            'confidence': confidence
        }
    
    def predict_pose_only(self, features):
        """Fast inference without classification"""
        rotation_quat = self.rotation_head(features)
        translation = self.translation_head(features)
        confidence = self.confidence_head(features)
        
        rotation_quat = rotation_quat / (torch.norm(rotation_quat, dim=1, keepdim=True) + 1e-8)
        
        return rotation_quat, translation, confidence


class PoseLoss(nn.Module):
    """
    Combined loss for pose estimation training with NaN protection
    Returns None if batch contains NaN, allowing training to skip it
    """
    
    def __init__(self, rotation_weight=1.0, translation_weight=1.0, 
                 classification_weight=0.5):
        super(PoseLoss, self).__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.classification_weight = classification_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def quaternion_distance(self, q1, q2):
        """
        Compute geodesic distance between quaternions
        Returns angle in radians
        """
        # Ensure quaternions are normalized
        q1 = q1 / (torch.norm(q1, dim=1, keepdim=True) + 1e-8)
        q2 = q2 / (torch.norm(q2, dim=1, keepdim=True) + 1e-8)
        
        # Compute dot product
        dot_product = torch.abs(torch.sum(q1 * q2, dim=1))
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Geodesic distance
        return 2 * torch.acos(dot_product)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with 'rotation', 'translation', 'building_logits'
            targets: dict with 'rotation', 'translation', 'building_id'
            
        Returns:
            dict with losses, or None if NaN detected (batch will be skipped)
        """
        if torch.isnan(predictions['rotation']).any():
            print("⚠️  NaN in predicted rotation - skipping batch")
            return None
        if torch.isnan(predictions['translation']).any():
            print("⚠️  NaN in predicted translation - skipping batch")
            return None
        if torch.isnan(targets['rotation']).any():
            print("⚠️  NaN in target rotation - skipping batch")
            return None
        if torch.isnan(targets['translation']).any():
            print("⚠️  NaN in target translation - skipping batch")
            return None
        
        pred_rot = predictions['rotation']
        pred_rot = pred_rot / (torch.norm(pred_rot, dim=1, keepdim=True) + 1e-8)
        
        target_rot = targets['rotation']
        target_rot = target_rot / (torch.norm(target_rot, dim=1, keepdim=True) + 1e-8)
        
        # Rotation loss (geodesic distance)
        rot_loss = self.quaternion_distance(pred_rot, target_rot).mean()
        
        if torch.isnan(rot_loss) or torch.isinf(rot_loss):
            print("⚠️  NaN/Inf in rotation loss - skipping batch")
            return None
        
        # Clamp to prevent explosion
        rot_loss = torch.clamp(rot_loss, 0, 10.0)
        
        # Translation loss (L2 distance)
        trans_loss = torch.nn.functional.mse_loss(
            predictions['translation'], 
            targets['translation']
        )
        
        if torch.isnan(trans_loss) or torch.isinf(trans_loss):
            print("⚠️  NaN/Inf in translation loss - skipping batch")
            return None
        
        # Clamp translation loss
        trans_loss = torch.clamp(trans_loss, 0, 100.0)
        
        # Classification loss (if building_id is provided)
        cls_loss = 0
        if 'building_id' in targets and 'building_logits' in predictions:
            cls_loss = self.ce_loss(
                predictions['building_logits'], 
                targets['building_id']
            )
            if torch.isnan(cls_loss) or torch.isinf(cls_loss):
                print("⚠️  NaN/Inf in classification loss - skipping batch")
                return None
        
        # Combined loss
        total_loss = (
            self.rotation_weight * rot_loss +
            self.translation_weight * trans_loss +
            self.classification_weight * cls_loss
        )
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("⚠️  NaN/Inf in total loss - skipping batch")
            return None
        
        return {
            'total_loss': total_loss,
            'rotation_loss': rot_loss,
            'translation_loss': trans_loss,
            'classification_loss': cls_loss
        }



def quaternion_to_rotation_matrix(quaternion):
    """
    Convert quaternion to 3x3 rotation matrix.
    This is a wrapper for the shared utility function for backward compatibility.

    Args:
        quaternion: torch.Tensor of shape (4,) or (N, 4) [w, x, y, z]

    Returns:
        Rotation matrix of shape (3, 3) or (N, 3, 3)
    """
    return quat_to_rot_matrix(quaternion)
