import json
import cv2
import torch
import torch.nn as nn

from classificator_training.data.dataset import preprocess_single_image
from classificator_training.model.feature_extractor import FeatureExtractor
from classificator_training.utils import move_to_device


class BuildingPoseNet(nn.Module):
    """
    Neural network for estimating 6DoF pose from pre-extracted CLIP features.

    This is much faster than the full model since CLIP feature extraction
    is done offline once, and training only updates the pose heads.
    """

    def __init__(self, num_buildings=10, feature_dim=1664):
        super(BuildingPoseNet, self).__init__()

        self.feature_dim = feature_dim

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
        """
        Fast inference without classification
        """
        rotation_quat = self.rotation_head(features)
        translation = self.translation_head(features)
        confidence = self.confidence_head(features)

        rotation_quat = rotation_quat / (torch.norm(rotation_quat, dim=1, keepdim=True) + 1e-8)

        return rotation_quat, translation, confidence


class PoseLoss(nn.Module):
    """
    Combined loss for pose estimation training
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
        q1 = q1 / (torch.norm(q1, dim=1, keepdim=True) + 1e-8)
        q2 = q2 / (torch.norm(q2, dim=1, keepdim=True) + 1e-8)

        dot_product = torch.abs(torch.sum(q1 * q2, dim=1))
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        return 2 * torch.acos(dot_product)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with 'rotation', 'translation', 'building_logits'
            targets: dict with 'rotation', 'translation', 'building_id'
        """
        pred_rot = predictions['rotation']
        pred_rot = pred_rot / (torch.norm(pred_rot, dim=1, keepdim=True) + 1e-8)

        target_rot = targets['rotation']
        target_rot = target_rot / (torch.norm(target_rot, dim=1, keepdim=True) + 1e-8)

        rot_loss = self.quaternion_distance(pred_rot, target_rot).mean()

        rot_loss = torch.clamp(rot_loss, 0, 10.0)

        if torch.isnan(rot_loss):
            print("NaN detected in rotation loss! Using fallback.")
            rot_loss = torch.tensor(1.0, device=rot_loss.device)

        trans_loss = torch.nn.functional.mse_loss(
            predictions['translation'],
            targets['translation']
        )

        trans_loss = torch.clamp(trans_loss, 0, 100.0)

        if torch.isnan(trans_loss):
            print("NaN detected in translation loss! Using fallback.")
            trans_loss = torch.tensor(1.0, device=trans_loss.device)

        cls_loss = 0
        if 'building_id' in targets and 'building_logits' in predictions:
            cls_loss = self.ce_loss(
                predictions['building_logits'],
                targets['building_id']
            )
            if torch.isnan(cls_loss):
                print("NaN detected in classification loss! Using fallback.")
                cls_loss = torch.tensor(0.1, device=predictions['building_logits'].device)

        total_loss = (
            self.rotation_weight * rot_loss +
            self.translation_weight * trans_loss +
            self.classification_weight * cls_loss
        )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("NaN/Inf in total loss! Emergency fallback.")
            total_loss = torch.tensor(1.0, device=total_loss.device, requires_grad=True)

        return {
            'total_loss': total_loss,
            'rotation_loss': rot_loss,
            'translation_loss': trans_loss,
            'classification_loss': cls_loss
        }


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert quaternion to 3x3 rotation matrix

    Args:
        quaternion: torch.Tensor of shape (4,) or (N, 4) [w, x, y, z]

    Returns:
        Rotation matrix of shape (3, 3) or (N, 3, 3)
    """
    if quaternion.dim() == 1:
        quaternion = quaternion.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)

    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    R = torch.zeros((quaternion.shape[0], 3, 3), device=quaternion.device)

    R[:, 0, 0] = 1 - 2*y**2 - 2*z**2
    R[:, 0, 1] = 2*x*y - 2*w*z
    R[:, 0, 2] = 2*x*z + 2*w*y

    R[:, 1, 0] = 2*x*y + 2*w*z
    R[:, 1, 1] = 1 - 2*x**2 - 2*z**2
    R[:, 1, 2] = 2*y*z - 2*w*x

    R[:, 2, 0] = 2*x*z - 2*w*y
    R[:, 2, 1] = 2*y*z + 2*w*x
    R[:, 2, 2] = 1 - 2*x**2 - 2*y**2

    if squeeze:
        R = R.squeeze(0)

    return R



class BuildingPoseEstimator:
    """
    Process folders of building images and generate pose predictions
    Works with models trained on cached CLIP features
    """

    def __init__(self, model_path, config_path, feature_extractor: FeatureExtractor, device='cuda'):
        self.device = device

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.feature_extractor = feature_extractor
        feature_dim = feature_extractor.feature_dims['clip']

        self.model = BuildingPoseNet(
            num_buildings=self.config['num_buildings'],
            feature_dim=feature_dim
        )

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

    def predict_pose(self, image):
        """
        Predict pose from single image

        Args:
            image: numpy array (H, W, 3) - raw image

        Returns:
            dict with pose predictions
        """
        inputs = preprocess_single_image(image, self.feature_extractor.use_models)
        inputs = move_to_device(inputs, "cuda" if torch.cuda.is_available() else "cpu")
        features = self.feature_extractor.extract_feature_from_model("clip", inputs["clip_input"])

        with torch.no_grad():
            output = self.model(features)

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
        