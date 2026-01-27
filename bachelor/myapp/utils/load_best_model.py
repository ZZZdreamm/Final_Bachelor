from argparse import Namespace
import os
import torch
from classificator_training.model.model import FusedFeatureModel
from classificator_training.model.feature_extractor import FeatureExtractor
from classificator_training.helpers.args import _override_args_from_model_name
from myapp.AR.pose_network import BuildingPoseEstimator

ENVIRONEMENT = os.getenv("ENVIRONMENT", "production")
MODEL_PATH = "1_fused_feature_model.pth_full_clip1_segformer0_midas0_dpt0_gate0_batch64_traintypehardmining_bigfusionhead2_lr2e-07_margin1.2_alpha64.0.model"
FULL_MODEL_PATH = f"trained_model/" if ENVIRONEMENT == "development" else f"myapp/trained_model/"
FULL_CLASSIFICATION_MODEL_PATH = FULL_MODEL_PATH + MODEL_PATH
FULL_ESTIMATION_MODEL_PATH = FULL_MODEL_PATH + "estimation_pose_model.pth"
FULL_ESTIMATION_DATASET_PATH = FULL_MODEL_PATH + "estimation_pose_dataset.json"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_best_model():
    args = Namespace(clip=0, segformer=0, midas=0, dpt=0, gate=0, big_fusion_head=0, train_type='', lr=0.0, margin=0.0, alpha=0.0)
    args = _override_args_from_model_name(args, MODEL_PATH)
    print("Loading model with args:", args)
    MODELS_USED = {
        'clip': args.clip,
        'segformer': args.segformer,
        'midas': args.midas,
        'dpt': args.dpt,
    }

    feature_extractor = FeatureExtractor(use_models=MODELS_USED)
    MODELS_USED_FOR_TRAINING = MODELS_USED.copy()
    MODELS_USED_FOR_TRAINING['gate'] = args.gate

    EMBEDDING_DIM = 1024 if args.big_fusion_head >= 2 else 512
    model = FusedFeatureModel(feature_dims=feature_extractor.feature_dims, embedding_dim=EMBEDDING_DIM, use_gate=bool(args.gate), big_fusion_head=args.big_fusion_head, use_models=MODELS_USED_FOR_TRAINING).to(device)

    checkpoint = torch.load(FULL_CLASSIFICATION_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    prototype_tensor = checkpoint.get('prototype_tensor', None)
    class_ids = checkpoint.get('class_ids', None)
    id_to_tag = checkpoint.get('id_to_tag', None)
    tag_to_id = {tag: id for id, tag in id_to_tag.items()}
    
    pose_model = BuildingPoseEstimator(
        model_path=FULL_ESTIMATION_MODEL_PATH,
        config_path=FULL_ESTIMATION_DATASET_PATH,
        feature_extractor=feature_extractor,
        device=device,
    )

    return model, prototype_tensor, class_ids, feature_extractor, id_to_tag, tag_to_id, pose_model
    
    