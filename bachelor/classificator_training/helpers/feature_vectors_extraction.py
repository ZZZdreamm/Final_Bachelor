import torch
import random
from classificator_training.data.load_data import get_dataloders, save_preprocessed_data_to_output
from classificator_training.model.loss import ProxyAnchorLoss
from classificator_training.model.model import FusedFeatureModel
from classificator_training.model.feature_extractor import FeatureExtractor
from classificator_training.test_helpers import evaluate_ncm_accuracy_top3
from classificator_training.train_helpers import train_or_load_model
from classificator_training.helpers.args import parse_and_get_args

SEED_VALUE = 42

torch.manual_seed(SEED_VALUE) 
random.seed(SEED_VALUE)

RENDERS_FOLDER_NAME = "BlenderRenders_do_sieci"
TEST_FOLDER_NAME = "test_data_przetworzone2"
RENDERS_FOLDER = f"/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/{RENDERS_FOLDER_NAME}"
TEST_FOLDER = f"/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/{TEST_FOLDER_NAME}"

MODELS_USED = {
    'clip': 1,
    'segformer': 1,
    'midas': 1,
    'dpt': 1,
    'resnet': 1,
    'mobilenet': 1,
    'efficientnet': 1,
    'vit': 1
}

VECTORS_SAVE_FOLDER = f"classificator_training/preprocessed_vectors/mega_clip_segformer_midas_dpt9.pth"
TEST_VECTORS_SAVE_FOLDER = f"classificator_training/preprocessed_vectors/test_mega_clip_segformer_midas_dpt9.pth"
feature_extractor = FeatureExtractor(use_models=MODELS_USED)

save_preprocessed_data_to_output(root_folder=RENDERS_FOLDER, root_test_folder=TEST_FOLDER, output_file_path=VECTORS_SAVE_FOLDER, test_output_file_path=TEST_VECTORS_SAVE_FOLDER, feature_extractor=feature_extractor, processor_flags=MODELS_USED)