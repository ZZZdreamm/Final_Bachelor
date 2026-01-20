import torch
import random
from classificator_training.data.load_data import get_dataloders
from classificator_training.model.loss import ProxyAnchorLoss
from classificator_training.model.model import FusedFeatureModel
from classificator_training.model.feature_extractor import FeatureExtractor
from classificator_training.test_helpers import evaluate_ncm_accuracy_top3
from classificator_training.train_helpers import train_or_load_model
from classificator_training.helpers.args import parse_and_get_args

SEED_VALUE = 42

torch.manual_seed(SEED_VALUE) 
random.seed(SEED_VALUE)

FULL_TRAIN = True

RENDERS_FOLDER_NAME = "AR_Renders7"
TEST_FOLDER_NAME = "test_data_przetworzone2"
RENDERS_FOLDER = f"/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/{RENDERS_FOLDER_NAME}"
TEST_FOLDER = f"/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/{TEST_FOLDER_NAME}"
VALIDATION_FOLDER = "/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/validation_data"

BATCH_PRINT_INTERVAL = 50 
NUM_EPOCHS = 100
VALIDATION_CHECK_INTERVAL = 50
VALIDATION_PATIENCE = 50
VALIDATION_MIN_DELTA = 0.001
sample_size = 8192

FILTER_HARD_DATA = False

LOAD_NETWORK = False 
LOAD_SPECIFIC_MODEL_NAME = None

args = parse_and_get_args(LOAD_SPECIFIC_MODEL_NAME)

BATCH_SIZE = args.batch
TRAIN_TYPE = args.train_type.lower()

MODELS_USED = {
    'clip': args.clip,
    'segformer': args.segformer,
    'midas': args.midas,
    'dpt': args.dpt,
    'resnet': args.resnet,
    'mobilenet': args.mobilenet,
    'efficientnet': args.efficientnet,
    'vit': args.vit
}

VECTORS_SAVE_FOLDER = f"classificator_training/preprocessed_vectors/mega_clip_segformer_midas_dpt5.pth"
TEST_VECTORS_SAVE_FOLDER = f"classificator_training/preprocessed_vectors/test_mega_clip_segformer_midas_dpt7.pth"
feature_extractor = FeatureExtractor(use_models=MODELS_USED)

MODELS_USED_FOR_TRAINING = MODELS_USED.copy()
MODELS_USED_FOR_TRAINING['gate'] = args.gate

train_dataloader, validation_dataloader, test_dataloader, id_to_tag = get_dataloders(RENDERS_FOLDER, TEST_FOLDER, preprocessed_file_path=VECTORS_SAVE_FOLDER, test_preprocessed_file_path=TEST_VECTORS_SAVE_FOLDER, validation_root_dir=None, is_full_train=FULL_TRAIN, batch_size=BATCH_SIZE, sample_size=sample_size, processor_flags=MODELS_USED, filter_hard_data=FILTER_HARD_DATA)
dataset_size = len(train_dataloader.dataset)

MODEL_NAME = (
    "fused_feature_model.pth"
    + f"_clip{args.clip}"
    + f"_segformer{args.segformer}"
    + f"_midas{args.midas}"
    + f"_dpt{args.dpt}"
    + f"_resnet{args.resnet}"           
    + f"_mobilenet{args.mobilenet}"     
    + f"_efficientnet{args.efficientnet}"
    + f"_vit{args.vit}"                 
    + f"_gate{args.gate}"
    + f"_bigfusionhead{args.big_fusion_head}"
    + f"_lr{args.lr}"
    + f"_margin{args.margin}"
    + f"_alpha{args.alpha}"
    + f"_datasetsize{dataset_size}"
    + ".model"
)

EMBEDDING_DIM = 1024 if args.big_fusion_head >= 2 else 512
NUM_BUILDINGS = len(id_to_tag)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = FusedFeatureModel(feature_dims=feature_extractor.feature_dims, embedding_dim=EMBEDDING_DIM, use_gate=bool(args.gate), big_fusion_head=args.big_fusion_head, use_models=MODELS_USED_FOR_TRAINING).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = ProxyAnchorLoss(
    num_classes=NUM_BUILDINGS, 
    embedding_dim=EMBEDDING_DIM, 
    margin=args.margin, 
    alpha=args.alpha 
).to(device)

model, prototype_tensor, class_ids, total_batches_processed, min_validation_loss = train_or_load_model(
    model, 
    load_network=LOAD_NETWORK, 
    specific_model_name=LOAD_SPECIFIC_MODEL_NAME,
    optimizer=optimizer, 
    criterion=criterion, 
    train_dataloader=train_dataloader, 
    device=device, 
    num_epochs=NUM_EPOCHS, 
    batch_print_interval=BATCH_PRINT_INTERVAL, 
    val_dataloader=validation_dataloader,
    validation_interval=VALIDATION_CHECK_INTERVAL,
    model_name=MODEL_NAME,
    patience=VALIDATION_PATIENCE,
    min_delta=VALIDATION_MIN_DELTA,
    train_type=TRAIN_TYPE,
)

print(f"\nMin validation loss loaded: {min_validation_loss}")

final_accuracy, accuracy_top3 = evaluate_ncm_accuracy_top3(model, test_dataloader, prototype_tensor, class_ids, id_to_tag, device)
print(f"\nFinal Test Accuracy - Top-1: {final_accuracy:.4f}, Top-3: {accuracy_top3:.4f}")
