from typing import List, Dict
import io

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from myapp.database.metadata.config import DatabaseQueries, database_client
from myapp.AR.model_template import TRAINED_CLASSIFICATION_MODEL
from myapp.models.Building import BuildingSearch
from myapp.utils.image_utils import get_image_stream
from classificator_training.data.dataset import preprocess_single_image
from classificator_training.utils import move_to_device

device = "cuda" if torch.cuda.is_available() else "cpu"
METERS_IN_KM = 1000

def get_buildings_in_proximity(latitude: float, longitude: float, distance_meters=METERS_IN_KM * 3) -> list[dict]:
    response = database_client.rpc(
        DatabaseQueries.GET_NEARBY_PLACES.value,
        {"latitude": latitude, "longitude": longitude, "distance_meters": distance_meters}
    ).execute()
    
    return response.data

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

POSSIBLE_CLASSES = [
    "a photo of a building", 
    "a photo of a house",
    "a photo of a skyscraper",
]

NEGATIVE_CLASSES = [
    "a photo of a car", 
    "a photo of a solar dish", 
    "a photo of a satellite dish", 
    "a close up of an object",
    "a forest",
    "a street"
]

ALL_LABELS = POSSIBLE_CLASSES + NEGATIVE_CLASSES

def is_building_on_image(img_bytes: bytes) -> bool:
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        inputs = processor(
            text=ALL_LABELS,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)[0]

        top_prob, top_idx = probs.topk(1)
        predicted_label = ALL_LABELS[top_idx.item()]
        print(f"Predicted label: {predicted_label} with probability {top_prob.item()}")

        if predicted_label in POSSIBLE_CLASSES and top_prob.item() > 0.5:
            return True

        return False
    except Exception:
        return False

def get_image_predicted_classes(image_bytes: bytes, models_in_proximity: List[BuildingSearch]):
    image = get_image_stream(image_bytes)
    feature_extractor = TRAINED_CLASSIFICATION_MODEL["feature_extractor"]
    inputs = preprocess_single_image(image, feature_extractor.use_models)
    inputs = move_to_device(inputs, device)
    extracted_features = feature_extractor.extract_all_features(**inputs)
    return _get_model_prediction(extracted_features, models_in_proximity)

def _get_model_prediction(inputs, models_in_proximity: List[BuildingSearch]) -> List[BuildingSearch]:
    model = TRAINED_CLASSIFICATION_MODEL["model"]
    prototypes = TRAINED_CLASSIFICATION_MODEL["prototype_tensor"]
    class_ids = TRAINED_CLASSIFICATION_MODEL["class_ids"]
    models_class_ids = set([model.class_id for model in models_in_proximity])

    classes_in_proximity = [model.class_id for model in models_in_proximity]
    prototypes_in_proximity = _filter_prototypes(prototypes, class_ids, classes_in_proximity)

    test_embeddings = model(**inputs)
    similarity_matrix = test_embeddings @ prototypes_in_proximity.T

    number_of_prototypes = prototypes_in_proximity.shape[0]
    K = 5 if number_of_prototypes > 5 else number_of_prototypes
    topK_results = torch.topk(similarity_matrix, K, dim=1)
    
    predicted_class_indices_topK = topK_results.indices

    topk_predicted_labels = torch.tensor([[classes_in_proximity[idx.item()] for idx in row]
                                            for row in predicted_class_indices_topK], device=device)
    predicted_class_ids = topk_predicted_labels.cpu().numpy().tolist()[0]

    matching_models = []
    for class_id in predicted_class_ids:
        if class_id in models_class_ids:
            matching_models.append(next(model for model in models_in_proximity if model.class_id == class_id))

    return matching_models

def _filter_prototypes(
    prototype_tensor: torch.Tensor,
    all_class_ids: List[int],
    classes_to_compare_with: List[int]
) -> torch.Tensor:
    """
    Creates a new tensor containing only the prototypes for the specified class IDs.
    """

    class_id_to_index: Dict[int, int] = {
        class_id: index for index, class_id in enumerate(all_class_ids)
    }

    target_indices: List[int] = []

    for class_id in classes_to_compare_with:
        if class_id in class_id_to_index:
            target_indices.append(class_id_to_index[class_id])

    index_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)
    filtered_prototypes = torch.index_select(prototype_tensor, 0, index_tensor)

    return filtered_prototypes
