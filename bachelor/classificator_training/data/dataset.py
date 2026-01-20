import re
from typing import Dict, List, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import os
import pandas as pd
import random
from torchvision import transforms
from classificator_training.model.feature_extractor import MODEL_PROCESSORS

def load_data_index_from_directory(root_dir: str, should_filter = False) -> pd.DataFrame:
    """
    Indexes all image files (png, jpg, jpeg) within a directory and its subdirectories 
    into a pandas DataFrame. The subdirectory name is used as the 'tag' and 'y' value.

    Args:
        root_dir (str): The root directory to start indexing from.
        should_filter (bool): If True, strictly matches filenames with pattern '{something}_{something}'.

    Returns:
        pd.DataFrame: A DataFrame containing 'img_path', 'tags', and 'y'.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    indexed_data: List[Dict[str, Any]] = []

    target_pattern = re.compile(r'^[^_]+_[^_]+$')
    
    valid_extensions = ('.png', '.jpg', '.jpeg')

    for dirpath, dirnames, filenames in os.walk(root_dir):
        tag: str = os.path.basename(dirpath)

        if dirpath == root_dir or not tag:
            continue

        for filename in filenames:
            if not filename.lower().endswith(valid_extensions):
                continue

            if should_filter and not target_pattern.match(filename):
                continue

            full_img_path: str = os.path.join(dirpath, filename)
            relative_img_path: str = os.path.relpath(full_img_path, root_dir)

            indexed_data.append({
                'img_path': relative_img_path,
                'tags': set([tag]),
                'y': tag
            })

    return pd.DataFrame(indexed_data)

class TripletMultiModalDataset(Dataset):
    """
    Returns a single image and its class label, ready for batch processing.
    Supports Hugging Face processors AND standard Torchvision transforms.
    """
    def __init__(self, data_df, processors, root_dir, processor_flags=None, hard_mining_mode=False, preprocessed_file_path=None, feature_extraction_mode=False, id_to_tag=None):
        self.data_df = data_df
        self.root_dir = root_dir
        self.paths = data_df['img_path'].tolist()
        self.labels = data_df['y'].tolist() 
        self.processors = processors
        self.hard_mining_mode = hard_mining_mode
        self.preprocessed_file_path = preprocessed_file_path
        self.feature_extraction_mode = feature_extraction_mode
        self.id_to_tag = id_to_tag
        
        if processor_flags is None:
            self.processor_flags = {k: True for k in processors.keys()}
        else:
            self.processor_flags = processor_flags
            
        self.standard_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.preprocessed_data = None
        if self.preprocessed_file_path and os.path.exists(self.preprocessed_file_path):
            self.preprocessed_data = torch.load(self.preprocessed_file_path)
            self.id_to_tag = self.preprocessed_data.get('id_to_tag', None)

            valid_keys = [k for k in self.preprocessed_data.keys() if k not in ['y', 'id_to_tag']]
            if valid_keys:
                first_key = valid_keys[0]
                if len(self.preprocessed_data[first_key]) != len(data_df):
                    raise ValueError(f"Loaded feature length ({len(self.preprocessed_data[first_key])}) != dataframe length ({len(data_df)}).")

    def __len__(self):
        return len(self.data_df)

    def _get_similar_dissimilar_indices(self, anchor_tags, anchor_idx):
        positive_idx = self._get_positive_index(anchor_idx, anchor_tags)
        negative_idx = self._get_negative_index(anchor_idx, anchor_tags)
        return positive_idx, negative_idx

    def _get_positive_index(self, anchor_idx, anchor_tags): 
        possible_positives = self.data_df[
            (self.data_df['tags'].apply(lambda t: t == anchor_tags)) & 
            (self.data_df.index != anchor_idx)
        ].index.tolist()

        if not possible_positives:
            possible_positives = self.data_df[
                (self.data_df['tags'].apply(lambda t: not anchor_tags.isdisjoint(t))) & 
                (self.data_df.index != anchor_idx)
            ].index.tolist()
            
            if not possible_positives:
                return anchor_idx 
            
        return random.choice(possible_positives)

    def _get_negative_index(self, anchor_idx, anchor_tags):
        possible_negatives = self.data_df[
            self.data_df['tags'].apply(lambda t: anchor_tags.isdisjoint(t))
        ].index.tolist()

        if not possible_negatives:
            possible_negatives = self.data_df[
                self.data_df.index != anchor_idx
            ].index.tolist()
            
        return random.choice(possible_negatives)

    def _preprocess_image(self, image_path):
        """Loads one image and applies transforms for ALL flagged models."""
        
        expected_input_shapes = {
            'clip': (3, 224, 224),     
            'segformer': (3, 512, 512),
            'midas': (3, 384, 384),    
            'dpt': (3, 384, 384),
            'resnet': (3, 224, 224),
            'mobilenet': (3, 224, 224),
            'efficientnet': (3, 224, 224),
            'vit': (3, 224, 224)
        }
        
        inputs = {}
        
        try:
            full_path = os.path.join(self.root_dir, image_path)
            image = Image.open(full_path).convert("RGB")
        
            for modality, is_enabled in self.processor_flags.items():
                if not is_enabled:
                    continue

                if modality in self.processors:
                    processor = self.processors[modality]
                    out = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
                    inputs[modality] = out
                
                elif modality in ['resnet', 'mobilenet', 'efficientnet', 'vit']:
                    out = self.standard_transform(image)
                    inputs[modality] = out
                
            return inputs

        except Exception as e:
            for modality, is_enabled in self.processor_flags.items():
                if is_enabled and modality not in inputs:
                    shape = expected_input_shapes.get(modality, (3, 224, 224))
                    dummy_tensor = torch.full(shape, -1.0, dtype=torch.float32)
                    inputs[modality] = dummy_tensor
            
            return inputs

    def __getitem__(self, idx):
        if self.preprocessed_data:
            anchor_inputs = {}
            for modality in self.preprocessed_data.keys():
                if modality not in ['y', 'id_to_tag'] and self.processor_flags.get(modality, False):
                    anchor_inputs[modality] = self.preprocessed_data[modality][idx]
            
            A_label = self.preprocessed_data['y'][idx]
            return {
                'anchor': anchor_inputs,
                'y': A_label 
            }
        
        A_row = self.data_df.iloc[idx]
        A_path = A_row['img_path']
        A_tags = A_row['tags']
        A_label = A_row['y']
            
        if self.feature_extraction_mode:
            inputs = self._preprocess_image(A_path)
            return {
                **{f'{k}_input': v for k, v in inputs.items()},
                'y': torch.tensor(A_label, dtype=torch.long)
            }

        if self.hard_mining_mode:
            inputs = self._preprocess_image(A_path)
            return {
                'anchor': inputs,
                'y': torch.tensor(A_label, dtype=torch.long)
            }

        else:
            P_idx, N_idx = self._get_similar_dissimilar_indices(A_tags, idx)
            
            P_path = self.data_df.iloc[P_idx]['img_path']
            N_path = self.data_df.iloc[N_idx]['img_path']

            A_inputs = self._preprocess_image(A_path)
            P_inputs = self._preprocess_image(P_path)
            N_inputs = self._preprocess_image(N_path)
            
            return {
                'anchor': A_inputs,
                'positive': P_inputs,
                'negative': N_inputs,
                'y': torch.tensor(A_label, dtype=torch.long)
            }
        

def preprocess_single_image(image: Image, processor_flags: Dict[str, bool]):
    inputs = {}
    for modality, use_processor in processor_flags.items():
        if use_processor and modality in MODEL_PROCESSORS:
            processor = MODEL_PROCESSORS[modality]
            input_tensor = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            inputs[f"{modality}_input"] = input_tensor
    
    return inputs
        