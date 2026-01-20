import os
import random
import time
import pandas as pd
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader, random_split
from classificator_training.data.dataset import TripletMultiModalDataset, load_data_index_from_directory, MODEL_PROCESSORS
from classificator_training.model.feature_extractor import FeatureExtractor

SEED_VALUE = 42
torch.manual_seed(SEED_VALUE) 
random.seed(SEED_VALUE)

def load_dataset_from_directory(root_dir: str, test_samples_count: int = 100, batch_size: int = 32, tag_to_id = None, has_tags = False, processor_flags=dict[str, bool], hard_mining_mode=False, filter_hard_data = False,preprocessed_file_path=None, feature_extraction = False) -> TripletMultiModalDataset:
    if not os.path.isdir(root_dir):
        return None, None, None
    
    data_df = load_data_index_from_directory(root_dir, should_filter=filter_hard_data)
    data_df.index = pd.RangeIndex(len(data_df.index)) 

    if tag_to_id:
        data_df['y'] = data_df['y'].apply(lambda tag: tag_to_id[tag])
        id_to_tag = {i: tag for tag, i in tag_to_id.items()}
    else:
        unique_tags = sorted(data_df['y'].unique())
        tag_to_id = {tag: i for i, tag in enumerate(unique_tags)}
        id_to_tag = {i: tag for tag, i in tag_to_id.items()}
        data_df['y'] = data_df['y'].apply(lambda tag: tag_to_id[tag])
        
    total_size = len(data_df)
    
    new_total_size = (total_size // batch_size) * batch_size
    
    samples_cut = total_size - new_total_size
    if samples_cut > 0 and not has_tags:
        data_df = data_df.iloc[:new_total_size]
    
    custom_dataset = TripletMultiModalDataset(
        data_df=data_df, 
        processors=MODEL_PROCESSORS, 
        root_dir=root_dir,
        processor_flags=processor_flags,
        hard_mining_mode=hard_mining_mode,
        feature_extraction_mode=feature_extraction,
        preprocessed_file_path=preprocessed_file_path,
        id_to_tag=id_to_tag
    )

    if has_tags:
        return custom_dataset, None, None, None
    
    total_size = len(custom_dataset)
    remaining_size = total_size - test_samples_count
    lengths = [remaining_size, test_samples_count]

    train_subset, test_subset = random_split(custom_dataset, lengths)
    
    return custom_dataset, train_subset, test_subset, tag_to_id, id_to_tag

def get_dataloders(root_dir: str, test_root_dir: str = None, validation_root_dir: str = None, is_full_train: bool = True, batch_size: int = 32, sample_size: int = 1024, processor_flags: dict[str, bool] = None, preprocessed_file_path=None, test_preprocessed_file_path=None, filter_hard_data = False):
    custom_dataset, train_subset, test_subset, tag_to_id, id_to_tag = load_dataset_from_directory(root_dir, processor_flags=processor_flags, preprocessed_file_path=preprocessed_file_path)
    validation_set, _, _, _ = load_dataset_from_directory(validation_root_dir if validation_root_dir else test_root_dir, tag_to_id=tag_to_id, has_tags=True, processor_flags=processor_flags, hard_mining_mode=True, filter_hard_data=filter_hard_data, preprocessed_file_path=test_preprocessed_file_path)
    test_set = validation_set
    
    validation_dataloader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    test_dataloader = DataLoader(
        test_set if test_set is not None else test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    
    if is_full_train:
        train_dataloader = DataLoader(
            custom_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
    else:
        all_indices = list(train_subset.indices)
        sampled_indices = random.sample(all_indices, sample_size)
        sample_subset = Subset(custom_dataset, sampled_indices)

        train_dataloader = DataLoader(
            sample_subset,
            batch_size=batch_size, 
            shuffle=False,        
            num_workers=4,        
            pin_memory=False
        )

    return train_dataloader, validation_dataloader, test_dataloader, id_to_tag

def save_preprocessed_data_to_output(root_folder: str, root_test_folder: str, output_file_path: str, test_output_file_path: str, feature_extractor: FeatureExtractor, processor_flags=dict[str, bool]):
    train_dataset, _, _, tag_to_id, _ = load_dataset_from_directory(root_folder, processor_flags=processor_flags, feature_extraction=True)
    test_dataset = load_dataset_from_directory(root_test_folder, processor_flags=processor_flags, feature_extraction=True, tag_to_id=tag_to_id, has_tags=True)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)
    feature_extractor.eval()

    _save_data_to_output(train_dataset, output_file_path, feature_extractor)
    _save_data_to_output(test_dataset, test_output_file_path, feature_extractor)
  
  
  
def _save_data_to_output(dataset, output_file_path: str, feature_extractor: FeatureExtractor):
    print(f"\nProcessing test data for feature extraction...")
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_features = {modality: [] for modality in feature_extractor.feature_dims.keys()}
    all_labels = []

    print(f"Starting batched feature extraction...")

    with torch.no_grad():
        start_time = time.time()
        for i, batch in enumerate(data_loader):
            model_inputs = {
                k: v.to(device) 
                for k, v in batch.items() 
                if k != 'y'
            }
            labels = batch['y']
            
            extracted_feature_dict = feature_extractor.extract_all_features(**model_inputs)
            
            for modality, features in extracted_feature_dict.items():
                all_features[modality].append(features.cpu())
                
            all_labels.append(labels)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {(i + 1) * data_loader.batch_size} / {len(data_loader.dataset)} samples. Elapsed time: {elapsed:.2f} seconds.")

    final_data = {}
    for modality, feature_list in all_features.items():
        if feature_list:
            final_data[modality] = torch.cat(feature_list, dim=0)

    final_data['y'] = torch.cat(all_labels, dim=0)
    final_data['id_to_tag'] = dataset.id_to_tag
    
    
    folder_path = os.path.dirname(output_file_path)

    if folder_path:
        os.makedirs(folder_path, exist_ok=True)

    torch.save(final_data, output_file_path)
    print(f"Successfully saved preprocessed features to {output_file_path}")