import numpy as np
import torch
from classificator_training.test_helpers import generate_class_prototypes
from classificator_training.utils import move_to_device
import time
import os
import glob
import re

MODELS_SAVE_DIR = "classificator_training/saved_models/" 

def train_or_load_model(model, load_network=True, retrain_model=False, specific_model_name=None, optimizer=None, criterion=None, train_dataloader=None, device=None, num_epochs=1, batch_print_interval=5, val_dataloader=None, validation_interval=20, model_name="fused_feature_model.pth", patience=5, min_delta=0.05, train_type="hardmining"):
    
    save_dir = os.path.dirname(model_name) if os.path.dirname(model_name) else MODELS_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    latest_path = None
    
    if specific_model_name:
        latest_path = os.path.join(save_dir, specific_model_name)
        if not os.path.exists(latest_path):
            latest_path = None 
    
    if latest_path is None:
        base_filename = os.path.basename(model_name)
        max_num, latest_path = get_latest_checkpoint_info(save_dir, base_filename)

    if load_network and latest_path:
        try:
            checkpoint = torch.load(latest_path, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            total_batches_processed = checkpoint.get('total_batches_processed', None)
            min_validation_loss = checkpoint.get('min_validation_loss', None)
            
            if not retrain_model:
                model.eval()
                
                if 'prototype_tensor' in checkpoint and 'class_ids' in checkpoint:
                    prototype_tensor = checkpoint['prototype_tensor']
                    class_ids = checkpoint['class_ids']

                    return model, prototype_tensor, class_ids, total_batches_processed, min_validation_loss
                else:
                    prototype_tensor, class_ids = generate_class_prototypes(model, train_dataloader, device)

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'prototype_tensor': prototype_tensor,
                        'class_ids': class_ids,   
                        'total_batches_processed': total_batches_processed,
                        "id_to_tag": train_dataloader.dataset.id_to_tag,
                    }, latest_path)

                    return model, prototype_tensor, class_ids, total_batches_processed, min_validation_loss

        except Exception as e:
            pass
    if train_type == "hardmining":
        total_batches_processed, min_validation_loss = train_model_hard_mining(
            model, 
            optimizer, 
            criterion, 
            train_dataloader, 
            device, 
            num_epochs=num_epochs, 
            batch_print_interval=batch_print_interval,
            val_dataloader=val_dataloader,
            validation_interval=validation_interval,
            patience=patience,
            min_delta=min_delta
        )
        
    next_num = max_num + 1
    new_filename = f"{next_num}_{base_filename}"
    save_path = os.path.join(save_dir, new_filename)

    torch.save({
        'model_state_dict': model.state_dict(),
        "total_batches_processed": total_batches_processed,
        "min_validation_loss": min_validation_loss
    }, save_path)
    
    prototype_tensor, class_ids = generate_class_prototypes(model, train_dataloader, device)

    torch.save({
        'model_state_dict': model.state_dict(),
        'prototype_tensor': prototype_tensor,
        'class_ids': class_ids,             
        'total_batches_processed': total_batches_processed,
        'min_validation_loss': min_validation_loss,
        "id_to_tag": train_dataloader.dataset.id_to_tag,
    }, save_path)

    return model, prototype_tensor, class_ids, total_batches_processed, min_validation_loss

def get_latest_checkpoint_info(save_dir, base_filename):
    search_pattern = os.path.join(save_dir, f"*_{base_filename}")
    existing_files = glob.glob(search_pattern)
    
    max_num = 0
    latest_path = None
    
    pattern = re.compile(r'^(\d+)_') 
    
    for file in existing_files:
        name = os.path.basename(file)
        match = pattern.match(name)
        
        if match:
            current_num = int(match.group(1))
            
            if current_num > max_num:
                max_num = current_num
                latest_path = file
                
    return max_num, latest_path

class EarlyStopper:
    """
    Early stopping to stop training when the validation loss does not improve 
    after a given patience.
    """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta 
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_model_state = None

    def early_stop(self, validation_loss, model):
        """
        Returns True if early stopping criteria are met. 
        Stores the best model state if the current loss is an improvement.
        """
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model_state = model.state_dict() 
        elif validation_loss > self.min_validation_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    
    

def validate_model_mining(model, criterion, val_dataloader, device):
    """
    Calculates validation loss using Online Hard Mining (BatchHard).
    
    Args:
        model: The FusedFeatureModel.
        criterion: The loss criterion (used primarily to extract the margin).
        val_dataloader: DataLoader using the MultiModalDataset.
        device: 'cuda' or 'cpu'.
        
    Returns:
        (float, int): Average validation loss and number of batches checked.
    """
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    

    with torch.no_grad():
        for batch in val_dataloader:
            batch = move_to_device(batch, device) 
            
            inputs = batch['anchor']
            labels = batch['y']

            embeddings = model(**inputs) 
            
            loss = criterion(embeddings, labels)
            
            total_val_loss += loss.item()
            num_val_batches += 1
            
    if num_val_batches == 0:
        return 0.0, 0
        
    avg_val_loss = total_val_loss / num_val_batches
    return avg_val_loss, num_val_batches

def train_model_hard_mining(model, optimizer, criterion, train_dataloader, device, num_epochs=1, batch_print_interval=5, val_dataloader=None, validation_interval=20, patience=5, min_delta=0.05):
    
    
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    
    running_loss = 0.0
    running_time = 0.0
    total_batches_processed = 0
    
    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        for i, batch in enumerate(train_dataloader):
            batch_idx = i + 1
            total_batches_processed += 1
            batch = move_to_device(batch, device)
            
            inputs = batch['anchor']
            labels = batch['y']
            
            
            
            
            embeddings = model(**inputs)
            
            
            loss = criterion(embeddings, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if batch_idx % batch_print_interval == 0:
                avg_loss = running_loss / batch_print_interval
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {avg_loss:.4f}")

                running_loss = 0.0
                running_time += elapsed_time
                start_time = time.time()

            if val_dataloader is not None and batch_idx % validation_interval == 0:
                val_loss, num_val_batches = validate_model_mining(
                    model, criterion, val_dataloader, device
                )

                if early_stopper.early_stop(val_loss, model):
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best validation loss: {early_stopper.min_validation_loss:.4f}")
                    if early_stopper.best_model_state is not None:
                        model.load_state_dict(early_stopper.best_model_state)
                    return total_batches_processed, early_stopper.min_validation_loss

                model.train()
                start_time = time.time()

            if batch_idx % validation_interval == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

    print("\nTraining finished!")
    if early_stopper.best_model_state is not None:
        model.load_state_dict(early_stopper.best_model_state)

    return total_batches_processed, early_stopper.min_validation_loss
