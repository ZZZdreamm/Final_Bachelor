import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb
import h5py
from pose_network import BuildingPoseNetCached, PoseLoss


class CachedFeaturesDataset(Dataset):
    """
    Dataset that loads pre-extracted CLIP features from HDF5 file
    """
    
    def __init__(self, h5_file, split='train'):
        self.h5_file = h5_file
        self.split = split
        
        with h5py.File(h5_file, 'r') as f:
            all_splits = f['splits'][:]
            self.indices = np.where(all_splits == split.encode('utf-8'))[0]
            
            self.features = f['features'][self.indices]
            self.rotations = f['rotations'][self.indices]
            self.translations = f['translations'][self.indices]
            self.building_ids = f['building_ids'][self.indices]
            self.image_ids = f['image_ids'][self.indices]
            
            self.feature_dim = f.attrs['feature_dim']
            
        print(f"Loaded {len(self.indices)} samples for {split} split")
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return {
            'features': torch.from_numpy(self.features[idx]).float(),
            'rotation': torch.from_numpy(self.rotations[idx]).float(),
            'translation': torch.from_numpy(self.translations[idx]).float(),
            'building_id': torch.tensor(self.building_ids[idx], dtype=torch.long),
            'image_id': self.image_ids[idx].decode('utf-8')
        }


class PoseEstimationTrainer:
    """
    Trainer for pose estimation network with cached features
    """
    
    def __init__(self, model, train_loader, val_loader, 
                 device='cuda', use_wandb=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb
        
        # Loss function
        self.criterion = PoseLoss(
            rotation_weight=1.0,
            translation_weight=0.1, 
            classification_weight=0.5
        )
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3, 
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        self.best_val_loss = float('inf')
        self.best_rotation_error = float('inf')
    
    def train_epoch(self, epoch):
        """Train for one epoch with NaN batch skipping"""
        self.model.train()
        total_loss = 0
        total_rot_loss = 0
        total_trans_loss = 0
        total_cls_loss = 0
        valid_batches = 0
        skipped_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            
            targets = {
                'rotation': batch['rotation'].to(self.device),
                'translation': batch['translation'].to(self.device),
                'building_id': batch['building_id'].to(self.device)
            }
            
            predictions = self.model(features)
            
            loss_dict = self.criterion(predictions, targets)
            
            if loss_dict is None:
                skipped_batches += 1
                pbar.set_postfix({
                    'status': '⚠️ NaN',
                    'skipped': skipped_batches
                })
                continue  
            
            loss = loss_dict['total_loss']
            
            self.optimizer.zero_grad()
            
            try:
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                valid_batches += 1
                total_loss += loss.item()
                total_rot_loss += loss_dict['rotation_loss'].item()
                total_trans_loss += loss_dict['translation_loss'].item()
                total_cls_loss += loss_dict['classification_loss'].item() if isinstance(loss_dict['classification_loss'], torch.Tensor) else 0
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'rot': f"{loss_dict['rotation_loss'].item():.4f}",
                    'trans': f"{loss_dict['translation_loss'].item():.4f}",
                    'skip': skipped_batches
                })
                
                if self.use_wandb:
                    wandb.log({
                        'train/batch_loss': loss.item(),
                        'train/rotation_loss': loss_dict['rotation_loss'].item(),
                        'train/translation_loss': loss_dict['translation_loss'].item(),
                    })
                    
            except RuntimeError as e:
                print(f"\n⚠️  RuntimeError: {e}")
                print("    Skipping batch...")
                skipped_batches += 1
                continue
        
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            avg_rot_loss = total_rot_loss / valid_batches
            avg_trans_loss = total_trans_loss / valid_batches
            avg_cls_loss = total_cls_loss / valid_batches
        else:
            print("⚠️  No valid batches!")
            avg_loss = avg_rot_loss = avg_trans_loss = avg_cls_loss = float('nan')
        
        if skipped_batches > 0:
            skip_pct = 100 * skipped_batches / len(self.train_loader)
            print(f"\n  ⚠️  Skipped {skipped_batches}/{len(self.train_loader)} batches ({skip_pct:.1f}%)")
            if skip_pct > 10:
                print(f"      WARNING: >10% skipped - check data for NaN/Inf!")
        
        return {
            'loss': avg_loss,
            'rotation_loss': avg_rot_loss,
            'translation_loss': avg_trans_loss,
            'classification_loss': avg_cls_loss,
            'valid_batches': valid_batches,
            'skipped_batches': skipped_batches
        }
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_rot_loss = 0
        total_trans_loss = 0
        rotation_errors = []
        translation_errors = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for batch in pbar:
                features = batch['features'].to(self.device)
                
                targets = {
                    'rotation': batch['rotation'].to(self.device),
                    'translation': batch['translation'].to(self.device),
                    'building_id': batch['building_id'].to(self.device)
                }
                
                # Forward pass
                predictions = self.model(features)
                
                # Compute loss
                loss_dict = self.criterion(predictions, targets)
                
                total_loss += loss_dict['total_loss'].item()
                total_rot_loss += loss_dict['rotation_loss'].item()
                total_trans_loss += loss_dict['translation_loss'].item()
                
                # Compute errors in degrees and meters
                for i in range(len(features)):
                    rot_error_rad = self.criterion.quaternion_distance(
                        predictions['rotation'][i:i+1],
                        targets['rotation'][i:i+1]
                    ).item()
                    rot_error_deg = rot_error_rad * 180 / np.pi
                    
                    trans_error = torch.norm(
                        predictions['translation'][i] - targets['translation'][i]
                    ).item()
                    
                    rotation_errors.append(rot_error_deg)
                    translation_errors.append(trans_error)
        
        # Average metrics
        n_batches = len(self.val_loader)
        avg_loss = total_loss / n_batches
        avg_rot_loss = total_rot_loss / n_batches
        avg_trans_loss = total_trans_loss / n_batches
        avg_rot_error = np.mean(rotation_errors)
        avg_trans_error = np.mean(translation_errors)
        median_rot_error = np.median(rotation_errors)
        median_trans_error = np.median(translation_errors)
        
        print(f"\nValidation Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Rotation Error: {avg_rot_error:.2f}° (median: {median_rot_error:.2f}°)")
        print(f"  Translation Error: {avg_trans_error:.2f}m (median: {median_trans_error:.2f}m)")
        
        return {
            'loss': avg_loss,
            'rotation_loss': avg_rot_loss,
            'translation_loss': avg_trans_loss,
            'rotation_error_deg': avg_rot_error,
            'translation_error_m': avg_trans_error,
            'median_rotation_error_deg': median_rot_error,
            'median_translation_error_m': median_trans_error
        }
    
    def save_checkpoint(self, epoch, metrics, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def train(self, num_epochs, checkpoint_dir='checkpoints'):
        """Full training loop"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            self.scheduler.step(val_metrics['loss'])
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'val/loss': val_metrics['loss'],
                    'val/rotation_error_deg': val_metrics['rotation_error_deg'],
                    'val/translation_error_m': val_metrics['translation_error_m'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_metrics['rotation_error_deg'] < self.best_rotation_error:
                self.best_rotation_error = val_metrics['rotation_error_deg']
                self.save_checkpoint(
                    epoch, 
                    val_metrics,
                    checkpoint_dir / 'best_model.pth'
                )
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(
                    epoch,
                    val_metrics,
                    checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
                )
        
        print("\nTraining completed!")
        print(f"Best rotation error: {self.best_rotation_error:.2f}°")


def main():
    """Main training script"""
    
    config = {
        'batch_size': 128,  
        'num_epochs': 100,
        'num_buildings': 3,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_wandb': False,  
        'features_file': '/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/AR_pose1/annotations/clip_features.h5'
    }
    
    print(f"\n{'='*70}")
    print(f"CACHED CLIP FEATURES TRAINING")
    print(f"{'='*70}")
    print(f"Features file: {config['features_file']}")
    print(f"Batch size: {config['batch_size']} (much larger than image-based training!)")
    print(f"Device: {config['device']}")
    print(f"{'='*70}\n")
    
    if config['device'] == 'cpu':
        print("⚠️  WARNING: Training on CPU will be slower!")
    
    if config['use_wandb']:
        wandb.init(project='building-pose-estimation-cached', config=config)
    
    train_dataset = CachedFeaturesDataset(
        config['features_file'],
        split='train'
    )
    
    val_dataset = CachedFeaturesDataset(
        config['features_file'],
        split='val'
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    feature_dim = train_dataset.feature_dim
    print(f"Feature dimension: {feature_dim}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print("\nInitializing pose estimation heads...")
    model = BuildingPoseNetCached(
        num_buildings=config['num_buildings'],
        feature_dim=feature_dim
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")
    print("(Compare to ~428M in full CLIP model!)")
    
    trainer = PoseEstimationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        use_wandb=config['use_wandb']
    )
    
    print("\n🚀 Starting training with cached features...")
    print("This should be MUCH faster than image-based training!\n")
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == "__main__":
    main()