# Classificator Training

Training pipeline for building classification using multi-modal metric learning with fused feature models.

## Overview

This module trains deep metric learning models that classify buildings from images by learning discriminative embeddings. The system combines multiple pre-trained vision models (CLIP, SegFormer, MiDaS, DPT, etc.) through a learned fusion head.

**Key Features:**
- Multi-modal feature fusion from 8+ vision models
- Metric learning with ProxyAnchor loss
- Hard mining training strategy
- Nearest Class Mean (NCM) classification
- Pre-computed feature caching for fast training
- Extensive hyperparameter configuration

## Quick Start

### Installation

```bash
pip install torch torchvision transformers pytorch-metric-learning pandas
```

### Example Training

```bash
python main.py \
    --clip 1 --segformer 0 --midas 0 --dpt 0 \
    --batch 64 \
    --train_type hardmining \
    --lr 0.00001 \
    --margin 0.2 \
    --alpha 32.0
```

## Architecture

### Multi-Modal Feature Extraction

The model supports 8 pre-trained feature extractors:

| Model | Input Size | Features | Purpose |
|-------|-----------|----------|---------|
| CLIP ViT-bigG-14 | 224×224 | 1280 | Vision-language understanding |
| SegFormer MIT-B5 | 512×512 | 768 | Semantic segmentation |
| DPT-Large | 384×384 | 768 | Dense depth prediction |
| MiDaS DPT-Hybrid | 384×384 | 768 | Depth estimation |
| ResNet-152 | 224×224 | 2048 | General visual features |
| MobileNet-V3-Large | 224×224 | 960 | Efficient features |
| EfficientNet-V2-L | 224×224 | 1280 | Efficient high-capacity |
| ViT-L/16 | 224×224 | 1024 | Transformer features |

All backbones are **frozen** during training (no gradient computation).

### Fusion Head Architecture

```
Concatenated Features (variable dimension)
    ↓
[Optional] Gating Head
    ├─ Linear(total_dim, hidden_dim)
    ├─ GELU → Linear(hidden_dim, hidden_dim//2)
    ├─ GELU → Linear(hidden_dim//2, total_dim)
    └─ Sigmoid → element-wise gating weights
    ↓
Fusion Head (big_fusion_head = 0/1/2/3)
    ├─ Variable depth (2-7 layers)
    ├─ Layer sizes: 1024 → 8192+ dimensions
    ├─ BatchNorm1d + ReLU/GELU + Dropout(0.3)
    └─ Final layer → embedding_dim (512 or 1024)
    ↓
L2 Normalization → Final Embedding
```

**Fusion Head Variants:**
- `big_fusion_head=0`: Minimal (1024→512→embedding)
- `big_fusion_head=1`: Medium (1024→3072→2048→embedding)
- `big_fusion_head=2`: Large (5 layers, 1024D output)
- `big_fusion_head=3`: Extra-large (7 layers, 1024D output)

## Training

### Data Preparation

Organize dataset in this structure:

```
dataset_root/
├── Building_A/
│   ├── image_001.png
│   ├── image_002.jpg
│   └── ...
├── Building_B/
│   ├── image_001.png
│   └── ...
└── Building_C/
    └── ...
```

### Training Modes

**Hard Mining**:
```python
python main.py --train_type hardmining
```

- Uses ProxyAnchor loss
- Automatically mines hard positive/negative examples
- Faster convergence


### Configuration

**Command-Line Arguments:**

```bash
python main.py \
    # Feature extractors (0=off, 1=on)
    --clip 1 \
    --segformer 0 \
    --midas 0 \
    --dpt 0 \
    --resnet 0 \
    --mobilenet 0 \
    --efficientnet 0 \
    --vit 0 \

    # Architecture
    --gate 0 \                    
    --big_fusion_head 2 \         

    # Training
    --batch 64 \                  
    --lr 0.00001 \                
    --train_type hardmining \     

    # ProxyAnchor Loss
    --margin 0.2 \                
    --alpha 32.0                  
```

**Key Hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch` | 64 | Batch size for training |
| `lr` | 1e-5 | Adam optimizer learning rate |
| `margin` | 0.2 | ProxyAnchor margin (intra-class threshold) |
| `alpha` | 32.0 | ProxyAnchor alpha (temperature scaling) |
| `big_fusion_head` | 0 | Fusion head architecture size |
| `gate` | 0 | Enable/disable gating mechanism |

### Pre-Computing Features

For faster training, features are pre-extracted and saved in file:

```python
from data.load_data import save_preprocessed_data_to_output

save_preprocessed_data_to_output(
    root_dir="path/to/dataset",
    output_file="preprocessed_features.pth",
    use_models={
        'clip': True,
        'segformer': False,
        'midas': False,
        'dpt': False
    }
)
```

Then load during training:
```python
train_loader = get_dataloders(
    preprocessed_file="preprocessed_features.pth",
    batch_size=64
)
```

### Training Loop

The training loop includes:
1. **Forward pass**: Extract/load features, pass through fusion head
2. **Loss computation**: ProxyAnchor loss on embeddings
3. **Validation**: Every 50 batches
4. **Early stopping**: Patience of 10 epochs (no improvement)
5. **Checkpointing**: Save best model + metadata


## Evaluation

### Nearest Class Mean (NCM) Classifier

After training, the model is evaluated using NCM:

1. Generate **class prototypes** (mean embeddings per building)
2. Classify test images by finding nearest prototype
3. Report Top-1 and Top-3 accuracy


## Loss Functions

### ProxyAnchor Loss

**Concept:** Learn proxies (anchors) for each class; optimize embeddings to be close to positive proxies and far from negative proxies.

**Parameters:**
- `margin`: Distance threshold (typical: 0.1 - 1.2)
- `alpha`: Temperature scaling (typical: 16 - 64)

**Effect of Hyperparameters:**
- Higher `margin`: Larger separation between classes
- Higher `alpha`: Sharper decision boundaries
- Both affect convergence speed and final accuracy

**Implementation:**
```python
from pytorch_metric_learning.losses import ProxyAnchorLoss

loss_fn = ProxyAnchorLoss(
    num_classes=num_buildings,
    embedding_size=embedding_dim,
    margin=0.2,
    alpha=32.0
)
```