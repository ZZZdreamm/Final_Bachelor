# Pose Estimation Training

Training pipelines for 6DoF camera pose estimation using multiple approaches: direct regression, feature matching with PnP, and YOLO-based keypoint detection.

## Overview

This directory contains three independent methods for estimating 6-degree-of-freedom (6DoF) camera pose relative to 3D building models:

1. **Direct Estimation**: End-to-end neural network regression from image features to pose
2. **Feature Matching**: Match keypoints between query and rendered images, solve PnP
3. **YOLO-Pose**: Detect semantic keypoints on buildings for pose estimation

Each method has trade-offs between accuracy, speed, and data requirements.

## Directory Structure & File Roles

### Common Utilities (`common/`)
Shared code used across all three pose estimation methods:

- **[pose_utils.py](common/pose_utils.py)** - Math operations for quaternions, rotation matrices, and pose transformations
- **[camera_utils.py](common/camera_utils.py)** - Extracting camera parameters and poses from Blender scenes
- **[camera_config.py](common/camera_config.py)** - Camera calibration parameters and intrinsic matrices
- **[mesh_utils.py](common/mesh_utils.py)** - Loading and processing 3D building meshes
- **[blender_utils.py](common/blender_utils.py)** - Managing Blender scenes, rendering setup, and environment configuration

### Method 1: Direct Estimation (`direct_estimation/`)
Neural network that directly regresses camera pose from image features:

- **[pose_network.py](direct_estimation/pose_network.py)** - Network architecture (BuildingPoseNetCached) and pose estimation logic
- **[train_pose_network.py](direct_estimation/train_pose_network.py)** - Training loop, data loading, and model checkpointing
- **[batch_inference.py](direct_estimation/batch_inference.py)** - Run pose prediction on multiple images efficiently
- **[pose_refinement.py](direct_estimation/pose_refinement.py)** - Iteratively improve pose estimates through rendering and comparison
- **[blender.py](direct_estimation/blender.py)** - Generate synthetic training data by rendering building models

### Method 2: Feature Matching (`feature_matching/`)
Match keypoints between query and reference images, then solve for pose using PnP:

- **[extract_features.py](feature_matching/extract_features.py)** - Build database of features extracted from rendered images
- **[pose_test_ensemble.py](feature_matching/pose_test_ensemble.py)** - Complete pipeline: retrieval → matching → PnP solving
- **[pose_test_manual.py](feature_matching/pose_test_manual.py)** - Interactive tool for testing matching against specific renders
- **[keypoints_visualization.py](feature_matching/keypoints_visualization.py)** - Visualize detected keypoints and matches
- **[blender.py](feature_matching/blender.py)** - Render reference images with known camera poses

#### Feature Matching Utilities (`feature_matching/common/`)
- **[matchers.py](feature_matching/common/matchers.py)** - Implementations of LoFTR, SuperPoint, SIFT, and ensemble matching
- **[pose_utils.py](feature_matching/common/pose_utils.py)** - PnP-RANSAC solving and finding 3D correspondences
- **[retrieval.py](feature_matching/common/retrieval.py)** - Global descriptor-based image retrieval to find candidate matches
- **[visualization.py](feature_matching/common/visualization.py)** - Draw matches, pose overlays, and 3D bounding boxes

### Method 3: YOLO-Pose (`yolo-pose/`)
Detect semantic keypoints on building facades for pose estimation:

- **automatic/create_renders.py** - Generate training images with keypoint annotations using Blender
- **automatic/visualize_points.py** - Visualize 3D keypoints projected onto images
- **semantic/annotator.py** - Interactive GUI tool for manual keypoint annotation and pose refinement

## Methods Comparison

| Method | Pros | Cons |
|--------|------|------|
| **Direct Estimation** | Fast (single forward pass), handles ambiguity | Requires large dataset, potential bias |
| **Feature Matching** | Interpretable, no dataset size limit | Slower, sensitive to parameters |
| **YOLO-Pose** | Semantic keypoints, works with fixed points | Requires keypoint annotations |
## Installation

### Dependencies

```bash
pip install torch torchvision opencv-python numpy h5py trimesh pyrender
pip install kornia lightglue
pip install wandb
```

### Blender Integration

For rendering and data generation, install Blender 4.0+ and add Python API:

```bash
pip install bpy
```

## Workflow by Method

### Method 1: Direct Estimation

**Workflow:**
1. Use `blender.py` to render synthetic training images from 3D models
2. Use `clip_features.py` to Extract CLIP features from images and cache them to HDF5 format
3. Run `train_pose_network.py` to train the neural network on cached features
4. Use `batch_inference.py` to predict poses for new images
5. Optionally use `pose_refinement.py` to iteratively improve predictions


### Method 2: Feature Matching

**Workflow:**
1. Use `blender.py` to render reference images from known camera viewpoints
2. Run `extract_features.py` to build a database of keypoints and descriptors from renders
3. For new query images, use `pose_test_ensemble.py` to:
   - Retrieve similar renders using global descriptors
   - Match local features between query and candidates
   - Solve PnP-RANSAC to estimate camera pose
4. Use `pose_test_manual.py` to debug specific image pairs
5. Use `keypoints_visualization.py` to inspect feature detection quality


### Method 3: YOLO-Pose

**Workflow:**
1. Use `automatic/create_renders.py` to generate annotated training data
2. Train a YOLO model to detect these keypoints in images using `yolo_train.py`
3. Use detected keypoints with PnP to estimate pose
4. Use `semantic/annotator.py` for manual annotation of test data



## Data Requirements

### Direct Estimation
- **Input:** HDF5 files containing pre-extracted CLIP features and ground truth poses
- **Minimum size:** 500+ images per building for robust training
- **Generated by:** `blender.py` creates renders, separate feature extraction caches them
- **Stored in:** HDF5 format for fast random access during training

### Feature Matching
- **Input:** Rendered reference images with known camera parameters
- **Minimum size:** 300+ diverse viewpoints per building
- **Generated by:** `blender.py` with systematic camera sampling
- **Processed by:** `extract_features.py` builds searchable feature database
- **Storage:** Original renders + pickle file with extracted features and metadata

### YOLO-Pose
- **Input:** Images annotated with 2D keypoint locations and visibility
- **Minimum size:** 500+ images with visible keypoints
- **Generated by:** `automatic/create_renders.py` for synthetic data from Blender
- **Annotation:** `semantic/annotator.py` for manual labeling of test data
- **Format:** YOLO-compatible annotation files (TXT)
