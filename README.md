# Building Recognition & 3D Model Search

A full-stack computer vision application that enables users to capture building photos and automatically retrieve their 3D models using AI-powered building recognition and camera pose estimation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mobile App (React Native)                    │
│         Camera capture, image cropping, 3D model viewing        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API (image + optional GPS)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend Server (FastAPI)                     │
│    Multi-modal classification, pose estimation, model serving   │
└────────┬──────────────────────┬──────────────────┬──────────────┘
         │                      │                  │
         ▼                      ▼                  ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│    Supabase     │   │    Pinecone     │   │  Backblaze B2   │
│   (Metadata)    │   │ (Vector Search) │   │ (3D Models)     │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

## Project Structure

```
final_bachelor/
├── Bachelor_mobile/              # React Native mobile application
├── bachelor/                     # Backend and ML training pipelines
│   ├── myapp/                    # Production FastAPI server
│   ├── classificator_training/   # Building classification training
│   ├── pose_estimation_training/ # Camera pose estimation (3 methods tested)
│   ├── blender/                  # Synthetic data generation
└── └── data_deploy_automation/   # Cloud deployment scripts
```

## Components

### Mobile Application ([Bachelor_mobile/](Bachelor_mobile/))

React Native + Expo application for capturing building photos and viewing 3D models.

**Technology:** React Native, Expo 54, Filament 3D, TypeScript

**Workflow:**
1. Capture photo via camera or select from gallery
2. Crop image to focus on the building
3. Send to API (optionally with GPS location for proximity matching)
4. View matched building as interactive 3D model

### Backend Server ([bachelor/myapp/](bachelor/myapp/))

FastAPI service handling building classification and 3D model retrieval.

**Technology:** FastAPI, PyTorch, Transformers, Redis

**Processing Pipeline:**
1. Validate if image contains a building
2. If GPS provided: query nearby buildings from database
3. Extract multi-modal features
4. Classify building using Nearest Class Mean on learned prototypes
5. Return matched 3D model from object storage

**Cloud Services:**
- **Supabase** - Building metadata and geo-spatial queries
- **Pinecone** - Vector similarity search for classification
- **Backblaze B2** - 3D model and image storage
- **Redis** - Response caching

### Classification Training ([bachelor/classificator_training/](bachelor/classificator_training/))

Multi-modal metric learning pipeline for building recognition.

**Approach:** Combine features from different combinations of 8 pre-trained models (CLIP, SegFormer, MiDaS, DPT, ResNet, MobileNet, EfficientNet, ViT), fuse through learned head, train with ProxyAnchor loss.

**Key Features:**
- Frozen backbones for efficient training
- Hard mining strategy
- Pre-computed feature caching
- NCM classifier for inference

### Pose Estimation ([bachelor/pose_estimation_training/](bachelor/pose_estimation_training/))

Three independent methods for 6DoF camera pose estimation:

| Method | Approach | 
|--------|----------|
| Direct Estimation | Neural network regression from features |
| Feature Matching | Keypoint matching + PnP-RANSAC |
| YOLO-Pose | Semantic keypoint detection + PnP |

### Data Generation ([bachelor/blender/](bachelor/blender/))

Blender scripts for rendering synthetic training data with ground truth camera poses and annotations.

### Deployment Automation ([bachelor/data_deploy_automation/](bachelor/data_deploy_automation/))

Scripts for deploying building data (3D models, metadata, embeddings) to cloud services.

## Getting Started

### Mobile App

```bash
cd Bachelor_mobile
npm install
npm run start
```

Requires Expo development build on device. See [Bachelor_mobile/README.md](Bachelor_mobile/README.md) for details.

### Backend Server

```bash
cd bachelor
# Configure environment variables in .env
docker-compose up --build
```

Requires NVIDIA GPU with Docker support. See [bachelor/README.md](bachelor/README.md) for configuration.

**Required Environment Variables:**
```bash
SUPABASE_URL=...
SUPABASE_KEY=...
B2_KEY_ID=...
B2_APP_KEY=...
B2_BUCKET_NAME=...
PINECONE_API_KEY=...
PINECONE_REGION=...
```

## Deployment

- **Development:** Docker Compose with local GPU
- **Production:** Hugging Face Spaces (Docker SDK)

API Endpoints:
- Development: `http://localhost:8000`
- Production: `https://zzzdream95-bachelor.hf.space/`

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Mobile | React Native, Expo, Filament, TypeScript |
| Backend | FastAPI, PyTorch, Transformers, Uvicorn |
| Storage | Supabase (PostgreSQL), Pinecone, Backblaze B2, Redis |
| ML Training | PyTorch, ProxyAnchor Loss, CLIP, SegFormer |
| Data Generation | Blender 4.0+ |
