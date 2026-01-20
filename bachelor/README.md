---
title: 3D Model Search API
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - computer-vision
  - 3d
  - deep-learning
  - fastapi
  - pytorch
  - image-classification
  - similarity-search
  - pose-estimation
short_description: 3D building model classification and pose estimation
python_version: 3.12
app_file: main.py
---

# Bachelor Project - 3D Model Search API

A FastAPI-based backend service for 3D building model classification, search, and pose estimation using deep learning models.

## myapp - Main Application Server

The `myapp` folder contains the production-ready FastAPI server that provides REST API endpoints for 3D model search and classification.

### Contents

- **main.py** - FastAPI application with CORS middleware and model loading lifecycle management
- **routers/** - API route handlers:
  - `buildings_router.py` - Building model CRUD operations
  - `buildings_search_router.py` - Image-based 3D model search endpoints
- **database/** - Database interaction layers:
  - `buckets/` - B2 cloud storage operations
  - `cache/` - Redis caching layer
  - `metadata/` - Supabase metadata management
  - `vectors/` - Pinecone vector database operations
- **AR/** - AR model logic for pose estimation
- **utils/** - Helper utilities including model loading functions
- **trained_model/** - Trained neural network weights and model files
- **models/** - Pydantic data models for request/response validation
- **logging/** - Application logging configuration

### Starting the Server with Docker Compose

The application uses Docker Compose with GPU support for running the FastAPI server and Redis cache.

#### Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with Docker GPU support
- Environment variables configured in `.env` file

#### Required Environment Variables

Create a `.env` file in the project root with:

```bash
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
B2_KEY_ID=your_b2_key_id
B2_APP_KEY=your_b2_app_key
B2_BUCKET_NAME=your_bucket_name
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_REGION=your_pinecone_region
ENVIRONMENT=development
```

#### Start the Server

```bash7
# Build and start all services (FastAPI server + Redis)
docker-compose up --build
```

The API will be available at `http://localhost:8000`

#### Stop the Server

```bash
docker-compose down
```

### Technology Stack

- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning framework for model inference
- **Transformers** - Hugging Face models for feature extraction
- **Redis** - In-memory caching
- **Supabase** - PostgreSQL database for metadata
- **Pinecone** - Vector database for similarity search
- **B2 Cloud Storage** - Object storage for 3D models and images
- **Trimesh & PyRender** - 3D model processing and rendering

## Other Project Folders

### classificator_training
Scripts and tools for training the 3D model classification neural network using prototype learning and feature extraction.

### pose_estimation_training
Contains multiple approaches for camera pose estimation including YOLO-based pose detection, direct estimation methods, and feature matching algorithms.

### blender
Blender Python scripts for automated 3D model rendering from different camera angles to generate training datasets for classification and pose estimation.

### data_deploy_automation
Automation scripts for deploying 3D model data and metadata to cloud services through server in one script run (B2 storage, Supabase database, Pinecone vectors).

## Development

The project uses Python 3.12 and includes dependency management via:
- `requirements.txt` - Production dependencies
- `pyproject.toml` - Poetry/UV package configuration

