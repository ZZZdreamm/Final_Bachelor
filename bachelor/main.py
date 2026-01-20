from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from myapp.routers import buildings_router
from myapp.routers import buildings_search_router
from myapp.utils.load_best_model import load_best_model
from myapp.AR.model_template import TRAINED_CLASSIFICATION_MODEL

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Loading Neural Network Model...")
    try:
        model, prototype_tensor, class_ids, feature_extractor, id_to_tag, tag_to_id, pose_model = load_best_model()
        TRAINED_CLASSIFICATION_MODEL["model"] = model
        TRAINED_CLASSIFICATION_MODEL["prototype_tensor"] = prototype_tensor
        TRAINED_CLASSIFICATION_MODEL["class_ids"] = class_ids
        TRAINED_CLASSIFICATION_MODEL["feature_extractor"] = feature_extractor
        TRAINED_CLASSIFICATION_MODEL["id_to_tag"] = id_to_tag
        TRAINED_CLASSIFICATION_MODEL['tag_to_id'] = tag_to_id
        TRAINED_CLASSIFICATION_MODEL["pose_model"] = pose_model
        TRAINED_CLASSIFICATION_MODEL["model"].eval()

        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")

    yield 

    print("Shutting down: Releasing resources.")
    TRAINED_CLASSIFICATION_MODEL.clear()

app = FastAPI(title="3D Model Search API", version="1.0.0", lifespan=lifespan)

origins = [
    "http://192.168.0.93:8000", 
    "http://localhost:8000",
    "http://localhost:3000",
    
    "https://uq4wlfc-zzzdream-8081.exp.direct", 

    "http://localhost:8081",
    
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          
    allow_credentials=True,         
    allow_methods=["*"],            
    allow_headers=["*"],            
)

app.include_router(buildings_router.router)
app.include_router(buildings_search_router.router)

@app.get("/", response_description="API is running")
async def root():
    """
    Root endpoint
    """
    return {"message": "API is running"}

