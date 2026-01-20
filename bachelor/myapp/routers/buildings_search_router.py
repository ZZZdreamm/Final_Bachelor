import base64
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from myapp.AR.model_template import TRAINED_CLASSIFICATION_MODEL
from myapp.models.Building import BuildingSearch, Location
from myapp.AR.pose_estimate_utils import process_image_heuristic
from myapp.AR.search_utils import get_buildings_in_proximity, get_image_predicted_classes, is_building_on_image
from myapp.database.buckets.building_client import building_bucket_client
from myapp.database.cache.config import CACHE_EXPIRATION_SECONDS, redis_client
from myapp.database.metadata.building_client import building_metadata_client
from myapp.utils.image_utils import sanitize_filename

router = APIRouter(
    prefix="/buildings_search",
    tags=["buildings_search"]
)

@router.post(
    "/find/",
    response_description="Get building",
    status_code=status.HTTP_200_OK,
)
async def find_historical_models(location: str = Form(""), building_image: UploadFile = File(...)):
    """
    Search for place matching to photo
    """
    try:
        lat, lon = map(float, location.split(","))
        location = Location(latitude=lat, longitude=lon)
    except:
        pass
    
    img_bytes = await building_image.read() 
    
    is_building = is_building_on_image(img_bytes)
    if not is_building:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No building detected in the image")
    
    if location:
        nearby_buildings: List[BuildingSearch] = get_buildings_in_proximity(location.latitude, location.longitude)
    else:
        nearby_buildings: List[BuildingSearch] = building_metadata_client.get_all_buildings()

    tag_to_id = TRAINED_CLASSIFICATION_MODEL["tag_to_id"]
    mapped_buildings = [BuildingSearch(**building, class_id=tag_to_id[building['name']]) for building in nearby_buildings]
    
    result_buildings = get_image_predicted_classes(img_bytes, mapped_buildings)
    matched_building = result_buildings[0] if result_buildings else None
    
    if not matched_building:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No building from database nearby")
    
    model_url = matched_building.model_url
    cache_key = f"model_file:{model_url}"
    
    try:
        model_bytes = redis_client.get(cache_key)
        
        if model_bytes:
            print(f"CACHE HIT for key: {cache_key}")
        else:
            print(f"CACHE MISS for key: {cache_key}. Fetching from bucket.")
            model_bytes = building_bucket_client.get_file_by_url(model_url)
            redis_client.set(cache_key, model_bytes, ex=CACHE_EXPIRATION_SECONDS)

    except Exception as e:
        print(f"Redis operation failed, fetching directly: {e}")
        model_bytes = building_bucket_client.get_file_by_url(model_url)
    
    changed_image_bytes = process_image_heuristic(img_bytes, model_bytes, matched_building.name)
    
    model_filename = sanitize_filename(matched_building.name + ".glb")

    model_base64 = base64.b64encode(model_bytes).decode("utf-8")
    image_base64 = base64.b64encode(changed_image_bytes).decode("utf-8")

    return JSONResponse(
        content={
            "model": {
                "filename": model_filename,
                "media_type": "model/gltf-binary",
                "data_base64": model_base64
            },
            "image": {
                "filename": building_image.filename,
                "media_type": building_image.content_type,
                "data_base64": image_base64
            },
            "building_info": {
                "name": model_filename,
                "url": model_url
            }
        }
    )
