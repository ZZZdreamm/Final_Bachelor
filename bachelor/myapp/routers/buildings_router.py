from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from myapp.database.metadata.building_client import building_metadata_client
from myapp.database.buckets.building_client import building_bucket_client
from myapp.models.Building import BuildingCreate, BuildingColumns
from myapp.database.buckets.config import BuildingBucketFolders

router = APIRouter(
    prefix="/buildings",
    tags=["buildings"]
)
    
@router.post(
    "/add_model/",
    response_description="3D Model has been succesfully added!",
    status_code=status.HTTP_200_OK,
)
async def add_building_model(
    building = Form(...),
    model_file: UploadFile = File(...),
):  
    """
    Upload 3D model building
    """
    building: BuildingCreate = BuildingCreate.model_validate_json(building)

    response = building_metadata_client.table_request().select(BuildingColumns.MODEL_URL.value).eq(BuildingColumns.NAME.value, building.name).execute()
    if response.data:
        raise HTTPException(status_code=404, detail=f"Building with name '{building.name}' already exists.")
    
    file_name = f"{BuildingBucketFolders.MODELS.value}{model_file.filename}"
    model_bytes = await model_file.read()
    await building_bucket_client.upload_model(model_bytes, file_name)
    
    building.model_url = file_name
    create_response = building_metadata_client.insert(building)

    return { "status_code": 200, "message": "3D Model has been succesfully added!", "data": create_response }