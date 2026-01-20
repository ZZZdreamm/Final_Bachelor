from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from myapp.database.metadata.building_client import building_metadata_client
from myapp.database.buckets.building_client import building_bucket_client
from myapp.database.vectors.building_client import building_vector_client
from myapp.models.Building import Building, BuildingCreate, BuildingVector, BuildingColumns
from myapp.database.buckets.config import BuildingBucketFolders

router = APIRouter(
    prefix="/buildings",
    tags=["buildings"]
)

@router.post(
    "/add/",
    response_description="Building has been succesfully added!",
    status_code=status.HTTP_200_OK,
    response_model=Building,
)
async def add_building(
    building = Form(...),
    model_renders_files: List[UploadFile] = File(None),
):  
    building: Building = BuildingCreate.model_validate_json(building)
    
    if len(model_renders_files) < 4:
        raise HTTPException(status_code=400, detail="At least 4 model renders must be provided.")
    
    renders_vectors: List[BuildingVector] = []
    file_folder_name = f"{BuildingBucketFolders.MODELS.value}{building.name}/"
    for model_render_file in model_renders_files:
        remote_file_name: str = file_folder_name + model_render_file.filename
        img_bytes = await model_render_file.read()
        img_embeddings = img_bytes
        renders_vectors.append(BuildingVector(
            values=img_embeddings,
            img_bytes=img_bytes,
            model_url=file_folder_name,
            render_url=remote_file_name
        ))

    for batch_start in range(0, len(renders_vectors), 25):
        batch = renders_vectors[batch_start:batch_start + 25]
        if batch:
            building_vector_client.upsert_vectors(batch)
    
    building.model_url = file_folder_name
    building = building_metadata_client.insert(building) 
    if not building:
        raise HTTPException(status_code=500, detail="Failed to insert building model.")
    
    return building

@router.get(
    "/",
    response_description="List all building names",
    status_code=status.HTTP_200_OK,
    response_model=list[str],
)
async def get_all_buildings_names() -> list[str]:
    """
    Return list of buildings names stored in database.
    """
    return await building_metadata_client.get_all_buildings_names()

    
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

    return create_response