from typing import List

from supabase import Client

from myapp.database.metadata.client import DatabaseSQLClient
from myapp.database.metadata.config import Tables, database_client
from myapp.models.Building import BuildingColumns, BuildingCreate, BuildingSearch, BuildingUpdateMetadata
from myapp.logging.log import log_call

class BuildingSQLClient(DatabaseSQLClient):
    def __init__(self, database_connection: Client):
        super().__init__(Tables.BUILDINGS.value, database_connection)
        
    @log_call("Inserted building '{building.name}' into metadata database")
    def insert(self, building: BuildingCreate):
        response = self.table_request().select(BuildingColumns.ID.value).eq(BuildingColumns.NAME.value, building.name).execute()
        
        point_wkt = f"SRID=4326;POINT({building.location.longitude} {building.location.latitude})" if building.location else None
        data = {
            BuildingColumns.NAME.value: building.name,
            BuildingColumns.MODEL_URL.value: building.model_url,
            BuildingColumns.LOCATION.value: point_wkt,
            BuildingColumns.HEIGHT.value: building.height,
            BuildingColumns.WIDTH.value: building.width,
            BuildingColumns.DEPTH.value: building.depth,
        }
    
        if response.data:
            building_id = response.data[0][BuildingColumns.ID.value]
            return super().update(building_id, data, BuildingColumns.ID.value)
        else:
            return super().insert(data)
        
    def update(self, building: BuildingUpdateMetadata):
        point_wkt = f"SRID=4326;POINT({building.location.longitude} {building.location.latitude})" if building.location else None
        data = {
            BuildingColumns.LOCATION.value: point_wkt,
            BuildingColumns.HEIGHT.value: building.height,
            BuildingColumns.WIDTH.value: building.width,
            BuildingColumns.DEPTH.value: building.depth,
        }
        return super().update(building.name, data, property_name=BuildingColumns.NAME.value)
    
    @log_call("Deleted building '{building_name}' from metadata database")
    def delete(self, building_name: str):
        return super().delete(building_name, property_name=BuildingColumns.NAME.value)
    
    def record_exists(self, building_name: str) -> bool:
        return super().record_exists(building_name, property_name=BuildingColumns.NAME.value)
    
    def get_all_buildings(self) -> List[BuildingSearch]:
        response = self.table_request().select(
            BuildingColumns.NAME.value, BuildingColumns.MODEL_URL.value, BuildingColumns.MODEL_URL.value,
            BuildingColumns.HEIGHT.value, BuildingColumns.WIDTH.value, BuildingColumns.DEPTH.value).execute()
        return response.data

    
building_metadata_client = BuildingSQLClient(database_connection=database_client)