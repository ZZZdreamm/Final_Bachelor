from fastapi import HTTPException
from postgrest import SyncRequestBuilder
from supabase import Client

from myapp.models.Building import BuildingColumns, Location

class DatabaseSQLClient:
    def __init__(self, table: str, database_connection: Client):
        self.table = table
        self.client = database_connection
        
    def insert(self, data: dict):
        response = self.client.table(self.table).insert(data).execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail=f"Supabase insert failed: {response.data}")

        inserted_row = response.data[0]
        inserted_row[BuildingColumns.LOCATION.value] = _geojson_to_location(inserted_row[BuildingColumns.LOCATION.value])
        
        return inserted_row
    
    def update(self, property_value, data: dict, property_name: str = BuildingColumns.ID.value):
        response = self.client.table(self.table).update(data).eq(property_name, property_value).execute()
        
        if not response.data:
            raise HTTPException(status_code=500, detail=f"Supabase update failed: {response.data}")
        
        updated_row = response.data[0]
        updated_row[BuildingColumns.LOCATION.value] = _geojson_to_location(updated_row[BuildingColumns.LOCATION.value])
        
        return updated_row
    
    def delete(self, property_value, property_name: str = BuildingColumns.ID.value):
        return self.client.table(self.table).delete().eq(property_name, property_value).execute()
    
    def record_exists(self, property_value, property_name: str = BuildingColumns.ID.value) -> bool:
        response = self.client.table(self.table).select(BuildingColumns.ID.value).eq(property_name, property_value).execute()
        return bool(response.data)
    
    def table_request(self) -> SyncRequestBuilder:
        return self.client.table(self.table)

def _geojson_to_location(location_geojson: dict) -> Location:
    try:
        lon, lat = location_geojson["coordinates"]
    except (KeyError, ValueError, TypeError):
        raise ValueError("Invalid GeoJSON format for Point")

    return Location(latitude=lat, longitude=lon)
