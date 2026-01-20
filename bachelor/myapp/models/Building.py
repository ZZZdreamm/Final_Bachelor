from typing import List, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field

class Location(BaseModel):
    latitude: float = Field(...)
    longitude: float = Field(...)
    
    
class BuildingCreate(BaseModel):
    """
    Model of 3D Building
    """
    name: str = Field(...)
    location: Location = Field(...)
    height: float = Field(...)
    width: float = Field(...)
    depth: float = Field(...)
    model_url: Optional[str] = Field(None)

class Building(BaseModel):
    id: str
    name: str
    location: Location
    height: float = Field(...)
    width: float = Field(...)
    depth: float = Field(...)
    model_url: str
    created_at: datetime
    
    
class BuildingUpdateMetadata(BaseModel):
    name: str = Field(...)
    location: Location = Field(...)
    height: float = Field(...)
    width: float = Field(...)
    depth: float = Field(...)
    
class BuildingSearch(BaseModel):
    name: str
    model_url: str
    height: float
    width: float
    depth: float
    class_id: int
    
class BuildingVector(BaseModel):
    values: List[float]
    class_id: int
    tag: str
    
    
class BuildingColumns(Enum):
    ID = "id"
    NAME = "name"
    MODEL_URL = "model_url"
    LOCATION = "location"
    HEIGHT = "height"
    WIDTH = "width"
    DEPTH = "depth"
    CREATED_AT = "created_at"