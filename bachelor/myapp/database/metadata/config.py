import os
from enum import Enum
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
database_client: Client = create_client(url, key)

class Tables(str, Enum):
    BUILDINGS = "buildings"
    
    
class DatabaseQueries(str, Enum):
    GET_NEARBY_PLACES = "get_nearby_places"