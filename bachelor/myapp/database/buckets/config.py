from enum import Enum
import os
from b2sdk.v2 import InMemoryAccountInfo, B2Api

BUILDING_BUCKET_NAME = os.environ.get("B2_BUCKET_NAME")
BUCKET_STORAGE_PREFIX = "https://f005.backblazeb2.com/file/"
BUILDING_BUCKET_URL_PREFIX = BUCKET_STORAGE_PREFIX + BUILDING_BUCKET_NAME + "/"

class BuildingBucketFolders(str, Enum):
    MODELS = "models/"
    IMAGES = "images/"

b2_info = InMemoryAccountInfo()
b2_api = B2Api(b2_info)
b2_api.authorize_account(
    "production",
    os.environ.get("B2_KEY_ID"),
    os.environ.get("B2_APP_KEY")
)
