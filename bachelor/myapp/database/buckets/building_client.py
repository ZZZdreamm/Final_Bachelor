from b2sdk.v2 import B2Api
from fastapi import HTTPException

from myapp.database.buckets.client import BucketClient
from myapp.database.buckets.config import BUILDING_BUCKET_NAME, b2_api
from myapp.logging.log import log_call

class BuildingBucketClient(BucketClient):
    def __init__(self, b2_api: B2Api):
        super().__init__(b2_api, BUILDING_BUCKET_NAME)
            
    @log_call("Uploaded model file '{remote_file_name}' to building bucket")
    async def upload_model(self, file_contents: bytes, remote_file_name: str):
        try:
            super().upload_model(file_contents, remote_file_name)
        except Exception as e:  
            raise HTTPException(status_code=500, detail=f"Failed to upload model file: {str(e)}")
         
    @log_call("Deleted file '{remote_file_name}' from building bucket")
    async def delete_file(self, remote_file_name: str):
        super().delete_file(remote_file_name)
         
    async def get_all_file_names(self, path: str = "") -> list[str]:
        try:
            file_names = [file_info.file_name for file_info, _ in self.bucket.ls(path, recursive=True)]
            return file_names
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list files in bucket: {str(e)}")
        

building_bucket_client = BuildingBucketClient(b2_api)