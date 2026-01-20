import io
from b2sdk.v2 import B2Api
from fastapi import HTTPException

class BucketClient:
    def __init__(self, b2_api: B2Api, bucket_name: str):
        self.bucket = b2_api.get_bucket_by_name(bucket_name)

    def upload_model(self, file_bytes: bytes, remote_file_name: str):
        try:
            self.bucket.get_file_info_by_name(remote_file_name)
            return 
        except Exception:
            pass
        self.bucket.upload_bytes(file_bytes, remote_file_name)
        
    def get_file_by_url(self, image_url: str):
        downloaded_file = io.BytesIO()
        
        self.bucket.download_file_by_name(image_url).save(downloaded_file)
        downloaded_file.seek(0)
        
        return downloaded_file.read()

    def delete_file(self, remote_file_name: str):
        try:
            file_version = self.bucket.get_file_info_by_name(remote_file_name)
            if not file_version:
                raise HTTPException(status_code=404, detail=f"File '{remote_file_name}' not found in bucket.")
            self.bucket.delete_file_version(file_version.id_, remote_file_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete file '{remote_file_name}': {str(e)}")
        
    async def delete_folder(self, folder_path: str):
        try:
            files_to_delete = [file_info for file_info, _ in self.bucket.ls(folder_path, recursive=True)]
            for file_info in files_to_delete:
                self.bucket.delete_file_version(file_info.id_, file_info.file_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete folder '{folder_path}': {str(e)}")