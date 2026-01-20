import hashlib
from pinecone import Pinecone, QueryResponse
from myapp.database.vectors.client import DatabaseVectorClient
from myapp.database.vectors.config import VectorIndexes, pinecone_client
from myapp.logging.log import log_call
from myapp.models.Building import BuildingColumns, BuildingVector

# Vector client for building embeddings if approach using feature matching would be used
class BuildingVectorClient(DatabaseVectorClient):
    def __init__(self, database_connection: Pinecone):
        super().__init__(database_connection, VectorIndexes.BUILDINGS.value)
        
    def upsert_vectors(self, vectors: list[BuildingVector]):
        building_vectors = []
        for vector in vectors:
            vector_id = self._hash_image_url(vector['tag'])
            building_vectors.append({
                BuildingColumns.ID.value: vector_id,
                "values": vector['values'],
                "metadata": {"tag": vector['tag'], "class_id": vector['class_id']}
            })
        return super().upsert_vectors(building_vectors)
    
    @log_call("Searched top {top_k} vectors in building vector index")
    def search(self, vector: list[float], top_k: int = 5, include_metadata: bool = True, filter: dict = None) -> QueryResponse:
        return super().search(vector, top_k, include_metadata, filter)    
        
    @log_call("Deleted vectors with url: {model_url} from building vector index")
    def delete_by_url(self, model_url: str):
        """
        Deletes all vectors whose metadata.url matches the given URL.
        """
        self.index.delete(filter={"model_url": {"$eq": model_url}})
        
    @log_call("Deleted all vectors from building vector index")
    def delete_all(self):
        result = super().delete_all_records()
        return result
    
    
    def _hash_image_url(self, img_name: str) -> str:
        """Generate SHA-256 hash of image content for a stable ID."""
        img_encoded = img_name.encode('utf-8')
        return hashlib.sha256(img_encoded).hexdigest()
        
        
building_vector_client = BuildingVectorClient(database_connection=pinecone_client)