from pinecone import Pinecone, QueryResponse

class DatabaseVectorClient:
    def __init__(self, pinecone_client: Pinecone, index_name: str):
        self.index_name = index_name
        self.index = pinecone_client.Index(index_name)
        
    def upsert_vectors(self, vectors: list[dict]):
        result = self.index.upsert(vectors, namespace="prototypes")
        return result
        
    def search(self, vector: list[float], top_k: int = 5, include_metadata: bool = True, filter: dict = None) -> QueryResponse:
        return self.index.query(vector=vector, top_k=top_k, include_metadata=include_metadata, filter=filter)
    
    def delete_all_records(self) -> None:
        result = self.index.delete(delete_all=True, namespace="prototypes")
        return result

    