from enum import StrEnum
import os
from pinecone import Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION")

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

class VectorIndexes(StrEnum):
    BUILDINGS = "model-prototypes"