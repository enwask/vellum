import os

from dotenv import find_dotenv, load_dotenv
from qdrant_client import QdrantClient

load_dotenv(find_dotenv())
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

vector_store = QdrantClient(url=QDRANT_URL)
