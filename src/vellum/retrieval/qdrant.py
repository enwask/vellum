import os
from collections.abc import Mapping
from dotenv import find_dotenv, load_dotenv
from uuid import UUID, uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, MultiVectorComparator,
                                  MultiVectorConfig, PointStruct,
                                  VectorParams)

from vellum.utils import config

load_dotenv(find_dotenv())
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    timeout=10,
)


def make_collection(name: str) -> bool:
    """
    Creates a multivector collection in Qdrant if it doesn't exist already.
    Returns True if the collection already exists or was created successfully.
    """
    if qdrant_client.collection_exists(name):
        return True

    return qdrant_client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            # on_disk=True,
            size=config.embeddings_vector_size,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM,
            ),
        ),
    )


class MultiVectorStore[DataType: Mapping[str, object]]:
    def __init__(self, name: str):
        self.name = f'{name}_{config.embeddings_vector_size}'
        if not make_collection(self.name):
            raise RuntimeError(f"Failed to create collection {self.name}")

    def _make_id(self) -> UUID:
        return uuid4()

    def put_one(self,
                vector: list[list[float]],
                meta: DataType) -> None:
        """
        Adds a multi-vector embedding to the collection with associated
        metadata. Returns the point ID of the newly added vector.
        """
        self.put_many([vector], [meta])

    def put_many(self,
                 vectors: list[list[list[float]]],
                 datas: list[DataType],
                 batch_size: int = 32) -> None:
        """
        Adds multiple multi-vector embeddings to the collection with
        associated metadata.
        """
        for i in range(0, len(vectors), batch_size):
            points = [
                PointStruct(
                    id=str(self._make_id()),
                    vector=vector,
                    payload=meta,  # type: ignore
                )
                for vector, meta in zip(vectors[i:i+batch_size], datas[i:i+batch_size])
            ]

            qdrant_client.upsert(
                collection_name=self.name,
                points=points,
            )

    def query(self,
              vector: list[list[float]],
              limit: int = 10) -> list[DataType]:
        """
        Queries the collection for the most similar multi-vector embeddings
        to the provided vector.
        """
        result = qdrant_client.query_points(
            collection_name=self.name,
            query=vector,
            limit=limit,
        )

        datas: list[DataType] = [point.payload for point in result.points]  # type: ignore
        return datas

