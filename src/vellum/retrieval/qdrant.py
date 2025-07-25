import os
from collections.abc import Mapping
from uuid import UUID, uuid4

from dotenv import find_dotenv, load_dotenv
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Batch,
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    MultiVectorComparator,
    MultiVectorConfig,
    PointIdsList,
    PointsSelector,
    PointStruct,
    VectorParams,
    QueryRequest,
    Prefetch,
)

from vellum.retrieval.documents import Component, Document
from vellum.retrieval.embeddings import embed_images, embed_queries
from vellum.utils import config

load_dotenv(find_dotenv())
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    timeout=30,
)


# Describes a multi-vector embedding
MultiVector = list[list[float]]


class MultiVectorStore[EntryType: Mapping[str, object], MetadataType]:
    def __init__(self, name: str, is_multi_category: bool = False):
        self.name = f'{name}_{config.embeddings_vector_size}'
        if not self._make_collection():
            raise RuntimeError(f"Failed to create collection {self.name}")

        self.meta = self._get_meta()
        self.is_multi_category = is_multi_category

    def _make_collection(self) -> bool:
        """
        Creates a multivector collection in Qdrant if it doesn't exist already.
        Returns True if the collection already exists or was created successfully.
        """
        if qdrant_client.collection_exists(self.name):
            return True

        return qdrant_client.create_collection(
            collection_name=self.name,
            vectors_config=self._vectors_config(),
        )

    def _vectors_config(self) -> VectorParams | dict[str, VectorParams]:
        """
        Returns the vector configuration for the collection.
        If multiple categories are used, one of them must be 'meta' for
        collection-level metadata.
        """
        return VectorParams(
            on_disk=False,
            size=config.embeddings_vector_size,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM,
            ),
        )

    def _make_id(self) -> UUID:
        return uuid4()

    def _put_meta(self, meta: MetadataType) -> None:
        """
        Stores collection-level metadata.
        """
        qdrant_client.upsert(
            collection_name=self.name,
            points=[PointStruct(
                id=0,
                vector={},  # No associated vectors
                payload={'meta': meta,}  # type: ignore}
            )]
        )

    def _get_meta(self) -> MetadataType | None:
        """
        Retrieves collection-level metadata.
        """
        result = qdrant_client.retrieve(
            collection_name=self.name,
            ids=[0],
        )
        if not result:
            return None
        return result[0].payload['meta']  # type: ignore

    def post_meta(self) -> None:
        """
        Posts collection-level metadata.
        """
        if self.meta is not None:
            self._put_meta(self.meta)

    def put_one(self,
                vectors: MultiVector | dict[str, MultiVector],
                meta: EntryType) -> None:
        """
        Adds a multi-vector embedding to the collection with associated
        metadata. Returns the point ID of the newly added vector.
        """
        if isinstance(vectors, dict):
            # Multiple vector categories
            self.put_many(
                vectors={k: [v] for k, v in vectors.items()},
                datas=[meta],
            )
        else:
            # Single vector category
            self.put_many(
                vectors=[vectors],
                datas=[meta],
            )


    def put_many(self,
                 vectors: list[MultiVector] | dict[str, list[MultiVector]],
                 datas: list[EntryType],
                 batch_size: int = 8) -> None:
        """
        Adds multiple multi-vector embeddings to the collection with
        associated metadata.
        """
        num_points = (len(vectors) if isinstance(vectors, list)
                      else len(next(iter(vectors.values()))))

        for i in range(0, num_points, batch_size):
            l, r = i, min(i + batch_size, num_points)
            batch_vectors = vectors[l:r] if isinstance(vectors, list) else {
                k: v[l:r] for k, v in vectors.items()
            }

            qdrant_client.upsert(
                collection_name=self.name,
                points=Batch(
                    ids=[str(self._make_id()) for _ in range(l, r)],
                    vectors=batch_vectors,  # type: ignore
                    payloads=datas[l:r]  # type: ignore
                ),
            )

    def query(self,
              vector: MultiVector,
              limit: int = 8,
              threshold: float | None = None) -> list[tuple[EntryType, float]]:
        """
        Performs a similarity search in a single vector collection.
        """
        query_result = qdrant_client.query_points(
            collection_name=self.name,
            query=vector,
            limit=limit,
            score_threshold=threshold,
        )

        result: list[tuple[EntryType, float]] = [
            (point.payload, point.score) for point in query_result.points  # type: ignore
        ]
        return result

    def query_by(self,
                 by: str,
                 vector: MultiVector,
                 *,
                 limit: int = 8,
                 prefetch: dict[str, MultiVector] | None = None,
                 prefetch_limit: int = 32,
                 threshold: float | None = None) -> list[tuple[EntryType, float]]:
        """
        Performs a similarity search by a specific vector category with
        optional prefetching by other categories.
        """
        query_req = QueryRequest(
            using=by,
            query=vector,

            prefetch=[
                Prefetch(
                    using=cat,
                    query=vec,
                    limit=prefetch_limit,
                )
                for cat, vec in prefetch.items()
            ] if prefetch else None,

            limit=limit,
            score_threshold=threshold,
            with_payload=True,
        )

        # Can't pass a QueryRequest to unbatched query for some reason
        query_result = qdrant_client.query_batch_points(
            collection_name=self.name,
            requests=[query_req],
        )[0]
        result: list[tuple[EntryType, float]] = [
            (point.payload, point.score) for point in query_result.points  # type: ignore
        ]

        return result

    def _delete(self, selector: PointsSelector) -> None:
        """
        Deletes points from the collection that match the given filter.
        """
        qdrant_client.delete(
            collection_name=self.name,
            points_selector=selector,
        )

    def delete_by_ids(self, *ids: UUID) -> None:
        """
        Deletes points from the collection by their IDs.
        """
        self._delete(PointIdsList(points=list(map(str, ids))))

    def delete_by_filter(self, filter: Filter) -> None:
        """
        Deletes points from the collection that match the given filter.
        """
        self._delete(FilterSelector(filter=filter))


class DocumentStore(MultiVectorStore[Component, dict[Document, int]]):
    """
    A vector store for document retrieval.
    """
    def __init__(self, name: str):
        super().__init__(name)
        if self.meta is None:
            self.meta = {}
            self.post_meta()

    def _vectors_config(self) -> dict[str, VectorParams]:
        return {
            'page': VectorParams(
                on_disk=True,
                size=config.embeddings_vector_size,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM,
                ),
            ),
            'component': VectorParams(
                on_disk=True,
                size=config.embeddings_vector_size,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM,
                ),
            )
        }

    def get_document_version(self, uri: str) -> int:
        """
        Gets the metadata version of the document if it exists in the store,
        or -1 if it does not.
        """
        # return any(doc['uri'] == uri for doc in self.meta)
        if self.meta is not None and uri in self.meta:
            return self.meta[uri]
        return -1

    def put_document(self, document: Document, batch_size: int = 8) -> None:
        """
        Adds a document to the store if it does not already exist.
        """
        stored_data_version = self.get_document_version(document['uri'])
        if stored_data_version == document['version']:
            print(f"Document {document['uri']}@{document['version']} exists in the store.")
            return

        print(f"Document {document['uri']} missing or outdated on remote")

        # Update collection metadata to reflect the new data version
        self.meta[document['uri']] = document['version']
        self.post_meta()

        # Delete all old entries with the given document URI
        print("Removing old entries from remote...")
        self.delete_by_filter(
            filter=Filter(
                must=FieldCondition(
                    key='document_uri',
                    match=MatchValue(value=document['uri']),
                ),
            ),
        )

        # Embed page components and store them
        # TODO: We store a copy of the page tensor for each component because it's easy
        # but there has to be a way to associate multiple components with the single page tensor
        print("Generating page/component embeddings...")
        batch_vectors: dict[str, list[MultiVector]] = {}
        batch_components: list[Component] = []
        for page in document['pages']:
            components = page['components']
            images = [
                Image.open(page['uri'])
            ] + [
                Image.open(component['uri'])
                for component in components
            ]

            # Retrieve embeddings for the page and components
            # FIXME: AAAAAAAAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaahhhhhhhhhhhhhhhhhhhhhhhhhhhhHHHHHHh
            embeddings = [vec.float().cpu().tolist() for vec in embed_images(*images)]
            page_vector, component_vectors = embeddings[0], embeddings[1:]

            # Extend the batch vectors and components
            batch_vectors.setdefault('page', []).extend([page_vector] * len(component_vectors))
            batch_vectors.setdefault('component', []).extend(component_vectors)
            batch_components.extend(components)

        # Upsert the batch of components
        print("Uploading components...")
        self.put_many(
            vectors=batch_vectors,
            datas=batch_components,
            batch_size=batch_size,
        )

    def query_documents(self,
                        query: str,
                        limit: int = 8,
                        prefetch_limit: int = 32,
                        threshold: float | None = None) -> list[tuple[Component, float]]:
        """
        Queries the document store for documents matching the given query.
        """
        # FIXME: godawful code i'll fix it some day
        query_embedding = [vec.float().cpu().tolist() for vec in embed_queries(query)][0]
        return self.query_by(
            by='component',
            vector=query_embedding,
            limit=limit,
            prefetch={
                'page': query_embedding,  # Prefetch by the full page vectors
            },
            prefetch_limit=prefetch_limit,
            threshold=threshold,
        )
