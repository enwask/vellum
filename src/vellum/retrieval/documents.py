import pdf2image
from PIL import Image
from pathlib import Path
from typing import TypedDict

from vellum.retrieval.embeddings import embed_images, embed_queries
from vellum.retrieval.qdrant import MultiVectorStore


class Page(TypedDict):
    document_uri: str
    page_number: int
    uri: str


class Document(TypedDict):
    uri: str
    num_pages: int


def process_pages(document_uri: str, dpi: int = 300, **kwargs) -> list[Page]:
    """
    Loads the document and splits it into page images, writing them to a
    pages directory for the document.
    """
    images = pdf2image.convert_from_path(
        pdf_path=document_uri,
        dpi=dpi,
        **kwargs,
    )

    doc_path = Path(document_uri)
    pages_dir = doc_path.parent / f"{doc_path.stem}_pages"
    pages_dir.mkdir(exist_ok=True)

    pages: list[Page] = []
    for i, image in enumerate(images, start=1):
        page_uri = str(pages_dir / f"{doc_path.stem}_page_{i}.png")
        image.save(page_uri, format='PNG')

        pages.append(Page(
            document_uri=document_uri,
            page_number=i,
            uri=page_uri,
        ))

    return pages


def load_document(uri: str) -> tuple[Document, list[Page]]:
    """
    Loads a document and its pages from the given URI.
    """
    pages: list[Page] = process_pages(uri)
    return Document(
        uri=uri,
        num_pages=len(pages),
    ), pages


class DocumentStore(MultiVectorStore[Page, list[Document]]):
    """
    A vector store for document retrieval.
    """
    def __init__(self, name: str):
        super().__init__(name)
        if self.meta is None:
            self.meta = []
            self.post_meta()

    def has_document(self, uri: str) -> bool:
        """
        Checks if a document with the given URI exists in the store.
        """
        return any(doc['uri'] == uri for doc in self.meta)

    def put_document(self, uri: str, batch_size: int = 8) -> None:
        """
        Adds a document to the store if it does not already exist.
        """
        if self.has_document(uri):
            print(f"Document {uri} already exists in the store.")
            return

        print(f"Adding document {uri} to the store...")

        # Load pages from the document
        document, pages = load_document(uri)

        # Update collection metadata to reflect the new document
        self.meta.append(document)
        self.post_meta()

        # Embed pages and store them
        embeddings = embed_images(*[Image.open(page['uri']) for page in pages]) \
            .float().cpu().tolist()
        self.put_many(embeddings, pages, batch_size=batch_size)

    def query_documents(self, query: str, limit: int = 10,
                        threshold: float | None = None) -> list[Page]:
        """
        Queries the document store for documents matching the given query.
        """
        query_embedding = embed_queries(query).float().cpu()[0].tolist()
        return self.query(query_embedding, limit=limit, threshold=threshold)
