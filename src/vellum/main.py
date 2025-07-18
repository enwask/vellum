from io import BytesIO
from typing import TypedDict
from PIL import Image
import requests

from vellum.retrieval import embed_queries, embed_images
from vellum.retrieval.documents import DocumentStore
from vellum.retrieval.qdrant import MultiVectorStore


class ImageData(TypedDict):
    url: str


def main() -> None:
    documents = DocumentStore('documents')
    documents.put_document('assets/devito.pdf')

    while True:
        query = input("Enter a query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        results = documents.query_documents(query, limit=3)

        print(f"Results for query '{query}':")
        for result_page in results:
            print(f" - Document: {result_page['document_uri']}, "
                  f"page {result_page['page_number']} "
                  f"({result_page['uri']})")


if __name__ == '__main__':
    main()
