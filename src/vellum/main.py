import base64
from io import BytesIO
from PIL import Image

from vellum.retrieval.documents import DocumentStore
from vellum.llm import chat_model


def get_image(uri: str, use_llama_format: bool = True) \
        -> dict[str, str | dict[str, str]]:
    """
    Returns a dictionary with the image URI and its type.
    """
    image = Image.open(uri)
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_data = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

    if use_llama_format:
        # Llama4 has a different image object format
        return {
            'type': "image_url",
            'image_url': {
                'url': f"data:image/png;base64,{image_data}",
            },
        }

    # Direct image object for e.g. qwen
    return {
        'type': "image",
        'source_type': "base64",
        'mime_type': "image/png",
        'data': image_data,
    }


def main() -> None:
    print("\n\n")  # yikes
    documents = DocumentStore('documents')
    documents.put_document('assets/devito.pdf')

    message_context = [
        {
            'role': 'system',
            'content': (
                "You are a helpful assistant that answers questions about "
                "documents you have access to. Relevant pages will be "
                "provided with each query; use the attached context to "
                "answer questions."
                ""
                "Keep responses short and to the point; only use information "
                "provided in the input context. Do not repeat yourself. "
                "Answer the question directly and concisely, and nothing "
                "more."
            ),
        }
    ]

    while True:
        print("\nEnter a query (or 'q' to quit):")
        query = input("> ")
        if query.lower() == 'q':
            break

        pages = documents.query_documents(query, limit=1)

        print(f"\nRetrieved relevant pages:")
        for result_page in pages:
            print(f" - {result_page['document_uri']}, "
                  f"p.{result_page['page_number']}\t"
                  f"({result_page['uri']})")

        print("\nQuerying LLM...\n")
        message = {
            'role': "user",
            'content': [
                {
                    'type': "text",
                    'text': query,
                },
            ] + [get_image(page['uri'], use_llama_format=True)
                 for page in pages],
        }

        # Stream response tokens from the LLM
        print("\033[92m", end='')  # Start green
        for token in chat_model.stream(message_context + [message]):
            print(token.text(), end='', flush=True)
        print("\033[0m")  # Reset color


if __name__ == '__main__':
    main()
