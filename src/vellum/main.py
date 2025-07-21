import base64
from io import BytesIO
from PIL import Image

from vellum.retrieval.documents import load_document
from vellum.retrieval import DocumentStore
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
    print()  # yikes
    documents = DocumentStore('documents')

    doc = load_document('assets/devito.pdf')
    documents.put_document(doc)

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

    limit = 5  # Default limit for query results
    threshold = 11.0  # Default threshold for relevance
    while True:
        print("\nEnter a query (or / + a command):\033[94m")
        query = input("> ")
        print("\033[0m", end='')

        if not query.strip():
            continue  # Skip empty queries

        if query.startswith('/'):
            args = query[1:].split()
            cmd, args = args[0], args[1:]

            match cmd:
                case 'quit' | 'q':
                    print("Exiting...")
                    break

                case 'limit' | 'l':
                    if len(args) != 1 or not args[0].isdigit():
                        print("Usage: /limit <number>")
                        continue
                    limit = int(args[0])
                    print("Component retrieval limit set to", limit)
                    continue

                case 'threshold' | 'thresh' | 't':
                    if len(args) != 1 or not args[0].replace('.', '', 1).isdigit():
                        print("Usage: /threshold <number>")
                        continue
                    threshold = float(args[0])
                    print(f"Component retrieval threshold set to {threshold}")
                    continue

                case _:
                    print(f"Unknown command: {cmd}")
                    continue

        # Query the document store for relevant components
        components = documents.query_documents(
            query,
            limit=limit,
            threshold=threshold,
        )

        print(f"\nRetrieved relevant components:")
        for component in components:
            print(f" - {component['document_uri']}:"
                  f"{str(component['page_number']).zfill(3)}:"
                  f"({component['uri'].split('/')[-1].split('.')[0]})")

        print("\nQuerying LLM...\n")
        message = {
            'role': "user",
            'content': [
                {
                    'type': "text",
                    'text': query,
                },
            ] + [get_image(component['uri'], use_llama_format=True)
                 for component in components],
        }

        # Stream response tokens from the LLM
        print("\033[92m", end='')  # Start green
        message_context += [message]
        for token in chat_model.stream(message_context):
            print(token.text(), end='', flush=True)
        print("\033[0m")  # Reset color


if __name__ == '__main__':
    main()
