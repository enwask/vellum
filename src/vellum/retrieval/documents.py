import json
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from shutil import rmtree
from typing import Any, TypedDict

import pdf2image
from PIL import Image, ImageDraw, ImageFont

from vellum.retrieval.layout import parse_layout
from vellum.utils import config


class Component(TypedDict):
    """
    Represents a component extracted from a document page.
    """
    document_uri: str
    page_number: int
    uri: str
    type: str


class Page(TypedDict):
    """
    Represents a single page of a document.
    """
    document_uri: str
    page_number: int
    uri: str
    components: list[Component]


class Document(TypedDict):
    uri: str
    num_pages: int
    pages: list[Page]
    version: int


# Load type label font
try:
    _debug_label_font = ImageFont.truetype("UbuntuMono-R.ttf", 36)
except Exception:
    _debug_label_font = ImageFont.load_default()


def load_page(document_uri: str,
              data_dir: Path,
              page_number: int,
              page: Image.Image) -> Page:
    """
    Loads a single page of a document, extracting its layout and components.
    """
    # Create the directory structure
    page_dir = data_dir / f'page_{page_number}'
    page_uri = page_dir / 'page.png'

    components_dir = page_dir / 'components'
    components_dir.mkdir(parents=True, exist_ok=True)

    # Save the page image
    page.save(page_uri, format='PNG')

    # Process the page layout
    layout = parse_layout(page)

    # For debugging, we also create a version of the page image with boxes drawn around components
    debug_image = page.copy()

    # Extract components from the layout
    components: dict[str, list[Component]] = {}
    for block in layout:
        component_list = components.setdefault(block.type, [])
        component_uri = components_dir / f'{block.type}_{len(component_list) + 1}.png'

        component_image = page.crop((block.x1, block.y1, block.x2, block.y2))
        component_image.save(component_uri, format='PNG')

        component_list.append(Component(
            document_uri=document_uri,
            page_number=page_number,
            uri=str(component_uri),
            type=block.type,
        ))

        # Draw a bounding box on the debug image
        draw = ImageDraw.Draw(debug_image)
        draw.rectangle(
            (block.x1, block.y1, block.x2, block.y2),
            outline='green',
            width=4,
        )

        # Draw the component label as white-on-green
        label_size = draw.textbbox(
            (block.x1, block.y1 - 5),
            component_uri.stem,
            font=_debug_label_font,
        )
        draw.rectangle(
            (label_size[0], label_size[1],
                label_size[2] + 10, label_size[3] + 10),
            fill='green',
        )
        draw.text(
            (block.x1 + 5, block.y1),
            component_uri.stem,
            fill='white',
            font=_debug_label_font,
        )

    # Save the annotated debug image
    debug_image.save(page_dir / 'page.annotated.png', format='PNG')

    # Create the page object
    return Page(
        document_uri=document_uri,
        page_number=page_number,
        uri=str(page_uri),
        components=list(chain.from_iterable(components.values()))
    )


# FIXME: This should probably be moved to DocumentStore, and maybe store images in Qdrant?
def load_document(document_uri: str,
                  dpi: int = 300,
                  workers: int = 4,
                  **kwargs: Any) -> Document:
    """
    Loads the document and splits it into page images, writing them to a
    pages directory for the document.

    If the document has already been processed, instead loads existing page
    information.
    """
    doc_path = Path(document_uri)
    data_dir = doc_path.parent / '.vellum' / doc_path.stem

    if data_dir.exists():
        # Load existing document data if available
        try:
            with open(data_dir / 'index.json', 'r') as f:
                pages_data = json.load(f)

            # Check if the data version matches
            if pages_data.get('version', -1) != config.data_version:
                # Need to reload the document
                print(f"Document {doc_path} index is outdated; reprocessing...")
                rmtree(data_dir)  # Remove the old data recursively

            else:
                # All good
                document = Document(**pages_data)
                return document

        except Exception as e:
            print(f"Failed to load existing metadata for {doc_path}: {e}")

    # Otherwise, process the document
    data_dir.mkdir(parents=True, exist_ok=True)
    images = pdf2image.convert_from_path(
        pdf_path=document_uri,
        dpi=dpi,
        **kwargs,
    )

    def _page_worker(page_number: int, image: Image.Image) -> Page:
        """
        Worker function to process a single page image.
        """
        return load_page(
            document_uri=document_uri,
            data_dir=data_dir,
            page_number=page_number,
            page=image,
        )

    # Load pages in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pages = list(executor.map(
            _page_worker,
            range(1, len(images) + 1),
            images,
        ))

    # Form an annotated PDF containing all debug images
    debug_pdf_path = data_dir / f'{doc_path.stem}_annotated.pdf'
    debug_images = [Image.open(f"{page['uri'].rsplit('.', 1)[0]}.annotated.png") for page in pages]
    debug_images[0].save(
        debug_pdf_path,
        save_all=True,
        append_images=debug_images[1:],
        resolution=dpi,
    )

    # Create the final document object
    document = Document(
        uri=document_uri,
        num_pages=len(pages),
        pages=pages,
        version=config.data_version,
    )

    # Write the document metadata to a file
    # TODO: loooootta redundant data nested here but it's fine for now
    with open(data_dir / 'index.json', 'w') as f:
        json.dump(document, f, indent=4)

    return document
