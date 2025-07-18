import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from torch import bfloat16, Tensor
from PIL.Image import Image

from vellum.utils.configuration import config


embeddings_model = ColQwen2_5.from_pretrained(
    config.embeddings_model,
    torch_dtype=bfloat16,
    device_map='cuda:0',
    attn_implementation='flash_attention_2',
).eval()


embeddings_processor = ColQwen2_5_Processor.from_pretrained(
    config.embeddings_model,
)


def embed_queries(*queries: str, batch_size: int = 8) -> Tensor:
    """
    Embeds a list of text queries into a tensor of multi-vector embeddings.
    """
    batch_queries = embeddings_processor \
        .process_queries(queries) \
        .to(embeddings_model.device)

    with torch.no_grad():
        return embeddings_model(**batch_queries)


def embed_images(*images: Image, batch_size: int = 8) -> Tensor:
    """
    Embeds a list of images into a tensor of multi-vector embeddings.
    """
    batch_images = embeddings_processor \
        .process_images(images) \
        .to(embeddings_model.device)

    with torch.no_grad():
        return embeddings_model(**batch_images)
