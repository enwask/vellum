from torch import bfloat16, Tensor
from transformers import AutoModel, PreTrainedModel
from PIL.Image import Image

from vellum.utils.configuration import config


embeddings_model: PreTrainedModel = AutoModel.from_pretrained(
    config.embeddings_model,
    device_map='cuda',
    trust_remote_code=True,
    torch_dtype=bfloat16,
    attn_implementation='flash_attention_2',
).eval()


def embed_queries(*queries: str, batch_size: int = 8) -> Tensor:
    """
    Embeds a list of text queries into a tensor of multi-vector embeddings.
    """
    return embeddings_model \
        .forward_queries(queries, batch_size=batch_size)  # type: ignore


def embed_images(*images: Image, batch_size: int = 8) -> Tensor:
    """
    Embeds a list of images into a tensor of multi-vector embeddings.
    """
    return embeddings_model \
        .forward_passages(images, batch_size=batch_size)  # type: ignore
