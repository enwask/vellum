class Config:
    # Placeholder for possible dynamic configuration in the future
    @property
    def embeddings_model(self) -> str:
        return 'nomic-ai/colnomic-embed-multimodal-7b'

    @property
    def embeddings_vector_size(self) -> int:
        return 128

    @property
    def chat_model(self) -> str:
        return 'qwen3:32b'

config = Config()
