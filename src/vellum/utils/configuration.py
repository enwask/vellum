class Config:
    # Placeholder for possible dynamic configuration in the future
    @property
    def embeddings_model(self) -> str:
        return 'nvidia/llama-nemoretriever-colembed-3b-v1'

    @property
    def chat_model(self) -> str:
        return 'qwen3:32b'

config = Config()
