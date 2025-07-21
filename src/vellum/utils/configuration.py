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
        # return 'qwen3:32b'
        return 'llama4:latest'

    @property
    def layout_parser_model(self) -> str:
        return 'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x'

    @property
    def layout_parser_label_map(self) -> dict[int, str]:
        return {
            0: 'text',
            1: 'title',
            2: 'list',
            3: 'table',
            4: 'figure',
        }

    @property
    def layout_parser_config(self) -> list[str | float]:
        return [
            'MODEL.ROI_HEADS.SCORE_THRESH_TEST', .75,
        ]

    @property
    def data_version(self) -> int:
        # Versioning for parsed document metadata
        return 7

config = Config()
