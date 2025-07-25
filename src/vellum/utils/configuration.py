from functools import cached_property
from typing import NamedTuple


class LayoutParser(NamedTuple):
    """
    Describes a layout parser configuration. Multiple configurations mayb
    be specified to combine different models or settings.
    """
    model: str
    label_map: dict[int, str | None]
    config: list[str | float]

    def __str__(self) -> str:
        return f"LayoutParser(model={self.model}, labels={self.label_map})"


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
        return 'gpt-4o-mini'

    @cached_property
    def layout_parsers(self) -> list[LayoutParser]:
        """
        Returns a list of layout parser configurations.
        """
        return [
            # PubLayNet text + figures (low threshold)
            LayoutParser(
                model='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x',
                label_map={
                    0: None,
                    1: None,
                    2: None,
                    3: 'table',
                    4: 'figure',
                },
                config=[
                    'MODEL.ROI_HEADS.SCORE_THRESH_TEST', .2,
                ],
            ),

            # PubLayNet lists (higher threshold)
            LayoutParser(
                model='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x',
                label_map={
                    0: None,
                    1: None,
                    2: 'list',
                    3: None,
                    4: None,
                },
                config=[
                    'MODEL.ROI_HEADS.SCORE_THRESH_TEST', .6,
                ],
            ),

            # MFD (math formulas)
            LayoutParser(
                model='lp://MFD/faster_rcnn_R_50_FPN_3x',
                label_map={
                    1: 'math',
                },
                config=[
                    'MODEL.ROI_HEADS.SCORE_THRESH_TEST', .8,
                ],
            )
        ]

    @property
    def data_version(self) -> int:
        # Versioning for parsed document metadata
        return 56

config = Config()
