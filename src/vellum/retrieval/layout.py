import copy
import os
from pathlib import Path
from pyexpat import model
from re import search
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from layoutparser.elements import Rectangle, TextBlock
from layoutparser.models.detectron2 import Detectron2LayoutModel, catalog
from PIL.Image import Image
import requests

from vellum.utils import config


# https://github.com/pradhanhitesh/LayoutParser-Install?tab=readme-ov-file#11-downloading-detectron2-via-layout-parser  # noqa
def load_model() -> Detectron2LayoutModel:
    config_path = config.layout_parser_config

    config_path_split = config_path.split('/')
    dataset_name = config_path_split[-3]
    model_name = config_path_split[-2]

    # get the URLs from the MODEL_CATALOG and the CONFIG_CATALOG
    # (global variables .../layoutparser/models/detectron2/catalog.py)
    model_url = catalog.MODEL_CATALOG[dataset_name][model_name]
    config_url = catalog.CONFIG_CATALOG[dataset_name][model_name]

    model_path = Path('.vellum') / 'layout_model'
    model_path.mkdir(parents=True, exist_ok=True)

    config_file_path, model_file_path = None, None
    for url in [model_url, config_url]:
        filename = url.split('/')[-1].split('?')[0]
        save_to_path = model_path / filename

        if 'config' in filename:
            config_file_path = copy.deepcopy(save_to_path)
        if 'model_final' in filename:
            model_file_path = copy.deepcopy(save_to_path)

        # skip if file exist in path
        if filename in os.listdir(model_path):
            continue

        # Download file from URL
        r = requests.get(
            url,
            stream=True,
            headers={
                'user-agent': "Wget/1.16 (linux-gnu)",
            },
        )

        with open(save_to_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)

    return Detectron2LayoutModel(
        config_path=str(config_file_path),
        model_path=str(model_file_path),
        extra_config=['MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.8],
        label_map={
            0: 'text',
            1: 'title',
            2: 'list',
            3: 'table',
            4: 'figure'
        },
    )


layout_model = load_model()


class LayoutElement(NamedTuple):
    """
    Describes an element in the identified page layout.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    type: str


def parse_layout(image: Image) -> list[LayoutElement]:
    cv2_image = np.array(image.convert('RGB'))
    cv2_image = cv2_image[:, :, ::-1].copy()  # BGR for OpenCV

    layout = layout_model.detect(cv2_image)
    elements: list[LayoutElement] = []

    for block in layout:
        if TYPE_CHECKING:
            assert isinstance(block, TextBlock)

        rect: Rectangle = block.block  # type: ignore
        type: str = block.type # type: ignore

        elements.append(LayoutElement(
            x1=rect.x_1,
            y1=rect.y_1,
            x2=rect.x_2,
            y2=rect.y_2,
            type=type,
        ))

    return elements
