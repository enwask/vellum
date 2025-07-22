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

from vellum.utils.configuration import config, LayoutParser


# https://github.com/pradhanhitesh/LayoutParser-Install?tab=readme-ov-file#11-downloading-detectron2-via-layout-parser  # noqa
def load_model(parser: LayoutParser) -> Detectron2LayoutModel:
    config_path = f'{parser.model}/config'

    config_path_split = config_path.split('/')
    dataset_name = config_path_split[-3]
    model_name = config_path_split[-2]

    # get the URLs from the MODEL_CATALOG and the CONFIG_CATALOG
    # (global variables .../layoutparser/models/detectron2/catalog.py)
    model_url = catalog.MODEL_CATALOG[dataset_name][model_name]
    config_url = catalog.CONFIG_CATALOG[dataset_name][model_name]

    model_path = Path('.vellum') / 'layout_models' / dataset_name / model_name
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
        extra_config=parser.config,
        label_map=parser.label_map,
    )


# Load all layout models defined in the configuration
layout_models = [
    load_model(parser)
    for parser in config.layout_parsers
]


class LayoutElement(NamedTuple):
    """
    Describes an element in the identified page layout.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    type: str

    def width(self) -> float:
        """
        Returns the width of the element.
        """
        return self.x2 - self.x1

    def height(self) -> float:
        """
        Returns the height of the element.
        """
        return self.y2 - self.y1

    def size(self) -> float:
        """
        Returns the area of the element.
        """
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def contains(self, other: 'LayoutElement', epsilon: float) -> bool:
        """
        Returns True if the other element is within this element's bounding
        box, padded by the specified amount.
        """
        return (self.x1 - epsilon <= other.x1 <= self.x2 + epsilon and
                self.y1 - epsilon <= other.y1 <= self.y2 + epsilon and
                self.x1 - epsilon <= other.x2 <= self.x2 + epsilon and
                self.y1 - epsilon <= other.y2 <= self.y2 + epsilon)

    @classmethod
    def promote_types(cls, *types: str) -> str:
        """
        Returns a type obtained by merging the provided types.
        Largely meaningless.
        """
        if 'math' in types:
            return 'math'

        if 'figure' in types:
            return 'figure'

        if 'list' in types:
            return 'list'

        if 'text' in types:
            return 'text'

        return types[0]

    @classmethod
    def merge(cls, *elements: 'LayoutElement') -> 'LayoutElement':
        """
        Merges multiple elements into a single element that covers the bounding
        box of all provided elements.

        Elements are assumed to be of the same type.
        """

        return cls(
            x1=min(e.x1 for e in elements),
            y1=min(e.y1 for e in elements),
            x2=max(e.x2 for e in elements),
            y2=max(e.y2 for e in elements),
            type=cls.promote_types(*(e.type for e in elements)),
        )


def remove_contained(elements: list[LayoutElement],
                     epsilon: float) -> list[LayoutElement]:
    """
    Removes elements that are completely contained by another element.
    If certain types (e.g. math) are present within the element, its type
    may be promoted to that type.

    Epsilon is relative to the max dimension of the containing element.
    """
    # We can do this pretty naively because the number of elements is tiny
    elements = sorted(elements, key=LayoutElement.size, reverse=True)
    res: list[LayoutElement] = []
    for element in elements:
        abs_epsilon = epsilon * max(element.width(), element.height())
        any_contained = False
        for i, containing in enumerate(res):
            if containing.contains(element, epsilon=abs_epsilon):
                any_contained = True

                # If the containing element is of a different type, promote it
                if containing.type != element.type:
                    res[i] = LayoutElement.merge(containing, element)

                break

        if not any_contained:
            # If the element is not contained, add it to the result
            res.append(element)

    return res


def merge_similar(elements: list[LayoutElement],
                   epsilon: float) -> list[LayoutElement]:
    """
    Merges elements that cover similar areas, within epsilon. If two elements
    are considered the same, they are replaced with a full cover of both.

    Epsilon is relative to the max dimension of any element being considered.

    Only merges compatible types; see LayoutElement._merge_types.
    """
    def _similar(a: LayoutElement, b: LayoutElement) -> bool:
        abs_epsilon = epsilon * max(a.width(), a.height(),
                                    b.width(), b.height())
        dist = abs(a.x1 - b.x1) + abs(a.y1 - b.y1) + abs(a.x2 - b.x2) + abs(a.y2 - b.y2)
        return dist < abs_epsilon

    # Again, we do this naively because the number of elements is small
    i = 0
    while i < len(elements):
        a = elements[i]
        j = i + 1

        sim: list[LayoutElement] = []
        while j < len(elements):
            b = elements[j]
            if _similar(a, b):
                # Merge the two elements
                sim.append(elements.pop(j))
            else:
                j += 1
        if sim:
            elements[i] = LayoutElement.merge(a, *sim)
        i += 1

    return elements


def parse_layout(image: Image,
                 x_padding: float = 20,
                 y_padding: float = 20,
                 merge_epsilon: float = .20,
                 containment_epsilon: float = .10) -> list[LayoutElement]:
    cv2_image = np.array(image.convert('RGB'))
    cv2_image = cv2_image[:, :, ::-1].copy()  # BGR for OpenCV

    # Combine layouts from all models
    layouts = [model.detect(cv2_image) for model in layout_models]
    elements: list[LayoutElement] = []

    for layout in layouts:
        for block in layout:
            if TYPE_CHECKING:
                assert isinstance(block, TextBlock)

            rect: Rectangle = block.block  # type: ignore
            type: str | None = block.type

            if type is None:
                # Key omitted from the label map (or mapped to None); skip it
                continue

            # We pad the coordinates a little bit, clamping to the image size and rounding
            x1 = int(max(0, rect.x_1 - x_padding))
            y1 = int(max(0, rect.y_1 - y_padding))
            x2 = int(min(image.width, rect.x_2 + x_padding))
            y2 = int(min(image.height, rect.y_2 + y_padding))
            elements.append(LayoutElement(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                type=type,
            ))

    # Merge similar elements and remove fully contained elements
    elements = merge_similar(elements, epsilon=merge_epsilon)
    elements = remove_contained(elements, epsilon=containment_epsilon)
    return elements
