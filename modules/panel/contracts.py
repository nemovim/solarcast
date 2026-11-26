"""Input and output contract for one RGB panel detection image."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PanelDetections:
    """Instance masks and boxes refer to the input image pixel grid."""
    masks: Any
    scores: Any
    labels: Any
    boxes_xyxy: Any
    model_version: str
