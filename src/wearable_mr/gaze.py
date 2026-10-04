from __future__ import annotations

from .types import Detection


def select_detection(detections: list[Detection], x: float, y: float) -> Detection | None:
    """Select the most specific confident box under a pixel-space gaze point."""
    hits = [item for item in detections if item.box.contains(x, y)]
    if not hits:
        return None
    return min(hits, key=lambda item: (item.box.area, -item.confidence))

