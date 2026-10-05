from __future__ import annotations

from .types import Detection, Gaze


def select_detection(detections: list[Detection], x: float, y: float) -> Detection | None:
    """Select the most specific confident box under a pixel-space gaze point."""
    hits = [item for item in detections if item.box.contains(x, y)]
    if not hits:
        return None
    return min(hits, key=lambda item: (item.box.area, -item.confidence))


def _iou(first: Detection, second: Detection) -> float:
    left, top = max(first.box.left, second.box.left), max(first.box.top, second.box.top)
    right, bottom = min(first.box.right, second.box.right), min(first.box.bottom, second.box.bottom)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.box.area + second.box.area - intersection
    return intersection / union if union else 0.0


class DetectionGazeTracker:
    """Turn per-frame detections and a gaze point into a stable object gaze target.

    Untracked detections keep the same instance id across detector jitter when
    the label matches and the boxes overlap (IoU >= ``iou``).
    """

    def __init__(self, iou: float = .45):
        self.iou = iou
        self._last: Detection | None = None
        self._instance: str | None = None
        self._counter = 0
        self.label: str | None = None

    def update(self, detections: list[Detection], x: float, y: float) -> Gaze:
        hit = select_detection(detections, x, y)
        if hit is None:
            self.label = None
            return Gaze("none")
        if hit.track_id is not None:
            target = f"{hit.label}:{hit.track_id}"
        elif self._last is not None and hit.label == self._last.label and _iou(hit, self._last) >= self.iou:
            target = self._instance
        else:
            self._counter += 1
            target = f"{hit.label}:local-{self._counter}"
        self._last, self._instance, self.label = hit, target, hit.label
        return Gaze("object", target)
