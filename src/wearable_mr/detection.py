from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .types import BoundingBox, Detection


class YoloDetector:
    """Thin adapter around current Ultralytics YOLO detection models."""

    def __init__(self, weights: str | Path, confidence: float = 0.35, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the optional detector with: pip install -e .[yolo]") from exc
        value = str(weights)
        if not Path(value).exists() and not re.fullmatch(r"yolo\d+[a-z-]*\.pt", value, re.IGNORECASE):
            raise FileNotFoundError(f"Local weights do not exist: {value}")
        self.model = YOLO(value)
        self.confidence = confidence
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame, conf=self.confidence, device=self.device, verbose=False)[0]
        names = result.names
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for index, box in enumerate(result.boxes):
            left, top, right, bottom = (float(value) for value in box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            track = None if box.id is None else str(int(box.id[0]))
            detections.append(
                Detection(str(names[class_id]), float(box.conf[0]), BoundingBox(left, top, right, bottom), track)
            )
        return detections
