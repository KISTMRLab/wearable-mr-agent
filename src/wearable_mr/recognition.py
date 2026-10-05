"""Vision manager: recognise the object the user is gazing at (paper Section 2.1.2).

Any recogniser that can answer a request works: a local YOLO detector on a
camera frame, a remote vision endpoint, or (in the desktop web demo) a
simulated recogniser for scene props whose labels come from the authored room
description.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .clients import ServiceError, post_json
from .gaze import select_detection


@dataclass(frozen=True)
class Recognition:
    label: str
    confidence: float
    source: str


class RecognitionError(RuntimeError):
    pass


class Recognizer(Protocol):
    def recognize(self, target_id: str | None, frame: Any | None) -> Recognition: ...


class SimulatedRecognizer:
    """Scene-prop recogniser for simulated rooms; ``latency_seconds`` stands in for a cloud round trip."""

    def __init__(self, labels: dict[str, str], latency_seconds: float = 0.0, sleep: Callable[[float], None] = time.sleep):
        self.labels, self.latency_seconds, self.sleep = dict(labels), latency_seconds, sleep
        self.calls = 0

    def recognize(self, target_id: str | None, frame: Any | None = None) -> Recognition:
        self.calls += 1
        if self.latency_seconds:
            self.sleep(self.latency_seconds)
        if target_id not in self.labels:
            raise RecognitionError("Nothing recognisable is under your gaze.")
        return Recognition(self.labels[target_id], 1.0, "simulated")


class DetectorRecognizer:
    """Run a detector (e.g. :class:`YoloDetector`) and take the box under the gaze point (frame centre by default)."""

    def __init__(self, detector: Any, gaze_point: tuple[float, float] | None = None):
        self.detector, self.gaze_point = detector, gaze_point

    def recognize(self, target_id: str | None, frame: Any | None) -> Recognition:
        if frame is None:
            raise RecognitionError("No camera frame was supplied.")
        detections = self.detector.detect(frame)
        height, width = frame.shape[:2]
        x, y = self.gaze_point or (width / 2, height / 2)
        hit = select_detection(detections, x, y)
        if hit is None:
            raise RecognitionError("I can't see an object at the centre of your view." if detections
                                   else "I can't see a recognisable object right now.")
        return Recognition(hit.label, hit.confidence, "yolo")


class HttpRecognizer:
    """POST a JPEG frame ``{"image": base64}`` to a vision endpoint answering ``{"label","confidence"}``."""

    def __init__(self, endpoint: str, timeout: float = 15.0):
        self.endpoint, self.timeout = endpoint, timeout

    def recognize(self, target_id: str | None, frame: Any | None) -> Recognition:
        if frame is None:
            raise RecognitionError("No camera frame was supplied.")
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RecognitionError("Could not encode the camera frame.")
        try:
            value = post_json(self.endpoint, {"image": base64.b64encode(encoded.tobytes()).decode()}, timeout=self.timeout)
        except ServiceError as exc:
            raise RecognitionError(str(exc)) from exc
        if not isinstance(value, dict) or not value.get("label"):
            raise RecognitionError("The vision endpoint returned no label.")
        return Recognition(str(value["label"]), float(value.get("confidence", 0.0)), "http")


class RoutingRecognizer:
    """Camera frames go to ``camera``; gazed scene props go to ``scene``."""

    def __init__(self, scene: Recognizer | None = None, camera: Recognizer | None = None):
        self.scene, self.camera = scene, camera

    def recognize(self, target_id: str | None, frame: Any | None) -> Recognition:
        if frame is not None:
            if self.camera is None:
                raise RecognitionError("No object detector is configured; restart the demo with --weights to recognise camera frames.")
            return self.camera.recognize(target_id, frame)
        if self.scene is None:
            raise RecognitionError("No recogniser is configured for scene objects.")
        return self.scene.recognize(target_id, None)
