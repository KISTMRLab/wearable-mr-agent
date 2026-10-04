"""Drive detection selection, dwell interaction, and grounded response offline."""
import json
import argparse
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from wearable_mr.gaze import select_detection
from wearable_mr.interaction import InteractionState, InteractionStateMachine
from wearable_mr.knowledge import DomainKnowledge
from wearable_mr.types import BoundingBox, Detection


def check_yolo(output: Path, frame: np.ndarray) -> None:
    from ultralytics import YOLO
    from wearable_mr.detection import YoloDetector

    # Architecture config initializes random weights locally; it does not download pretrained assets.
    model = YOLO("yolo11n.yaml")
    weights = output / "random-yolo11n.pt"
    model.save(weights)
    detections = YoloDetector(weights).detect(frame)
    assert isinstance(detections, list)
    print(f"optional YOLO adapter passed: {len(detections)} random-weight detection(s) -> {weights}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-yolo", action="store_true", help="also initialize and run the optional Ultralytics adapter")
    args = parser.parse_args()
    output = Path("outputs/smoke").resolve(); output.mkdir(parents=True, exist_ok=True)
    frame = np.full((110, 130, 3), 245, np.uint8)
    detections = [Detection("flower", .91, BoundingBox(20, 15, 100, 95), "demo-1")]
    selected = select_detection(detections, 60, 55)
    machine = InteractionStateMachine(dwell_seconds=4.0)
    assert machine.update_gaze(selected.label, 0.0) == ["dwell_started"]
    assert machine.update_gaze(selected.label, 4.1) == ["listening_started"]
    machine.submit_utterance()
    knowledge = DomainKnowledge({"flower": {"overview": "A demo flower.",
                                             "topics": {"care": "Give it indirect light."},
                                             "emotion": "joy"}})
    reply = knowledge.answer("How should I care for it?", selected.label)
    machine.begin_response(); machine.finish_response()
    assert machine.state == InteractionState.LISTENING and len(reply.events) == 4
    cv2.rectangle(frame, (20, 15), (100, 95), (20, 130, 20), 2)
    cv2.putText(frame, selected.label, (22, 13), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 80, 20), 1)
    cv2.imwrite(str(output / "annotated-detection.png"), frame)
    payload = {"detector": "procedural contract fixture", "target": selected.label,
               "reply": reply.text, "events": [asdict(event) for event in reply.events]}
    (output / "response.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.with_yolo:
        check_yolo(output, frame)
    print(f"smoke passed: target={selected.label}; response and annotated frame -> {output}")


if __name__ == "__main__":
    main()
