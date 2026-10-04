from __future__ import annotations

import argparse
import json
import time

import cv2

from .detection import YoloDetector
from .gaze import select_detection
from .knowledge import DomainKnowledge
from .runtime import AgentRuntime


def _draw(frame, detections, selected):
    for item in detections:
        color = (0, 220, 255) if item is selected else (80, 180, 80)
        box = item.box
        cv2.rectangle(frame, (int(box.left), int(box.top)), (int(box.right), int(box.bottom)), color, 2)
        cv2.putText(frame, f"{item.label} {item.confidence:.2f}", (int(box.left), int(box.top) - 6), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)


def image_command(args) -> None:
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")
    detections = YoloDetector(args.weights, args.confidence, args.device).detect(frame)
    print(json.dumps([{"label": d.label, "confidence": d.confidence, "box": vars(d.box)} for d in detections], indent=2))


def camera_command(args) -> None:
    detector = YoloDetector(args.weights, args.confidence, args.device)
    runtime = AgentRuntime(DomainKnowledge.load(args.knowledge), dwell_seconds=args.dwell)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    print("Keep the crosshair on an object for four seconds. Press A to ask, Q to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        detections = detector.detect(frame)
        h, w = frame.shape[:2]
        selected = select_detection(detections, w / 2, h / 2)
        events = runtime.observe(detections, w / 2, h / 2, time.monotonic())
        _draw(frame, detections, selected)
        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(frame, runtime.interaction.state.value, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        if events:
            print(", ".join(events))
        cv2.imshow("Wearable MR agent - portable camera adapter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            try:
                reply = runtime.ask(input("Question: "))
                print(json.dumps({"text": reply.text, "events": [vars(e) for e in reply.events]}, indent=2))
            except RuntimeError as exc:
                print(exc)
    camera.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    for name in ("image", "camera"):
        command = sub.add_parser(name)
        command.add_argument("--weights", required=True, help="YOLO weights name/path, e.g. yolo11n.pt or runs/.../best.pt")
        command.add_argument("--confidence", type=float, default=.35)
        command.add_argument("--device")
    image = sub.choices["image"]
    image.add_argument("image")
    image.set_defaults(func=image_command)
    camera = sub.choices["camera"]
    camera.add_argument("--knowledge", required=True)
    camera.add_argument("--camera", type=int, default=0)
    camera.add_argument("--dwell", type=float, default=4.0)
    camera.set_defaults(func=camera_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

