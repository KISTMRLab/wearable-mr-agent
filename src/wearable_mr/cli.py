"""Command-line adapters for the wearable MR agent core."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .factory import add_service_arguments, build_runtime

ROOT = Path(__file__).resolve().parents[2]


def image_command(args) -> None:
    import cv2

    from .detection import YoloDetector

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")
    detections = YoloDetector(args.weights, args.confidence, args.device).detect(frame)
    print(json.dumps([{"label": d.label, "confidence": d.confidence, "box": vars(d.box)} for d in detections], indent=2))


def _summary(reply: dict) -> dict:
    return {"text": reply["text"], "object": reply["object"], "command": reply["command"],
            "recognition": reply["recognition"], "sentiment": {k: reply["sentiment"][k] for k in ("class", "level", "source")},
            "expression": reply["expression"], "chatbot": reply["chatbot"],
            "animations": [f"{e['start_s']:.2f}s {e['trigger']!r} -> {e['clip_id'] or e['gesture']} ({e['match']})"
                           for e in reply["animation"]["entries"]]}


def camera_command(args) -> None:
    import time

    import cv2

    from .detection import YoloDetector
    from .gaze import DetectionGazeTracker
    from .types import Gaze

    detector = YoloDetector(args.weights, args.confidence, args.device)
    runtime = build_runtime(args, detector=detector)
    tracker = DetectionGazeTracker()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    print("The crosshair is the gaze proxy. Press A and say/type a voice command such as 'what is this' "
          "while it rests on an object, then ask follow-up questions. Q quits.")
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        detections = detector.detect(frame)
        h, w = frame.shape[:2]
        gaze = tracker.update(detections, w / 2, h / 2)
        for item in detections:
            color = (0, 220, 255) if item.label == tracker.label else (80, 180, 80)
            box = item.box
            cv2.rectangle(frame, (int(box.left), int(box.top)), (int(box.right), int(box.bottom)), color, 2)
            cv2.putText(frame, f"{item.label} {item.confidence:.2f}", (int(box.left), int(box.top) - 6), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(frame, f"{runtime.state.value} | object: {runtime.conversation_object or '-'}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        cv2.imshow("Wearable MR agent - camera adapter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            text = input("Utterance: ").strip()
            if text:
                started = time.perf_counter()
                reply = runtime.ask(text, gaze if gaze.kind == "object" else Gaze(), frame)
                print(json.dumps({**_summary(reply), "seconds": round(time.perf_counter() - started, 2)}, indent=2))
    camera.release()
    cv2.destroyAllWindows()


def chat_command(args) -> None:
    """Text conversation with the chatbot + sentiment engine + animation builder (no camera)."""
    from .types import Gaze

    runtime = build_runtime(args)
    lines = [args.say] if args.say else None
    print("Type a query (empty line quits). Prefix with '@label ' to simulate looking at an object, e.g. '@rose what is this'.")
    while True:
        line = lines.pop(0) if lines else (None if lines is not None else input("> "))
        if not line:
            break
        gaze = Gaze()
        if line.startswith("@"):
            label, _, line = line[1:].partition(" ")
            runtime.recognizer = _FixedRecognizer(label)
            gaze = Gaze("object", f"cli:{label}")
        print(json.dumps(_summary(runtime.ask(line, gaze)), indent=2))


class _FixedRecognizer:
    def __init__(self, label: str):
        self.label = label

    def recognize(self, target_id, frame):
        from .recognition import Recognition

        return Recognition(self.label, 1.0, "cli")


def animate_command(args) -> None:
    from .animation import AnimationBuilder, AnimationTable, ClipLibrary

    library = ClipLibrary.load(args.bank) if args.bank else None
    builder = AnimationBuilder(AnimationTable.load(args.table), library)
    for entry in builder.build(args.text):
        print(f"{entry.start_s:6.2f}-{entry.end_s:6.2f}s  {entry.match:6s} {entry.trigger!r:24s} -> "
              f"{entry.kind}:{entry.clip_id or entry.gesture}")


def sentiment_command(args) -> None:
    from .factory import sentiment_engine

    print(json.dumps(sentiment_engine(args).analyze(args.text).to_dict(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    image = sub.add_parser("image", help="Run YOLO on one image")
    image.add_argument("image")
    camera = sub.add_parser("camera", help="Camera loop with screen-centre gaze, voice commands and follow-ups")
    camera.add_argument("--camera", type=int, default=0)
    for command in (image, camera):
        command.add_argument("--weights", required=True, help="YOLO weights name/path, e.g. yolo11n.pt or runs/.../best.pt")
        command.add_argument("--confidence", type=float, default=.35)
        command.add_argument("--device")
    image.set_defaults(func=image_command)
    camera.set_defaults(func=camera_command)
    chat = sub.add_parser("chat", help="Text conversation through the chatbot, sentiment engine and animation builder")
    chat.add_argument("--say", help="Answer one utterance and exit")
    chat.set_defaults(func=chat_command)
    for command in (camera, chat):
        add_service_arguments(command)
    animate = sub.add_parser("animate", help="Print the timed animation list for a reply")
    animate.add_argument("text")
    animate.add_argument("--table", default=str(ROOT / "demo" / "animation-table.json"))
    animate.add_argument("--bank", default=str(ROOT / "outputs" / "beat-library" / "bank.json"))
    animate.set_defaults(func=animate_command)
    sentiment = sub.add_parser("sentiment", help="Classify reply text into Joy/Angry/Sad/Fear x High/Medium/Low")
    sentiment.add_argument("text")
    add_service_arguments(sentiment, chat=False)
    sentiment.set_defaults(func=sentiment_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
