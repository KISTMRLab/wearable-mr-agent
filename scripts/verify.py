"""Drive the paper's interaction loop offline and write the transcript to outputs/verify/.

Character dwell -> proactive greeting -> voice command with (simulated) object
recognition that loads the room's anchors -> follow-up without re-gazing ->
anchored object (recognition skipped) -> general conversation. Each reply
carries (text, sentiment class, sentiment level), the renderer expression and
the timed animation list from the phrase/word table.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from wearable_mr.factory import build_runtime, default_args
from wearable_mr.gaze import DetectionGazeTracker
from wearable_mr.types import BoundingBox, Detection, Gaze

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def check_yolo(output: Path, frame: np.ndarray) -> None:
    from ultralytics import YOLO
    from wearable_mr.detection import YoloDetector
    from wearable_mr.recognition import DetectorRecognizer, RecognitionError

    # Architecture config initializes random weights locally; it does not download pretrained assets.
    model = YOLO("yolo11n.yaml")
    weights = output / "random-yolo11n.pt"
    model.save(weights)
    detector = YoloDetector(weights)
    detections = detector.detect(frame)
    assert isinstance(detections, list)
    try:
        DetectorRecognizer(detector).recognize(None, frame)
    except RecognitionError:
        pass  # random weights rarely place a box on the gaze point; the adapter path is what is checked
    print(f"optional YOLO adapter passed: {len(detections)} random-weight detection(s) -> {weights}")


def summary(reply: dict) -> dict:
    return {"query": reply["query"], "text": reply["text"], "object": reply["object"], "query_type": reply["query_type"],
            "recognition": reply["recognition"], "sentiment": [reply["sentiment"]["class"], reply["sentiment"]["level"]],
            "expression": reply["expression"],
            "animations": [[e["start_s"], e["trigger"], e["clip_id"] if e["kind"] == "clip" else e["gesture"]]
                           for e in reply["animation"]["entries"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-yolo", action="store_true", help="also initialize and run the optional Ultralytics adapter")
    args = parser.parse_args()
    output = (ROOT / "outputs" / "verify").resolve()
    output.mkdir(parents=True, exist_ok=True)
    clock = Clock()
    runtime = build_runtime(default_args(recognition_latency=0.0), clock=clock)
    transcript = []

    def advance(seconds, gaze, speaking=False):
        clock.t += seconds
        return runtime.tick(gaze, speaking)

    def say(text, gaze):
        submitted = runtime.submit(text, gaze)
        assert submitted["accepted"], submitted
        reply = runtime.job(submitted["job"])["reply"]
        transcript.append({"thinking": submitted["thinking"], **summary(reply)})
        runtime.response_finished()
        return reply

    character = Gaze("character")
    advance(0, character)
    assert advance(runtime.config.dwell_seconds, character)["state"] == "listening"
    greeting = advance(runtime.config.greeting_after_seconds, character)
    assert greeting["events"] == ["proactive_greeting"], greeting
    transcript.append({"proactive": True, **summary({**greeting["speak"][0], "query": None})})
    runtime.response_finished()
    rose = say("What is this?", Gaze("object", "r1-rose"))
    assert rose["object"] == "rose" and rose["query_type"] == "A"
    care = say("How should I care for it?", character)
    assert care["object"] == "rose" and care["recognition"] is None
    tulip = say("Tell me about this", Gaze("object", "r1-tulip"))
    assert tulip["recognition"]["skipped"] and tulip["object"] == "tulip"
    chat = say("Who are you?", character)
    assert chat["query_type"] == "B"
    lily = say("what's this", Gaze("object", "r1-lily"))
    pets = say("Is it safe for my cat?", character)
    assert pets["sentiment"]["class"] == "Fear", pets["sentiment"]
    ended = advance(runtime.config.lookaway_grace_seconds + 0.1, Gaze("none"))
    ended = advance(runtime.config.lookaway_grace_seconds + 0.1, Gaze("none"))
    assert "conversation_ended" in ended["events"], ended

    # Camera adapter contract: a gaze point on a detection becomes an object gaze target.
    frame = np.full((110, 130, 3), 245, np.uint8)
    detections = [Detection("potted plant", .91, BoundingBox(20, 15, 100, 95), "demo-1")]
    camera_gaze = DetectionGazeTracker().update(detections, 60, 55)
    assert camera_gaze == Gaze("object", "potted plant:demo-1")
    cv2.rectangle(frame, (20, 15), (100, 95), (20, 130, 20), 2)
    cv2.drawMarker(frame, (60, 55), (0, 0, 0), cv2.MARKER_CROSS, 12, 1)
    cv2.imwrite(str(output / "annotated-detection.png"), frame)
    library = runtime.builder.library is not None
    payload = {"recognizer": "simulated room props (no detector)", "bank_prepared": library,
               "chatbot": runtime.chatbot.source, "transcript": transcript}
    (output / "response.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.with_yolo:
        check_yolo(output, frame)
    print(f"verification passed: {len(transcript)} agent turns (greeting, recognition + anchor load, follow-up, "
          f"anchor skip, general chat, sentiment); BEAT clips {'resolved' if library else 'not prepared: procedural stand-ins'} "
          f"-> {output / 'response.json'}")


if __name__ == "__main__":
    main()
