"""Local browser adapter for the portable detection and interaction core."""
from __future__ import annotations

import argparse
import base64
import json
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .detection import YoloDetector
from .knowledge import DomainKnowledge
from .runtime import AgentRuntime
from .speech_backend import SpeechBackend, speech_route
from .types import BoundingBox, Detection


def app(knowledge: DomainKnowledge, detector=None):
    runtime = AgentRuntime(knowledge)
    speech = SpeechBackend()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/speech":
                body = json.dumps(speech.status()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path in {"/static/avatar.js", "/static/speech.js", "/static/voice-input.js", "/static/vendor/three.module.js"}:
                body = (Path(__file__).resolve().parents[2] / self.path.lstrip("/")).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/":
                self.send_error(404)
                return
            page = (Path(__file__).resolve().parents[2] / "demo" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self):
            if speech_route(self, speech):
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/detect":
                    if detector is None:
                        raise ValueError("No YOLO weights configured. Use procedural example explicitly or restart with --weights.")
                    raw = base64.b64decode(payload["image"].split(",", 1)[-1])
                    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("Invalid image")
                    detections = detector.detect(frame)
                    answer = {"source": "YOLO inference", "width": frame.shape[1], "height": frame.shape[0],
                              "detections": [{"label": d.label, "confidence": d.confidence, "box": asdict(d.box), "track_id": d.track_id} for d in detections]}
                elif self.path == "/api/observe":
                    detections = [Detection(item["label"], float(item["confidence"]), BoundingBox(**item["box"]), item.get("track_id")) for item in payload["detections"]]
                    events = runtime.observe(detections, float(payload["x"]), float(payload["y"]), time.monotonic())
                    answer = {"state": runtime.interaction.state.value, "events": events, "object": runtime.focused_object,
                              "target": runtime.interaction.target}
                elif self.path == "/api/ask":
                    reply = runtime.ask(str(payload["question"]))
                    answer = {"text": reply.text, "object": reply.object_label, "events": [asdict(e) for e in reply.events]}
                else:
                    self.send_error(404)
                    return
                status = 200
            except (ValueError, KeyError, RuntimeError, TypeError) as exc:
                answer, status = {"error": str(exc)}, 400
            content = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge", default=str(Path(__file__).resolve().parents[2] / "demo" / "knowledge.json"))
    parser.add_argument("--weights", help="Optional local or public Ultralytics detection checkpoint")
    parser.add_argument("--port", type=int, default=8761)
    args = parser.parse_args()
    detector = YoloDetector(args.weights) if args.weights else None
    print(f"Wearable demo at http://127.0.0.1:{args.port}; detector={'YOLO' if detector else 'not configured'}")
    ThreadingHTTPServer(("127.0.0.1", args.port), app(DomainKnowledge.load(args.knowledge), detector)).serve_forever()


if __name__ == "__main__":
    main()
