"""Local browser demo: the character is the gaze target in simulated rooms, with optional webcam recognition.

The server plays the role of the paper's cloud modules (vision manager,
chatbot with sentiment engine, animation builder, anchor manager). The browser
supplies gaze (mouse ray, screen-centre ray), voice activity, utterances
(browser speech recognition or typed text) and optional webcam frames.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from .avatar_http import serve_avatar_asset
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from beat_runtime import serve_beat

from .animation import ClipLibrary
from .factory import add_service_arguments, build_runtime, default_args, load_scenario
from .knowledge import DomainKnowledge
from .speech_backend import SpeechBackend, speech_route
from .types import Gaze

ROOT = Path(__file__).resolve().parents[2]
PAGES = {"/": ROOT / "demo" / "index.html", "/static/wearable-app.js": ROOT / "static" / "wearable-app.js",
         "/static/voice-input.js": ROOT / "static" / "voice-input.js"}
MAX_BODY = 8 * 1024 * 1024


def _decode_frame(data: str):
    import cv2
    import numpy as np

    raw = base64.b64decode(data.split(",", 1)[-1])
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image")
    return frame


def app(knowledge: DomainKnowledge | None = None, detector=None, args: argparse.Namespace | None = None, runtime=None):
    """Build the request handler. ``knowledge`` overrides the knowledge file named in ``args``."""
    args = args or default_args()
    if runtime is None:
        runtime = build_runtime(args, detector=detector, executor=ThreadPoolExecutor(max_workers=2))
        if knowledge is not None:  # tests and embedders: replace the offline knowledge chatbot
            runtime.chatbot.client, runtime.chatbot.fallback = knowledge, None
    scenario = load_scenario(getattr(args, "scenario", None))
    speech = SpeechBackend()
    bank_path = Path(getattr(args, "bank", "") or "")
    bank_state = {"mtime": bank_path.stat().st_mtime if bank_path.is_file() else None}

    def refresh_library() -> None:
        """Pick up a bank prepared after the server started (scripts/prepare_beat_demo.py)."""
        if runtime.builder is None or not bank_path.name:
            return
        mtime = bank_path.stat().st_mtime if bank_path.is_file() else None
        if mtime != bank_state["mtime"]:
            bank_state["mtime"] = mtime
            runtime.builder.library = ClipLibrary.load(bank_path) if mtime else None

    def config_payload() -> dict:
        cfg = runtime.config
        return {"agent": {**asdict(cfg), "voice_commands": list(cfg.voice_commands), "fillers": list(cfg.fillers)},
                "characters": scenario.get("characters", {}), "rooms": scenario.get("rooms", []),
                "detector": detector is not None, "chatbot": runtime.chatbot.source,
                "library_ready": bool(runtime.builder and runtime.builder.library), **runtime.snapshot()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console readable while the page polls
            pass

        def _send(self, status: int, payload, kind: str = "application/json") -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if serve_beat(self, ROOT, "wearable"): return
            if serve_avatar_asset(self, Path(__file__).resolve().parents[2] / "static"): return
            parsed = urlsplit(self.path)
            if parsed.path == "/api/speech":
                return self._send(200, speech.status())
            if parsed.path == "/api/scenario":
                refresh_library()
                return self._send(200, config_payload())
            if parsed.path == "/api/job":
                try:
                    return self._send(200, runtime.job(parse_qs(parsed.query).get("id", [""])[0]))
                except KeyError as exc:
                    return self._send(404, {"error": str(exc)})
            if parsed.path in {"/static/vendor/three.module.js"}:
                return self._send(200, (ROOT / parsed.path.lstrip("/")).read_bytes(), "text/javascript")
            page = PAGES.get(parsed.path)
            if page is None or not page.is_file():
                self.send_error(404)
                return
            self._send(200, page.read_bytes(), "text/html; charset=utf-8" if page.suffix == ".html" else "text/javascript")

        def do_POST(self):
            if serve_beat(self, ROOT, "wearable"): return
            if speech_route(self, speech):
                return
            path = urlsplit(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= MAX_BODY:
                    raise ValueError("Request body too large")
                payload = json.loads(self.rfile.read(length) or b"{}")
                if path == "/api/observe":
                    answer = runtime.tick(Gaze.from_dict(payload.get("gaze")), bool(payload.get("speaking")))
                elif path == "/api/utterance":
                    refresh_library()
                    frame = _decode_frame(payload["image"]) if payload.get("image") else None
                    answer = runtime.submit(str(payload.get("text", "")), Gaze.from_dict(payload.get("gaze")), frame)
                elif path == "/api/response-finished":
                    answer = runtime.response_finished()
                elif path == "/api/reset":
                    answer = runtime.reset()
                elif path == "/api/ask":
                    refresh_library()
                    answer = runtime.ask(str(payload.get("question", payload.get("text", ""))), Gaze.from_dict(payload.get("gaze")))
                elif path == "/api/animation-plan":
                    refresh_library()
                    if runtime.builder is None:
                        raise ValueError("No animation table configured")
                    text = str(payload.get("text", ""))
                    if not text.strip() or len(text) > 2000:
                        raise ValueError("Supply 1-2000 characters of text")
                    answer = runtime.builder.plan(text)
                elif path == "/api/sentiment":
                    answer = runtime.chatbot.sentiment.analyze(str(payload.get("text", ""))).to_dict()
                elif path == "/api/detect":
                    if detector is None:
                        raise ValueError("No YOLO weights configured. Restart with --weights (for example yolo11n.pt) to detect objects in camera frames.")
                    frame = _decode_frame(payload["image"])
                    answer = {"source": "YOLO inference", "width": frame.shape[1], "height": frame.shape[0],
                              "detections": [{"label": d.label, "confidence": d.confidence, "box": asdict(d.box), "track_id": d.track_id}
                                             for d in detector.detect(frame)]}
                else:
                    self.send_error(404)
                    return
                status = 200
            except (ValueError, KeyError, RuntimeError, TypeError) as exc:
                answer, status = {"error": str(exc)}, 400
            self._send(status, answer)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_service_arguments(parser)
    parser.add_argument("--weights", help="Optional local or public Ultralytics detection checkpoint for webcam recognition")
    parser.add_argument("--port", type=int, default=8761)
    args = parser.parse_args()
    detector = None
    if args.weights:
        from .detection import YoloDetector

        detector = YoloDetector(args.weights)
    print(f"Wearable demo at http://127.0.0.1:{args.port}; detector={'YOLO' if detector else 'not configured'}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), app(None, detector, args)).serve_forever()


if __name__ == "__main__":
    main()
