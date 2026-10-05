import json
import time
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from wearable_mr.demo import app
from wearable_mr.factory import default_args
from wearable_mr.knowledge import DomainKnowledge


@pytest.fixture()
def server():
    args = default_args(bank="", recognition_latency=0.2, dwell=0.2)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app(None, None, args))
    Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def post(base, route, payload):
    request = Request(base + route, data=json.dumps(payload).encode(), method="POST", headers={"Content-Type": "application/json"})
    try:
        return json.load(urlopen(request))
    except HTTPError as error:
        return {"status": error.code, **json.load(error)}


def test_missing_local_speech_configuration_is_reported_by_routes(monkeypatch):
    monkeypatch.delenv("KOKORO_MODEL_DIR", raising=False)
    monkeypatch.delenv("WHISPER_MODEL_DIR", raising=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), app(DomainKnowledge({})))
    Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert json.load(urlopen(base + "/api/speech"))["models_bundled"] is False
        for route, body, expected in (("tts", b'{"text":"hello"}', "KOKORO_MODEL_DIR"),
                                      ("asr", b"audio", "WHISPER_MODEL_DIR")):
            request = Request(base + "/api/" + route, data=body, method="POST")
            try:
                urlopen(request)
                assert False, "missing model should fail"
            except HTTPError as error:
                assert error.code == 400
                assert expected in json.load(error)["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_demo_routes_drive_dwell_command_thinking_and_reply(server):
    scenario = json.load(urlopen(server + "/api/scenario"))
    assert [room["id"] for room in scenario["rooms"]] == ["room-1", "room-2"]
    assert scenario["agent"]["greeting"] == "Hello, do you need help?" and scenario["detector"] is False
    assert scenario["characters"]["mira"]["voice"]["pitch"] != scenario["characters"]["rowan"]["voice"]["pitch"]
    assert post(server, "/api/observe", {"gaze": {"kind": "character"}})["state"] == "dwelling"
    time.sleep(0.25)
    assert post(server, "/api/observe", {"gaze": {"kind": "character"}, "speaking": True})["state"] == "listening"
    submitted = post(server, "/api/utterance", {"text": "What is this?", "gaze": {"kind": "object", "target": "r1-rose"}})
    assert submitted["accepted"] and submitted["state"] == "thinking" and submitted["thinking"]["filler"]
    for _ in range(100):
        job = json.load(urlopen(f"{server}/api/job?id={submitted['job']}"))
        if job["done"]:
            break
        time.sleep(0.05)
    reply = job["reply"]
    assert reply["object"] == "rose" and reply["recognition"]["anchors_loaded"] == "room-1"
    assert reply["expression"]["name"] in {"happiness", "anger", "sadness", "fear"}
    assert post(server, "/api/response-finished", {})["state"] == "listening"
    plan = post(server, "/api/animation-plan", {"text": "I think this is a rose."})
    assert plan["entries"] and plan["library_ready"] is False
    assert post(server, "/api/sentiment", {"text": "Beware, it is poisonous!"})["class"] == "Fear"


def test_webcam_routes_fail_gracefully_without_weights(server):
    detect = post(server, "/api/detect", {"image": "data:image/png;base64,AAAA"})
    assert detect["status"] == 400 and "--weights" in detect["error"]
    reset = post(server, "/api/reset", {})
    assert reset["state"] == "idle"
    # A frame from the webcam goes to the camera recogniser; with no detector the agent says so.
    import base64

    import cv2
    import numpy as np

    ok, png = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
    frame = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
    submitted = post(server, "/api/utterance", {"text": "what is this", "gaze": {"kind": "none"}, "image": frame})
    for _ in range(100):
        job = json.load(urlopen(f"{server}/api/job?id={submitted['job']}"))
        if job["done"]:
            break
        time.sleep(0.05)
    assert "--weights" in job["reply"]["recognition"]["error"]
