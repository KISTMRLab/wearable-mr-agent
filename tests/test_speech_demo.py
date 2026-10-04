import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from wearable_mr.demo import app
from wearable_mr.knowledge import DomainKnowledge


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
