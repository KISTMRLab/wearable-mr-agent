"""Small JSON-over-HTTP and local-command transports for pluggable services.

The paper ran its chatbot, sentiment engine and object recogniser as cloud web
services. These helpers keep that boundary: any OpenAI-compatible
``/chat/completions`` server (hosted or local, e.g. vLLM, llama.cpp, Ollama) or a
local command can be plugged in. Nothing here downloads a model.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ServiceError(RuntimeError):
    """A configured remote or local service failed or returned an unusable answer."""


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: float = 20.0) -> Any:
    request = Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                      headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise ServiceError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise ServiceError(f"{url} is unavailable: {exc}") from exc


class OpenAICompatibleClient:
    """Minimal chat-completions client.

    ``endpoint`` may be a base URL (``http://127.0.0.1:8000/v1``) or the full
    ``.../chat/completions`` URL. The API key is read from the environment
    variable named by ``api_key_env`` so secrets never enter config files.
    """

    def __init__(self, endpoint: str, model: str, api_key_env: str | None = "OPENAI_API_KEY",
                 timeout: float = 20.0, temperature: float | None = 0.4, max_tokens: int = 220):
        endpoint = endpoint.rstrip("/")
        self.url = endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: Sequence[dict[str, str]], **extra: Any) -> str:
        payload: dict[str, Any] = {"model": self.model, "messages": list(messages), "max_tokens": self.max_tokens}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(extra)
        headers = {}
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        answer = post_json(self.url, payload, headers, self.timeout)
        try:
            content = answer["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ServiceError(f"Unexpected chat-completions response from {self.url}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ServiceError(f"Empty chat-completions response from {self.url}")
        return content.strip()


def run_command(command: str | Sequence[str], payload: dict[str, Any], timeout: float = 30.0) -> str:
    """Send ``payload`` as JSON on stdin to a local command and return its stdout."""
    argv = shlex.split(command, posix=os.name != "nt") if isinstance(command, str) else list(command)
    if not argv:
        raise ServiceError("Empty command")
    try:
        done = subprocess.run(argv, input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ServiceError(f"Command failed: {exc}") from exc
    if done.returncode:
        raise ServiceError(f"Command exited with {done.returncode}: {done.stderr.strip()[:300]}")
    output = done.stdout.strip()
    if not output:
        raise ServiceError("Command produced no output")
    return output


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating a fenced block or surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, stop = text.find("{"), text.rfind("}")
    if start < 0 or stop <= start:
        return None
    try:
        value = json.loads(text[start:stop + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None
