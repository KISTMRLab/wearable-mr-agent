from __future__ import annotations

import json
import re
from pathlib import Path

from .types import AgentReply, BehaviorEvent


class DomainKnowledge:
    """Grounded object conversation over a user-authored JSON knowledge base."""

    def __init__(self, objects: dict[str, dict[str, object]]):
        self.objects = {key.casefold(): value for key, value in objects.items()}

    @classmethod
    def load(cls, path: str | Path) -> "DomainKnowledge":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload.get("objects"), dict):
            raise ValueError("Knowledge JSON must contain an 'objects' mapping")
        return cls(payload["objects"])

    def answer(self, query: str, object_label: str | None) -> AgentReply:
        if object_label is None:
            return self._reply("Please look at an object before asking an object-specific question.", None, "neutral")
        record = self.objects.get(object_label.casefold())
        if record is None:
            return self._reply(f"I recognized {object_label}, but it is not in this guide's knowledge base.", object_label, "neutral")
        normalized = re.sub(r"[^a-z0-9 ]", " ", query.casefold())
        topics = record.get("topics", {})
        selected: str | None = None
        if isinstance(topics, dict):
            for topic, text in topics.items():
                if str(topic).casefold() in normalized:
                    selected = str(text)
                    break
        text = selected or str(record.get("overview", "No authored description is available."))
        emotion = str(record.get("emotion", "neutral"))
        return self._reply(text, object_label, emotion)

    @staticmethod
    def _reply(text: str, object_label: str | None, emotion: str) -> AgentReply:
        gesture = "point_object" if object_label else "open_hand"
        events = (
            BehaviorEvent("speech", text),
            BehaviorEvent("expression", emotion, 0.65),
            BehaviorEvent("gesture", gesture, 0.8),
            BehaviorEvent("viseme", "derive_from_speech", 1.0),
        )
        return AgentReply(text, object_label, events)

