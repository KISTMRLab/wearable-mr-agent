"""Offline domain chatbot over a curator-authored knowledge JSON.

It implements the paper's chatbot contract ``(Query, Object) -> Reply`` for two
purposes: domain-specific information about a recognised object, and general
conversation (greetings, questions about the guide, answers to the guide's own
questions). Matching uses word boundaries, never substrings, so a topic such as
"use" does not fire inside "because".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .text import find_sequence, words

REFERENCE_WORDS = {"this", "that", "it", "its", "these", "those", "they", "them", "one", "here"}
QUESTION_STARTERS = {"what", "tell", "describe", "explain", "which", "how", "why", "where", "when", "who", "is", "does", "do", "can"}

DEFAULT_GENERAL: list[dict[str, Any]] = [
    {"intent": "greeting", "patterns": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
     "reply": "Hello! I'm your guide. Look at a flower and ask me \"what is this\", or ask me anything about this place."},
    {"intent": "wellbeing", "patterns": ["how are you", "how are you doing", "how is it going"],
     "reply": "I'm doing well, thank you! It's a lovely day to look around."},
    {"intent": "identity", "patterns": ["who are you", "your name", "what are you", "are you real"],
     "reply": "I'm a virtual guide. I can tell you about the things around us and answer your follow-up questions."},
    {"intent": "capability", "patterns": ["what can you do", "help me", "can you help", "need help", "how does this work"],
     "reply": "Look at something and say \"what is this\" or \"tell me about this\". After that you can ask follow-up questions without looking at it again."},
    {"intent": "affirm", "patterns": ["yes", "yes please", "sure", "yeah", "okay"],
     "reply": "Great! Look at a flower and say \"what is this\", and I will tell you about it."},
    {"intent": "decline", "patterns": ["no", "no thanks", "not now", "nope"],
     "reply": "Alright. I'll be right here if you need me."},
    {"intent": "thanks", "patterns": ["thank you", "thanks", "thank"],
     "reply": "You're welcome! I'm glad I could help."},
    {"intent": "farewell", "patterns": ["bye", "goodbye", "see you", "that's all"],
     "reply": "Goodbye! Enjoy the rest of your visit."},
    {"intent": "collection", "patterns": ["what flowers", "which flowers", "what is here", "what can i see", "what else", "what do you know"],
     "reply": "I know about {objects}. Look at one and ask me about it."},
]


@dataclass(frozen=True)
class KnowledgeMatch:
    text: str
    kind: str  # topic | overview | general | unknown_object | need_object | fallback
    object_label: str | None = None
    topic: str | None = None
    intent: str | None = None


class DomainKnowledge:
    """Curated knowledge base usable as the offline chatbot client."""

    name = "knowledge"

    def __init__(self, objects: dict[str, dict[str, Any]], general: list[dict[str, Any]] | None = None,
                 fallback: str | None = None):
        self.objects = {key.casefold(): value for key, value in objects.items()}
        self.general = [dict(item) for item in (general if general is not None else DEFAULT_GENERAL)]
        self.fallback = fallback or "I'm not sure about that. You can ask me about the flowers around you, or about me."

    @classmethod
    def load(cls, path: str | Path) -> "DomainKnowledge":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload.get("objects"), dict):
            raise ValueError("Knowledge JSON must contain an 'objects' mapping")
        general = payload.get("general")
        if general is not None and not isinstance(general, list):
            raise ValueError("Knowledge JSON 'general' must be a list of {intent, patterns, reply}")
        if general is None or payload.get("extend_general", True):
            general = DEFAULT_GENERAL + list(general or [])
        return cls(payload["objects"], general, payload.get("fallback"))

    # -- matching -----------------------------------------------------------------------------------------------------
    @staticmethod
    def _topics(record: dict[str, Any]) -> list[tuple[str, str, list[list[str]]]]:
        topics = record.get("topics", {})
        rows: list[tuple[str, str, list[list[str]]]] = []
        if not isinstance(topics, dict):
            return rows
        for key, value in topics.items():
            if isinstance(value, dict):
                text = str(value.get("text", ""))
                keys = [str(key), *map(str, value.get("keywords", []))]
            else:
                text, keys = str(value), [str(key)]
            rows.append((str(key), text, [words(k) for k in keys if words(k)]))
        return rows

    def _match_general(self, tokens: list[str]) -> dict[str, Any] | None:
        best, best_len = None, 0
        for item in self.general:
            for pattern in item.get("patterns", []):
                needle = words(pattern)
                if len(needle) > best_len and find_sequence(tokens, needle) >= 0:
                    best, best_len = item, len(needle)
        return best

    def _general_text(self, item: dict[str, Any]) -> str:
        names = sorted({str(record.get("name", label)) for label, record in self.objects.items()})
        listing = ", ".join(names[:-1]) + (" and " + names[-1] if len(names) > 1 else (names[0] if names else "this place"))
        return str(item.get("reply", "")).replace("{objects}", listing)

    def _named_object(self, tokens: list[str]) -> str | None:
        for label, record in self.objects.items():
            for name in {label, str(record.get("name", label)).casefold()}:
                if find_sequence(tokens, words(name)) >= 0:
                    return label
        return None

    def match(self, query: str, object_label: str | None) -> KnowledgeMatch:
        tokens = words(query)
        general = self._match_general(tokens)
        refers = bool(REFERENCE_WORDS.intersection(tokens))
        named = self._named_object(tokens)
        label = object_label.casefold() if object_label else None
        if label is None and named is not None and not general:
            label = named  # "tell me about roses" without looking at one
        if label is None:
            if general:
                return KnowledgeMatch(self._general_text(general), "general", None, intent=general.get("intent"))
            if refers:
                return KnowledgeMatch("Which one do you mean? Look at it and say \"what is this\".", "need_object")
            return KnowledgeMatch(self.fallback, "fallback")
        record = self.objects.get(label)
        if record is None:
            if general and not refers:
                return KnowledgeMatch(self._general_text(general), "general", object_label, intent=general.get("intent"))
            return KnowledgeMatch(f"I recognised a {object_label}, but it is not in my guide's knowledge base yet.",
                                  "unknown_object", object_label)
        best: tuple[int, str, str] | None = None
        for key, text, sequences in self._topics(record):
            for needle in sequences:
                if find_sequence(tokens, needle) >= 0 and (best is None or len(needle) > best[0]):
                    best = (len(needle), key, text)
        if best:
            return KnowledgeMatch(best[2], "topic", label, topic=best[1])
        if general and not refers and named is None:
            return KnowledgeMatch(self._general_text(general), "general", label, intent=general.get("intent"))
        if refers or named is not None or (tokens and tokens[0] in QUESTION_STARTERS):
            return KnowledgeMatch(str(record.get("overview", "No authored description is available.")), "overview", label)
        topics = [key for key, _, _ in self._topics(record)]
        hint = f" You can ask about its {', '.join(topics[:3])}." if topics else ""
        return KnowledgeMatch(self.fallback + hint, "fallback", label)

    # -- chatbot client protocol --------------------------------------------------------------------------------------
    def respond(self, query: str, object_label: str | None, history: Sequence[tuple[str, str]] = ()) -> str:
        return self.match(query, object_label).text

    def facts(self, object_label: str | None) -> str:
        """Curated facts used to ground a language-model client."""
        record = self.objects.get(object_label.casefold()) if object_label else None
        if not record:
            return ""
        lines = [str(record.get("overview", ""))]
        lines += [f"{key}: {text}" for key, text, _ in self._topics(record)]
        return "\n".join(line for line in lines if line)
