"""Chatbot with an integrated sentiment engine (paper Section 2.1.3, Figure 6).

``Chatbot.ask(query, object)`` returns ``(reply, sentiment class, sentiment
level)``. The language part is a pluggable client: an OpenAI-compatible
endpoint, a local command, or the offline :class:`DomainKnowledge` fallback.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .clients import OpenAICompatibleClient, ServiceError, parse_json_object, run_command
from .knowledge import DomainKnowledge
from .sentiment import Sentiment, SentimentEngine


class ChatbotClient(Protocol):
    def respond(self, query: str, object_label: str | None, history: Sequence[tuple[str, str]]) -> str: ...


GUIDE_PROMPT = (
    "You are a friendly virtual tour guide standing next to a visitor in a mixed-reality exhibition. "
    "Answer in one to three short spoken sentences, without lists or markdown. "
    "If the visitor asks about the object they are looking at, use only the curated facts provided; "
    "if the facts do not answer the question, say so briefly. You may also chat generally about yourself and the visit."
)


class OpenAICompatibleChatbot:
    """Hosted or local LLM behind an OpenAI-compatible endpoint, grounded by curated facts."""

    name = "openai-compatible"

    def __init__(self, client: OpenAICompatibleClient, knowledge: DomainKnowledge | None = None, system_prompt: str = GUIDE_PROMPT):
        self.client, self.knowledge, self.system_prompt = client, knowledge, system_prompt

    def respond(self, query: str, object_label: str | None, history: Sequence[tuple[str, str]] = ()) -> str:
        context = f"Object the visitor asked about: {object_label}." if object_label else "No object has been recognised yet."
        facts = self.knowledge.facts(object_label) if self.knowledge else ""
        if facts:
            context += "\nCurated facts:\n" + facts
        messages = [{"role": "system", "content": self.system_prompt + "\n\n" + context}]
        for user, agent in history:
            messages += [{"role": "user", "content": user}, {"role": "assistant", "content": agent}]
        messages.append({"role": "user", "content": query})
        return self.client.complete(messages)


class CommandChatbot:
    """A local program: JSON ``{"query","object","history"}`` on stdin, reply text or ``{"reply"}`` on stdout."""

    name = "command"

    def __init__(self, command: str | Sequence[str], timeout: float = 30.0):
        self.command, self.timeout = command, timeout

    def respond(self, query: str, object_label: str | None, history: Sequence[tuple[str, str]] = ()) -> str:
        output = run_command(self.command, {"query": query, "object": object_label, "history": [list(h) for h in history]}, self.timeout)
        value = parse_json_object(output) if output.lstrip().startswith("{") else None
        if value is not None:
            reply = value.get("reply", value.get("text"))
            if not isinstance(reply, str) or not reply.strip():
                raise ServiceError("Command JSON has no 'reply' text")
            return reply.strip()
        return output


@dataclass(frozen=True)
class ChatReply:
    text: str
    object_label: str | None
    sentiment: Sentiment
    source: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "object": self.object_label, "sentiment": self.sentiment.to_dict(),
                "source": self.source, "error": self.error}


class Chatbot:
    """(Query, Object) -> (Reply, Sentiment class, Sentiment level) with conversation history."""

    def __init__(self, client: ChatbotClient, sentiment: SentimentEngine | None = None,
                 fallback: ChatbotClient | None = None, history_turns: int = 6):
        self.client = client
        self.fallback = fallback
        self.sentiment = sentiment or SentimentEngine()
        self.history: deque[tuple[str, str]] = deque(maxlen=max(0, history_turns))

    @property
    def source(self) -> str:
        return getattr(self.client, "name", type(self.client).__name__)

    def reset(self) -> None:
        self.history.clear()

    def ask(self, query: str, object_label: str | None) -> ChatReply:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be non-empty text")
        query = query.strip()
        source, error = self.source, None
        try:
            text = self.client.respond(query, object_label, tuple(self.history))
        except (ServiceError, ValueError, json.JSONDecodeError) as exc:
            if self.fallback is None:
                raise
            error = str(exc)
            text = self.fallback.respond(query, object_label, tuple(self.history))
            source = getattr(self.fallback, "name", type(self.fallback).__name__) + " (fallback)"
        self.history.append((query, text))
        sentiment = self.sentiment.analyze(text)
        return ChatReply(text, object_label, sentiment, source, error or self.sentiment.last_error)
