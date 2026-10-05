"""Limited voice commands that trigger recognition of the gazed object (paper Section 2.1)."""
from __future__ import annotations

from typing import Iterable

from .text import find_sequence, words

DEFAULT_COMMANDS = ("what is this", "what's this", "tell me about this", "what is that", "what's that",
                    "tell me about that", "what am i looking at")


class VoiceCommands:
    def __init__(self, phrases: Iterable[str] = DEFAULT_COMMANDS):
        self.phrases = [p for p in dict.fromkeys(str(p).strip() for p in phrases) if words(p)]
        if not self.phrases:
            raise ValueError("At least one voice command is required")
        self._keys = sorted(((words(p), p) for p in self.phrases), key=lambda item: -len(item[0]))

    def match(self, utterance: str) -> str | None:
        """The command phrase contained in ``utterance`` (word boundaries), or None."""
        tokens = words(utterance)
        for key, phrase in self._keys:
            if find_sequence(tokens, key) >= 0:
                return phrase
        return None
