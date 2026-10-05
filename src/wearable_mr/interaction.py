"""Gaze-and-speech interaction state machine (paper Section 2.1.1, Figure 5).

The user starts a conversation by gazing at the *virtual character* for a dwell
time (four seconds in the paper). The speech recogniser then listens. If the
user stays silent, the character opens the conversation with a proactive
greeting. If the user gazes away from the character while the recogniser is
running *and* is silent, the conversation ends; gazing away while talking (for
example to look at an object while saying "what is this") keeps it open.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class InteractionState(str, Enum):
    IDLE = "idle"
    DWELLING = "dwelling"
    LISTENING = "listening"
    THINKING = "thinking"
    RESPONDING = "responding"


@dataclass(frozen=True)
class InteractionConfig:
    dwell_seconds: float = 4.0
    greeting_after_seconds: float = 3.0
    # The paper ends the conversation as soon as gaze leaves the character while
    # the user is silent. A short tolerance lets a user move their gaze to an
    # object before saying a voice command; set it to 0 for the strict rule.
    lookaway_grace_seconds: float = 1.5
    # Voice commands run on an always-on keyword recogniser, so "what is this"
    # can also open a conversation without first dwelling on the character.
    commands_start_conversation: bool = True
    # Safety net when a client never reports the end of the agent's speech.
    response_timeout_seconds: float = 6.0

    def __post_init__(self) -> None:
        for name in ("dwell_seconds", "greeting_after_seconds", "lookaway_grace_seconds", "response_timeout_seconds"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class InteractionStateMachine:
    config: InteractionConfig = field(default_factory=InteractionConfig)
    state: InteractionState = InteractionState.IDLE
    dwell_started_at: float | None = None
    listening_since: float | None = None
    last_speech_at: float | None = None
    offgaze_silent_since: float | None = None
    responding_until: float | None = None
    greeted: bool = False
    utterances: int = 0

    @property
    def in_conversation(self) -> bool:
        return self.state in {InteractionState.LISTENING, InteractionState.THINKING, InteractionState.RESPONDING}

    def update(self, now: float, on_character: bool, user_speaking: bool = False) -> list[str]:
        """Advance the machine with one gaze/voice-activity sample."""
        events: list[str] = []
        if self.state == InteractionState.IDLE:
            if on_character:
                self.state = InteractionState.DWELLING
                self.dwell_started_at = now
                events.append("dwell_started")
        if self.state == InteractionState.DWELLING:
            if not on_character:
                self.reset()
                events.append("dwell_cancelled")
            elif self.dwell_started_at is not None and now - self.dwell_started_at >= self.config.dwell_seconds:
                self._start_listening(now)
                events.append("listening_started")
            return events
        if self.state == InteractionState.LISTENING:
            if user_speaking:
                self.last_speech_at = now
            if not on_character and not user_speaking:
                if self.offgaze_silent_since is None:
                    self.offgaze_silent_since = now
                if now - self.offgaze_silent_since >= self.config.lookaway_grace_seconds:
                    self.reset()
                    events.append("conversation_ended")
                    return events
            else:
                self.offgaze_silent_since = None
            marks = [t for t in (self.listening_since, self.last_speech_at) if t is not None]
            quiet_since = max(marks) if marks else now
            if (not self.greeted and self.utterances == 0 and on_character and not user_speaking
                    and now - quiet_since >= self.config.greeting_after_seconds):
                self.greeted = True
                events.append("proactive_greeting")
        elif self.state == InteractionState.RESPONDING:
            if self.responding_until is not None and now >= self.responding_until:
                self.finish_response(now)
                events.append("response_timeout")
        return events

    def start_listening(self, now: float) -> None:
        """Open the recogniser directly (camera/CLI adapters without a character)."""
        if self.state in {InteractionState.IDLE, InteractionState.DWELLING}:
            self._start_listening(now)

    def _start_listening(self, now: float) -> None:
        self.state = InteractionState.LISTENING
        self.listening_since = now
        self.last_speech_at = None
        self.offgaze_silent_since = None
        self.dwell_started_at = None

    def accepts(self, is_command: bool = False) -> tuple[bool, str]:
        if self.state == InteractionState.LISTENING:
            return True, "listening"
        if self.state in {InteractionState.IDLE, InteractionState.DWELLING}:
            if is_command and self.config.commands_start_conversation:
                return True, "voice_command"
            return False, "not_listening"
        return False, "busy"

    def submit_utterance(self, now: float | None = None, is_command: bool = False) -> None:
        accepted, reason = self.accepts(is_command)
        if not accepted:
            raise RuntimeError("An utterance can only be submitted while listening" if reason == "not_listening"
                               else "The agent is still answering")
        self.state = InteractionState.THINKING
        self.utterances += 1
        self.offgaze_silent_since = None
        self.dwell_started_at = None

    def begin_response(self, now: float | None = None, duration_seconds: float | None = None) -> None:
        if self.state not in {InteractionState.THINKING, InteractionState.LISTENING}:
            raise RuntimeError("A response requires the thinking or listening state")
        self.state = InteractionState.RESPONDING
        self.responding_until = None if now is None else now + (duration_seconds or 0.0) + self.config.response_timeout_seconds

    def finish_response(self, now: float | None = None) -> None:
        if self.state not in {InteractionState.RESPONDING, InteractionState.THINKING}:
            return
        self.state = InteractionState.LISTENING
        self.responding_until = None
        self.offgaze_silent_since = None
        if now is not None:
            self.listening_since = now
            self.last_speech_at = now

    def reset(self) -> None:
        self.state = InteractionState.IDLE
        self.dwell_started_at = None
        self.listening_since = None
        self.last_speech_at = None
        self.offgaze_silent_since = None
        self.responding_until = None
        self.greeted = False
        self.utterances = 0
