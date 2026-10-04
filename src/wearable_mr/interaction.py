from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InteractionState(str, Enum):
    IDLE = "idle"
    DWELLING = "dwelling"
    LISTENING = "listening"
    THINKING = "thinking"
    RESPONDING = "responding"


@dataclass
class InteractionStateMachine:
    dwell_seconds: float = 4.0
    state: InteractionState = InteractionState.IDLE
    target: str | None = None
    dwell_started_at: float | None = None

    def update_gaze(self, target: str | None, now: float, user_speaking: bool = False) -> list[str]:
        events: list[str] = []
        if target is None:
            if self.state == InteractionState.LISTENING and not user_speaking:
                events.append("conversation_ended")
            if self.state in {InteractionState.IDLE, InteractionState.DWELLING, InteractionState.LISTENING}:
                self.reset()
            return events

        if target != self.target:
            self.target = target
            self.dwell_started_at = now
            self.state = InteractionState.DWELLING
            events.append("dwell_started")
            return events

        if self.state == InteractionState.DWELLING and self.dwell_started_at is not None:
            if now - self.dwell_started_at >= self.dwell_seconds:
                self.state = InteractionState.LISTENING
                events.append("listening_started")
        return events

    def submit_utterance(self) -> None:
        if self.state != InteractionState.LISTENING:
            raise RuntimeError("An utterance can only be submitted while listening")
        self.state = InteractionState.THINKING

    def begin_response(self) -> None:
        if self.state != InteractionState.THINKING:
            raise RuntimeError("A response requires the thinking state")
        self.state = InteractionState.RESPONDING

    def finish_response(self) -> None:
        self.state = InteractionState.LISTENING

    def reset(self) -> None:
        self.state = InteractionState.IDLE
        self.target = None
        self.dwell_started_at = None

