from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    @property
    def area(self) -> float:
        return max(0.0, self.right - self.left) * max(0.0, self.bottom - self.top)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: BoundingBox
    track_id: str | None = None


@dataclass(frozen=True)
class BehaviorEvent:
    channel: str
    value: str
    intensity: float = 1.0
    start_s: float = 0.0


@dataclass(frozen=True)
class AgentReply:
    text: str
    object_label: str | None
    events: tuple[BehaviorEvent, ...] = field(default_factory=tuple)

