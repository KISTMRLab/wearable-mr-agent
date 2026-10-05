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
class Gaze:
    """What the user's head/gaze ray currently hits.

    ``kind`` is ``"character"`` (the virtual agent), ``"object"`` (a real or
    simulated object; ``target_id`` identifies the instance or anchor) or
    ``"none"``.
    """

    kind: str = "none"
    target_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"character", "object", "none"}:
            raise ValueError(f"Unknown gaze kind: {self.kind!r}")

    @property
    def on_character(self) -> bool:
        return self.kind == "character"

    @classmethod
    def from_dict(cls, value: dict | None) -> "Gaze":
        if not value:
            return cls()
        target = value.get("target") or value.get("target_id")
        return cls(str(value.get("kind") or "none"), None if target is None else str(target))


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
