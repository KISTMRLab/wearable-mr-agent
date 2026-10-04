from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SpatialAnchor:
    anchor_id: str
    room_fingerprint: str
    object_label: str
    transform: list[float]


class AnchorProvider(Protocol):
    def save(self, anchor: SpatialAnchor) -> None: ...
    def for_room(self, room_fingerprint: str) -> list[SpatialAnchor]: ...


class JsonAnchorProvider:
    """Portable persistence adapter; device SDKs can implement the same protocol."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> list[SpatialAnchor]:
        if not self.path.exists():
            return []
        return [SpatialAnchor(**item) for item in json.loads(self.path.read_text(encoding="utf-8"))]

    def save(self, anchor: SpatialAnchor) -> None:
        anchors = [item for item in self._read() if item.anchor_id != anchor.anchor_id]
        anchors.append(anchor)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(item) for item in anchors], indent=2), encoding="utf-8")

    def for_room(self, room_fingerprint: str) -> list[SpatialAnchor]:
        return [item for item in self._read() if item.room_fingerprint == room_fingerprint]

