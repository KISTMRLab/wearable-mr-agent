"""Anchor map setup and loading (paper Section 2.3, Figure 11).

Placement: after scanning a room, anchors are saved together with the object
recognised at each anchor. Loading: devices cannot tell identical rooms apart,
so the anchor manager uses *object recognition* to pick the room-specific
anchor set. Once a room's anchors are loaded, gazing at an anchored object
resolves it from the anchor and object recognition is skipped (Table 1,
Query A versus Query C).

The provider protocol isolates device persistence. ``JsonAnchorProvider`` and
``InMemoryAnchorProvider`` store opaque transforms; real world-locked anchors
need a device adapter (for example WebXR Anchors on Android/Quest or an
OpenXR runtime).
"""
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
    def all(self) -> list[SpatialAnchor]: ...


class InMemoryAnchorProvider:
    def __init__(self, anchors: list[SpatialAnchor] | None = None):
        self._anchors: dict[str, SpatialAnchor] = {a.anchor_id: a for a in anchors or []}

    def save(self, anchor: SpatialAnchor) -> None:
        self._anchors[anchor.anchor_id] = anchor

    def for_room(self, room_fingerprint: str) -> list[SpatialAnchor]:
        return [a for a in self._anchors.values() if a.room_fingerprint == room_fingerprint]

    def all(self) -> list[SpatialAnchor]:
        return list(self._anchors.values())


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

    def all(self) -> list[SpatialAnchor]:
        return self._read()


class AnchorManager:
    def __init__(self, provider: AnchorProvider):
        self.provider = provider
        self.loaded_room: str | None = None
        self.loaded: dict[str, SpatialAnchor] = {}
        self._candidates: set[str] | None = None

    def place(self, anchor: SpatialAnchor) -> None:
        """Anchor placement: save an anchor with the object recognised at it."""
        self.provider.save(anchor)

    def rooms_with(self, object_label: str) -> set[str]:
        label = object_label.casefold()
        return {a.room_fingerprint for a in self.provider.all() if a.object_label.casefold() == label}

    def load(self, room_fingerprint: str) -> list[SpatialAnchor]:
        anchors = self.provider.for_room(room_fingerprint)
        self.loaded_room = room_fingerprint
        self.loaded = {a.anchor_id: a for a in anchors}
        self._candidates = None
        return anchors

    def unload(self) -> None:
        self.loaded_room, self.loaded, self._candidates = None, {}, None

    def observe_recognition(self, object_label: str) -> str | None:
        """Use a recognised label to select the room's anchors. Returns the room when anchors were (re)loaded.

        A label shared by several rooms narrows the candidates; the next
        recognition that leaves a single room loads it.
        """
        rooms = self.rooms_with(object_label)
        if not rooms or self.loaded_room in rooms:
            return None
        candidates = rooms if self._candidates is None else (self._candidates & rooms) or rooms
        if len(candidates) == 1:
            room = next(iter(candidates))
            self.load(room)
            return room
        self._candidates = candidates
        return None

    def resolve(self, target_id: str | None) -> SpatialAnchor | None:
        """The loaded anchor under the user's gaze, if any (recognition can be skipped)."""
        return self.loaded.get(target_id) if target_id else None
