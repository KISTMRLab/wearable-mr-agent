"""Animation builder over an expert-authored text -> animation mapping table.

Paper Section 2.2, Figure 8: human experts author a table that maps phrases and
individual words to body animations; the animation builder scans the reply text
and emits an animation sequence that plays in parallel with speech and the
facial emotion. Phrases are matched first, longest first, then single words in
the text that no phrase claimed. Entries are ordered by their position in the
reply, so "I think this" plays the "I"/"I think" animation before "this".

Animations are either recorded clips from the locally prepared public BEAT bank
(``outputs/beat-library/bank.json``, referenced by stable BEAT window id such as
``1_wayne_0_1_1:780-855``, or by bank clip id) or procedural
gestures understood by the shared renderer (``stage.gesture(name)``).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .text import Token, same_word, tokenize, words

PROCEDURAL_GESTURES = {"idle", "open", "welcome", "explain", "point", "wave", "think", "beat", "nod", "yes", "agree"}
WORDS_PER_SECOND = 2.6  # matches the shared speech timing estimate


@dataclass(frozen=True)
class AnimationSpec:
    clip: str | None = None
    procedural: str | None = None

    @classmethod
    def from_value(cls, value: Any, where: str) -> "AnimationSpec":
        if isinstance(value, str):
            value = {"procedural": value} if value in PROCEDURAL_GESTURES else {"clip": value}
        if not isinstance(value, dict):
            raise ValueError(f"{where}: entry must be an object with 'clip' and/or 'procedural'")
        clip, procedural = value.get("clip"), value.get("procedural")
        if clip is None and procedural is None:
            raise ValueError(f"{where}: give a 'clip' id, a 'procedural' gesture, or both")
        if procedural is not None and procedural not in PROCEDURAL_GESTURES:
            raise ValueError(f"{where}: unknown procedural gesture {procedural!r}; use one of {sorted(PROCEDURAL_GESTURES)}")
        return cls(None if clip is None else str(clip), procedural)


@dataclass(frozen=True)
class AnimationEntry:
    trigger: str
    match: str  # "phrase" | "word"
    kind: str  # "clip" | "procedural"
    clip_id: str | None
    gesture: str  # procedural gesture, also the stand-in when clip frames are unavailable
    start_word: int
    end_word: int
    start_char: int
    end_char: int
    start_s: float
    end_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnimationTable:
    """Editable phrase/word mapping. JSON: ``{"phrases": {...}, "words": {...}, "min_gap_words": 3}``."""

    def __init__(self, phrases: dict[str, Any], words_map: dict[str, Any], min_gap_words: int = 3):
        self.phrases: list[tuple[list[str], str, AnimationSpec]] = []
        for phrase, value in phrases.items():
            key = words(phrase)
            if len(key) < 2:
                raise ValueError(f"Phrase {phrase!r} must have at least two words; put single words under 'words'")
            self.phrases.append((key, " ".join(key), AnimationSpec.from_value(value, f"phrases[{phrase!r}]")))
        # Longest first; ties keep authoring order.
        self.phrases.sort(key=lambda item: -len(item[0]))
        self.words: dict[str, AnimationSpec] = {}
        for word, value in words_map.items():
            key = words(word)
            if len(key) != 1:
                raise ValueError(f"Word trigger {word!r} must be a single word")
            self.words[key[0]] = AnimationSpec.from_value(value, f"words[{word!r}]")
        if min_gap_words < 0:
            raise ValueError("min_gap_words must be non-negative")
        self.min_gap_words = int(min_gap_words)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnimationTable":
        return cls(payload.get("phrases", {}), payload.get("words", {}), int(payload.get("min_gap_words", 3)))

    @classmethod
    def load(cls, path: str | Path) -> "AnimationTable":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def clip_ids(self) -> set[str]:
        specs = [spec for _, _, spec in self.phrases] + list(self.words.values())
        return {spec.clip for spec in specs if spec.clip}

    def word_spec(self, token: Token) -> tuple[str, AnimationSpec] | None:
        for candidate in (token.norm, token.base):
            if candidate in self.words:
                return candidate, self.words[candidate]
        for key, spec in self.words.items():
            if same_word(token.norm, key):
                return key, spec
        return None


class ClipLibrary:
    """Read-only view of the locally prepared BEAT clip bank (shared format)."""

    def __init__(self, bank: dict[str, Any]):
        clips = bank.get("clips")
        if not isinstance(clips, list) or not clips:
            raise ValueError("Clip bank has no clips")
        self.bank = bank
        self.clips = {str(c["id"]): c for c in clips if "id" in c and c.get("positions")}
        self.order = [str(c["id"]) for c in clips if str(c.get("id")) in self.clips]
        # Stable BEAT window ids ("<take>:<start>-<end>" at 30 fps) survive bank rebuilds; ordinals do not.
        self.windows: dict[str, str] = {}
        for cid in self.order:
            source = self.clips[cid].get("source") or {}
            window = source.get("window_id")
            if not window and source.get("take") and "start_frame" in source and "end_frame" in source:
                window = f"{source['take']}:{int(source['start_frame'])}-{int(source['end_frame'])}"
            if window:
                self.windows.setdefault(str(window), cid)

    @classmethod
    def load(cls, path: str | Path) -> "ClipLibrary | None":
        path = Path(path)
        if not path.is_file():
            return None
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def resolve(self, clip_id: str | None) -> str | None:
        """Exact bank id or BEAT window id (``1_wayne_0_1_1:780-855``); otherwise an ordinal id ``beat_NN``
        selects the NN-th clip of the bank. A window id missing from this bank resolves to None, so the
        table's procedural gesture plays instead of an unrelated clip."""
        if not clip_id:
            return None
        if clip_id in self.clips:
            return clip_id
        if clip_id in self.windows:
            return self.windows[clip_id]
        if ":" in clip_id:
            return None
        stem, _, number = clip_id.rpartition("_")
        if stem and number.isdigit() and 0 < int(number) <= len(self.order):
            return self.order[int(number) - 1]
        return None

    def payload(self, clip_ids: list[str]) -> dict[str, Any]:
        fps = float(self.bank.get("fps", 30))
        return {"ready": True, "fps": fps, "joint_order": self.bank.get("joint_names", []),
                "axisSigns": self.bank.get("axisSigns", [1, 1, 1]),
                "clips": {cid: {"id": cid, "text": self.clips[cid].get("text", ""), "frames": self.clips[cid]["positions"],
                                "duration": len(self.clips[cid]["positions"]) / fps, "source": self.clips[cid].get("source")}
                          for cid in dict.fromkeys(clip_ids) if cid in self.clips}}


class AnimationBuilder:
    def __init__(self, table: AnimationTable, library: ClipLibrary | None = None,
                 words_per_second: float = WORDS_PER_SECOND, default_clip_standin: str = "beat"):
        self.table, self.library = table, library
        self.words_per_second = words_per_second
        self.default_clip_standin = default_clip_standin

    def _spans(self, tokens: list[Token]) -> list[tuple[int, int, str, str, AnimationSpec]]:
        claimed = [False] * len(tokens)
        norms = [t.norm for t in tokens]
        found: list[tuple[int, int, str, str, AnimationSpec]] = []
        for key, phrase, spec in self.table.phrases:
            i = 0
            while i <= len(tokens) - len(key):
                if not any(claimed[i:i + len(key)]) and all(same_word(norms[i + j], key[j]) for j in range(len(key))):
                    for j in range(i, i + len(key)):
                        claimed[j] = True
                    found.append((i, i + len(key), phrase, "phrase", spec))
                    i += len(key)
                else:
                    i += 1
        gap = self.table.min_gap_words
        for i, token in enumerate(tokens):
            if claimed[i]:
                continue
            hit = self.table.word_spec(token)
            if hit and all(abs(i - start) >= gap for start, *_ in found):
                found.append((i, i + 1, hit[0], "word", hit[1]))
        return sorted(found, key=lambda item: item[0])

    def build(self, text: str, duration_seconds: float | None = None) -> list[AnimationEntry]:
        tokens = tokenize(text)
        if not tokens:
            return []
        length = max(1, len(text))
        duration = duration_seconds or max(0.8, len(tokens) / self.words_per_second)
        spans = self._spans(tokens)
        entries: list[AnimationEntry] = []
        for index, (start, end, trigger, match, spec) in enumerate(spans):
            clip = spec.clip
            if clip and self.library is not None:
                clip = self.library.resolve(clip)
            if clip is None and spec.procedural is None:
                continue  # clip missing from this bank and no procedural alternative
            kind = "clip" if clip else "procedural"
            gesture = spec.procedural or self.default_clip_standin
            next_word = spans[index + 1][0] if index + 1 < len(spans) else len(tokens)
            start_char = tokens[start].start
            end_char = tokens[next_word].start if next_word < len(tokens) else length
            entries.append(AnimationEntry(trigger, match, kind, clip, gesture, start, max(end, next_word), start_char, end_char,
                                          round(start_char / length * duration, 3), round(end_char / length * duration, 3)))
        return entries

    def plan(self, text: str) -> dict[str, Any]:
        entries = self.build(text)
        plan: dict[str, Any] = {"text": text, "entries": [e.to_dict() for e in entries],
                                "estimated_duration_s": round(max(0.8, len(tokenize(text)) / self.words_per_second), 3),
                                "library_ready": self.library is not None}
        if self.library is not None:
            plan["motion"] = self.library.payload([e.clip_id for e in entries if e.kind == "clip" and e.clip_id])
        return plan
