"""Word tokenisation shared by commands, knowledge matching and the animation builder."""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*")


@dataclass(frozen=True)
class Token:
    text: str
    norm: str
    start: int
    end: int

    @property
    def base(self) -> str:
        """The word before an apostrophe: "i'll" -> "i", "flower's" -> "flower"."""
        return self.norm.split("'", 1)[0]


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(0), m.group(0).casefold().replace("’", "'"), m.start(), m.end())
            for m in _WORD.finditer(str(text))]


def words(text: str) -> list[str]:
    return [token.norm for token in tokenize(text)]


def same_word(a: str, b: str) -> bool:
    """Case-folded equality that tolerates a possessive or a plural -s/-es."""
    if a == b:
        return True
    a, b = a.split("'", 1)[0], b.split("'", 1)[0]
    if a == b:
        return True
    for x, y in ((a, b), (b, a)):
        if len(y) >= 3 and (x == y + "s" or x == y + "es"):
            return True
    return False


def find_sequence(haystack: list[str], needle: list[str], start: int = 0) -> int:
    """Index of the first contiguous word-boundary occurrence of ``needle``, or -1."""
    if not needle:
        return -1
    for i in range(start, len(haystack) - len(needle) + 1):
        if all(same_word(haystack[i + j], needle[j]) for j in range(len(needle))):
            return i
    return -1


def contains_phrase(text_or_words: str | list[str], phrase: str) -> bool:
    haystack = words(text_or_words) if isinstance(text_or_words, str) else text_or_words
    return find_sequence(haystack, words(phrase)) >= 0
