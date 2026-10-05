"""Sentiment engine: reply text -> (sentiment class, sentiment level).

The paper's chatbot attaches one of four classes (Joy, Angry, Sad, Fear) and one
of three levels (High, Medium, Low) to each reply; the character's face uses
them for the whole utterance (Section 2.1.3, Figure 6-7). A classifier client
can be plugged in (a JSON web service or an OpenAI-compatible model); the small
lexicon classifier is the offline fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .clients import OpenAICompatibleClient, ServiceError, parse_json_object, post_json
from .text import tokenize

CLASSES = ("Joy", "Angry", "Sad", "Fear")
LEVELS = ("Low", "Medium", "High")
# Shared renderer names (static/avatar.js setExpression) and preset indices.
RENDERER_EMOTION = {"Joy": "happiness", "Angry": "anger", "Sad": "sadness", "Fear": "fear"}
RENDERER_LEVEL = {"Low": 1, "Medium": 2, "High": 3}

_CLASS_ALIASES = {
    "joy": "Joy", "happy": "Joy", "happiness": "Joy", "positive": "Joy", "love": "Joy", "pos": "Joy",
    "angry": "Angry", "anger": "Angry", "annoyance": "Angry", "rage": "Angry", "disgust": "Angry",
    "sad": "Sad", "sadness": "Sad", "negative": "Sad", "grief": "Sad", "neg": "Sad",
    "fear": "Fear", "afraid": "Fear", "scared": "Fear", "anxiety": "Fear", "nervousness": "Fear",
}
_LEVEL_ALIASES = {"low": "Low", "weak": "Low", "mild": "Low", "1": "Low",
                  "medium": "Medium", "moderate": "Medium", "mid": "Medium", "2": "Medium",
                  "high": "High", "strong": "High", "intense": "High", "3": "High"}


def normalize_class(value: Any) -> str | None:
    return _CLASS_ALIASES.get(str(value).strip().casefold()) if value is not None else None


def normalize_level(value: Any) -> str | None:
    return _LEVEL_ALIASES.get(str(value).strip().casefold()) if value is not None else None


def level_from_score(score: float, medium: float = 1.25, high: float = 2.5) -> str:
    return "High" if score >= high else "Medium" if score >= medium else "Low"


@dataclass(frozen=True)
class Sentiment:
    sentiment_class: str
    level: str
    score: float = 0.0
    source: str = "lexicon"

    def __post_init__(self) -> None:
        if self.sentiment_class not in CLASSES:
            raise ValueError(f"Sentiment class must be one of {CLASSES}")
        if self.level not in LEVELS:
            raise ValueError(f"Sentiment level must be one of {LEVELS}")

    @property
    def expression(self) -> dict[str, Any]:
        """Arguments for the renderer: ``stage.setExpression(name, level)``."""
        return {"name": RENDERER_EMOTION[self.sentiment_class], "level": RENDERER_LEVEL[self.level]}

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["class"] = value.pop("sentiment_class")
        value["expression"] = self.expression
        return value


class SentimentClassifier(Protocol):
    def classify(self, text: str) -> Sentiment: ...


# Weighted cue words. Weight 2 marks a strong cue. Kept deliberately small and
# editable; pass ``lexicon=`` to replace it.
DEFAULT_LEXICON: dict[str, dict[str, float]] = {
    "Joy": {w: 1.0 for w in (
        "happy glad joy enjoy enjoys enjoyed love loves lovely beautiful pretty nice good great wonderful "
        "delight delighted pleasure pleased welcome cheerful bright sweet fragrant favorite favourite popular "
        "fun smile smiles celebrate celebrates thank thanks hope hopeful elegant gentle charming fresh hello").split()},
    "Sad": {w: 1.0 for w in (
        "sad sadly sorry unfortunately loss lost lose die dies died dying wilt wilts wilted wither withers "
        "grief mourning mourn lonely miss missed regret tears cry gone decline declining poor endangered "
        "fade fades faded tired").split()},
    "Angry": {w: 1.0 for w in (
        "angry anger annoyed annoying hate hates rage outrage outrageous mad irritating frustrating frustrated "
        "unacceptable destroy destroys destroyed vandal vandalism invasive aggressive ruin ruined damn stupid "
        "disgusting rude").split()},
    "Fear": {w: 1.0 for w in (
        "afraid fear fears scared scary frightening danger dangerous poison poisonous toxic warning careful "
        "beware threat threatened risk risky harmful venom venomous sharp thorn thorns panic worried worry "
        "nervous alarm alarming hazard caution").split()},
}
for _cls, _words in {"Joy": "wonderful delighted fantastic amazing magnificent", "Sad": "devastated heartbroken tragic",
                     "Angry": "furious outraged hateful", "Fear": "terrified terrifying deadly lethal"}.items():
    for _w in _words.split():
        DEFAULT_LEXICON[_cls][_w] = 2.0
INTENSIFIERS = {"very": 1.5, "really": 1.5, "so": 1.3, "extremely": 2.0, "incredibly": 2.0, "absolutely": 1.8,
                "highly": 1.5, "truly": 1.4, "quite": 1.2, "most": 1.3}
NEGATIONS = {"not", "no", "never", "without", "hardly", "isn't", "aren't", "don't", "doesn't", "didn't", "can't", "won't"}


class LexiconSentimentClassifier:
    """Offline fallback: weighted cue words with intensifiers, negation and '!' emphasis."""

    def __init__(self, lexicon: dict[str, dict[str, float]] | None = None,
                 default: tuple[str, str] = ("Joy", "Low"), medium: float = 1.25, high: float = 2.5):
        self.lexicon = {cls: {k.casefold(): float(v) for k, v in words.items()} for cls, words in (lexicon or DEFAULT_LEXICON).items()}
        unknown = set(self.lexicon) - set(CLASSES)
        if unknown:
            raise ValueError(f"Unknown sentiment classes in lexicon: {sorted(unknown)}")
        self.default = default
        self.medium, self.high = medium, high

    def scores(self, text: str) -> dict[str, float]:
        tokens = [t.norm for t in tokenize(text)]
        scores = {cls: 0.0 for cls in CLASSES}
        for i, word in enumerate(tokens):
            for cls, cues in self.lexicon.items():
                weight = cues.get(word)
                if weight is None:
                    continue
                window = tokens[max(0, i - 2):i]
                if any(w in NEGATIONS or w.endswith("n't") for w in window):
                    if cls == "Joy":  # "not happy" leans sad; negated negatives are dropped
                        scores["Sad"] += 0.5 * weight
                    continue
                boost = max([INTENSIFIERS.get(w, 1.0) for w in window] or [1.0])
                scores[cls] += weight * boost
        top = max(scores, key=lambda c: (scores[c], -CLASSES.index(c)))
        if scores[top] > 0:
            scores[top] += min(1.0, 0.5 * str(text).count("!"))
        return scores

    def classify(self, text: str) -> Sentiment:
        scores = self.scores(text)
        top = max(CLASSES, key=lambda c: (scores[c], -CLASSES.index(c)))
        if scores[top] <= 0:
            return Sentiment(self.default[0], self.default[1], 0.0, "lexicon-default")
        return Sentiment(top, level_from_score(scores[top], self.medium, self.high), round(scores[top], 3), "lexicon")


def sentiment_from_payload(value: Any, source: str) -> Sentiment:
    """Accept ``{"class","level"}``, ``{"sentiment_class","sentiment_level"}`` or ``{"label","score"}``."""
    if not isinstance(value, dict):
        raise ServiceError("Sentiment service did not return a JSON object")
    cls = normalize_class(value.get("class", value.get("sentiment_class", value.get("label"))))
    if cls is None:
        raise ServiceError(f"Sentiment service returned an unknown class: {value!r}")
    level = normalize_level(value.get("level", value.get("sentiment_level")))
    score = value.get("score")
    if level is None and isinstance(score, (int, float)):
        # Probability-like scores: < .5 low, < .8 medium, otherwise high.
        level = "High" if score >= .8 else "Medium" if score >= .5 else "Low"
    if level is None:
        raise ServiceError(f"Sentiment service returned no level or score: {value!r}")
    return Sentiment(cls, level, float(score) if isinstance(score, (int, float)) else 0.0, source)


class HttpSentimentClassifier:
    """POST ``{"text": ...}`` to a JSON endpoint (the paper's cloud sentiment engine boundary)."""

    def __init__(self, endpoint: str, timeout: float = 10.0, headers: dict[str, str] | None = None):
        self.endpoint, self.timeout, self.headers = endpoint, timeout, headers or {}

    def classify(self, text: str) -> Sentiment:
        return sentiment_from_payload(post_json(self.endpoint, {"text": text}, self.headers, self.timeout), "http")


SENTIMENT_PROMPT = ("Classify the emotion a virtual guide should show while saying the user's text. "
                    "Answer only with JSON: {\"class\": one of \"Joy\", \"Angry\", \"Sad\", \"Fear\", "
                    "\"level\": one of \"High\", \"Medium\", \"Low\"}.")


class OpenAICompatibleSentimentClassifier:
    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    def classify(self, text: str) -> Sentiment:
        answer = self.client.complete([{"role": "system", "content": SENTIMENT_PROMPT}, {"role": "user", "content": text}])
        value = parse_json_object(answer)
        if value is None:
            raise ServiceError(f"Sentiment model did not answer with JSON: {answer[:120]!r}")
        return sentiment_from_payload(value, "openai-compatible")


class SentimentEngine:
    """Use the configured classifier; fall back to the offline lexicon on failure."""

    def __init__(self, classifier: SentimentClassifier | None = None, fallback: SentimentClassifier | None = None):
        self.fallback = fallback or LexiconSentimentClassifier()
        self.classifier = classifier or self.fallback
        self.last_error: str | None = None

    def analyze(self, text: str) -> Sentiment:
        self.last_error = None
        if self.classifier is self.fallback:
            return self.fallback.classify(text)
        try:
            return self.classifier.classify(text)
        except (ServiceError, ValueError) as exc:
            self.last_error = str(exc)
            result = self.fallback.classify(text)
            return Sentiment(result.sentiment_class, result.level, result.score, result.source + "-fallback")
