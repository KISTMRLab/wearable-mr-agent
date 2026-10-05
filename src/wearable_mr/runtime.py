"""Agent runtime: gaze/speech interaction, vision manager, chatbot, anchors and animation builder.

The runtime is the module manager of the paper's architecture (Figure 3). It
is single-user and thread-safe. Long work (object recognition and the chatbot
round trip) runs as a job; while it is pending the agent is in the THINKING
state and the client plays a thinking animation and a filler utterance such as
"Let me see..." (paper Section 4, latency hiding).
"""
from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import Executor
from dataclasses import dataclass, field, fields
from typing import Any, Callable

from .anchors import AnchorManager
from .animation import AnimationBuilder
from .chatbot import Chatbot
from .commands import DEFAULT_COMMANDS, VoiceCommands
from .interaction import InteractionConfig, InteractionState, InteractionStateMachine
from .recognition import RecognitionError, Recognizer
from .text import tokenize
from .types import Gaze


@dataclass(frozen=True)
class AgentConfig:
    dwell_seconds: float = 4.0
    greeting_after_seconds: float = 3.0
    lookaway_grace_seconds: float = 1.5
    commands_start_conversation: bool = True
    response_timeout_seconds: float = 6.0
    voice_commands: tuple[str, ...] = DEFAULT_COMMANDS
    greeting: str = "Hello, do you need help?"
    fillers: tuple[str, ...] = ("Let me see.", "Let me think about it.", "Hmm, let me have a look.")
    # Without object recognition the chatbot is usually quick; speak a filler
    # only if the answer is still pending after this delay.
    filler_delay_seconds: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "AgentConfig":
        value = dict(value or {})
        known = {f.name for f in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"Unknown agent settings: {sorted(unknown)}")
        for key in ("voice_commands", "fillers"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)

    @property
    def interaction(self) -> InteractionConfig:
        return InteractionConfig(self.dwell_seconds, self.greeting_after_seconds, self.lookaway_grace_seconds,
                                 self.commands_start_conversation, self.response_timeout_seconds)


class _InlineExecutor:
    """Run jobs synchronously (tests and command-line use)."""

    def submit(self, fn: Callable[..., Any], *args: Any) -> None:
        fn(*args)


@dataclass
class Job:
    job_id: str
    text: str
    command: str | None
    gaze: Gaze
    frame: Any
    generation: int
    submitted_at: float
    done: bool = False
    result: dict[str, Any] | None = None
    events: list[str] = field(default_factory=list)


def estimate_speech_seconds(text: str, words_per_second: float = 2.6) -> float:
    return max(0.8, len(tokenize(text)) / words_per_second)


class AgentRuntime:
    def __init__(self, chatbot: Chatbot, *, builder: AnimationBuilder | None = None, recognizer: Recognizer | None = None,
                 anchors: AnchorManager | None = None, config: AgentConfig | None = None,
                 executor: Executor | _InlineExecutor | None = None, clock: Callable[[], float] = time.monotonic):
        self.chatbot = chatbot
        self.builder = builder
        self.recognizer = recognizer
        self.anchors = anchors
        self.config = config or AgentConfig()
        self.commands = VoiceCommands(self.config.voice_commands)
        self.interaction = InteractionStateMachine(self.config.interaction)
        self.executor = executor or _InlineExecutor()
        self.clock = clock
        self.conversation_object: str | None = None
        self.gaze = Gaze()
        self.jobs: dict[str, Job] = {}
        self._outbox: list[dict[str, Any]] = []
        self._events: list[str] = []
        self._ids = itertools.count(1)
        self._fillers = itertools.cycle(self.config.fillers or ("Let me see.",))
        self._generation = 0
        self._lock = threading.RLock()

    # -- state ------------------------------------------------------------------------------------------------------------
    @property
    def state(self) -> InteractionState:
        return self.interaction.state

    def snapshot(self) -> dict[str, Any]:
        return {"state": self.state.value, "object": self.conversation_object,
                "room": self.anchors.loaded_room if self.anchors else None,
                "anchors": sorted(self.anchors.loaded) if self.anchors else []}

    def _end_conversation(self) -> None:
        self._generation += 1
        self.conversation_object = None
        self.chatbot.reset()

    def tick(self, gaze: Gaze, user_speaking: bool = False, now: float | None = None) -> dict[str, Any]:
        """One gaze/voice-activity sample. Returns state, events and utterances the agent should speak."""
        with self._lock:
            now = self.clock() if now is None else now
            self.gaze = gaze
            events = self.interaction.update(now, gaze.on_character, user_speaking)
            if "conversation_ended" in events:
                self._end_conversation()
            if "proactive_greeting" in events:
                greeting = self._compose(self.config.greeting, None, kind="greeting")
                self.interaction.begin_response(now, estimate_speech_seconds(greeting["text"]))
                self._outbox.append(greeting)
            events = self._events + events
            self._events = []
            speak, self._outbox = self._outbox, []
            return {**self.snapshot(), "events": events, "speak": speak}

    def response_finished(self, now: float | None = None) -> dict[str, Any]:
        with self._lock:
            self.interaction.finish_response(self.clock() if now is None else now)
            return self.snapshot()

    def reset(self) -> dict[str, Any]:
        """New visit: end the conversation and unload anchors (to replay anchor loading)."""
        with self._lock:
            self.interaction.reset()
            self._end_conversation()
            if self.anchors:
                self.anchors.unload()
            self._outbox, self._events = [], []
            return self.snapshot()

    # -- utterances -------------------------------------------------------------------------------------------------------
    def submit(self, text: str, gaze: Gaze | None = None, frame: Any = None, now: float | None = None) -> dict[str, Any]:
        """Accept a recognised utterance. Returns at once with a job id and a thinking cue."""
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            raise ValueError("Utterance must be 1-1000 characters")
        command = self.commands.match(text)
        with self._lock:
            now = self.clock() if now is None else now
            gaze = gaze or self.gaze
            accepted, reason = self.interaction.accepts(command is not None)
            if not accepted:
                hint = ("Look at the character for a moment to start talking, or say a voice command such as "
                        f"\"{self.commands.phrases[0]}\" while looking at an object." if reason == "not_listening"
                        else "The agent is still answering.")
                return {**self.snapshot(), "accepted": False, "reason": reason, "message": hint}
            self.interaction.submit_utterance(now, command is not None)
            target = gaze.target_id if gaze.kind == "object" else None
            anchored = bool(command and self.anchors and self.anchors.resolve(target))
            recognition_needed = bool(command and not anchored)
            job = Job(f"job-{next(self._ids)}", text.strip(), command, gaze, frame, self._generation, now)
            self.jobs[job.job_id] = job
            thinking = {"animation": "think", "filler": next(self._fillers),
                        "filler_delay_seconds": 0.0 if recognition_needed else self.config.filler_delay_seconds,
                        "recognition": "anchor" if anchored else ("pending" if recognition_needed else None)}
        self.executor.submit(self._run, job)
        with self._lock:
            return {**self.snapshot(), "accepted": True, "job": job.job_id, "command": command, "thinking": thinking}

    def job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown job: {job_id}")
            if not job.done:
                return {**self.snapshot(), "job": job_id, "done": False}
            return {**self.snapshot(), "job": job_id, "done": True, "reply": job.result}

    def ask(self, text: str, gaze: Gaze | None = None, frame: Any = None) -> dict[str, Any]:
        """Synchronous convenience for command-line adapters: open listening, answer, return the reply."""
        with self._lock:
            now = self.clock()
            if self.state == InteractionState.RESPONDING:
                self.interaction.finish_response(now)
            self.interaction.start_listening(now)
        submitted = self.submit(text, gaze, frame)
        if not submitted.get("accepted"):
            raise RuntimeError(submitted.get("message", "Utterance was not accepted"))
        done = threading.Event()
        while not done.wait(0.02):
            if self.jobs[submitted["job"]].done:
                done.set()
        reply = self.jobs[submitted["job"]].result
        self.response_finished()
        return reply

    def _run(self, job: Job) -> None:
        try:
            result = self._answer(job)
        except Exception as exc:  # never leave the agent stuck in THINKING
            result = self._compose("Sorry, something went wrong while I was thinking. Could you ask again?",
                                   self.conversation_object, kind="error", extra={"error": str(exc)})
        with self._lock:
            job.result, job.done = result, True
            if job.generation == self._generation and self.state == InteractionState.THINKING:
                self.interaction.begin_response(self.clock(), estimate_speech_seconds(result["text"]))
                self._events.append("response_ready")
            self._events.extend(job.events)

    def _answer(self, job: Job) -> dict[str, Any]:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        recognition: dict[str, Any] | None = None
        with self._lock:
            obj = self.conversation_object
        if job.command:
            target = job.gaze.target_id if job.gaze.kind == "object" else None
            anchor = self.anchors.resolve(target) if self.anchors else None
            if anchor is not None:
                obj = anchor.object_label
                recognition = {"source": "anchor", "skipped": True, "anchor_id": anchor.anchor_id, "room": anchor.room_fingerprint}
            elif target is None and job.frame is None:
                recognition = {"source": None, "skipped": False, "error": "no gazed object"}
                return self._compose("I'm not sure which object you mean. Look at it and ask me again.", obj,
                                     kind="answer", job=job, recognition=recognition, timings=timings, started=started)
            elif self.recognizer is None:
                recognition = {"source": None, "skipped": False, "error": "no recogniser configured"}
                return self._compose("I can't recognise objects right now.", obj, kind="answer", job=job,
                                     recognition=recognition, timings=timings, started=started)
            else:
                t0 = time.perf_counter()
                try:
                    found = self.recognizer.recognize(target, job.frame)
                except RecognitionError as exc:
                    timings["recognition_s"] = round(time.perf_counter() - t0, 3)
                    recognition = {"source": None, "skipped": False, "error": str(exc)}
                    return self._compose(f"Hmm, I couldn't recognise it. {exc}", obj, kind="answer", job=job,
                                         recognition=recognition, timings=timings, started=started)
                timings["recognition_s"] = round(time.perf_counter() - t0, 3)
                obj = found.label
                recognition = {"source": found.source, "skipped": False, "label": found.label,
                               "confidence": round(found.confidence, 3)}
                if self.anchors is not None:
                    room = self.anchors.observe_recognition(found.label)
                    if room is not None:
                        recognition["anchors_loaded"] = room
                        job.events.append("anchors_loaded")
            with self._lock:
                if job.generation == self._generation:
                    self.conversation_object = obj
        t0 = time.perf_counter()
        reply = self.chatbot.ask(job.text, obj)
        timings["chatbot_s"] = round(time.perf_counter() - t0, 3)
        return self._compose(reply.text, obj, kind="answer", job=job, recognition=recognition, timings=timings,
                             started=started, chat=reply)

    def _compose(self, text: str, obj: str | None, *, kind: str, job: Job | None = None,
                 recognition: dict[str, Any] | None = None, timings: dict[str, float] | None = None,
                 started: float | None = None, chat: Any = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        sentiment = chat.sentiment if chat is not None else self.chatbot.sentiment.analyze(text)
        t0 = time.perf_counter()
        plan = self.builder.plan(text) if self.builder else {"entries": [], "library_ready": False}
        timings = dict(timings or {})
        timings["animation_s"] = round(time.perf_counter() - t0, 4)
        if started is not None:
            timings["total_s"] = round(time.perf_counter() - started, 3)
        events = [{"channel": "speech", "value": text, "start_s": 0.0},
                  {"channel": "expression", "value": sentiment.expression["name"], "level": sentiment.expression["level"], "start_s": 0.0},
                  *({"channel": "animation", "value": e["clip_id"] if e["kind"] == "clip" else e["gesture"],
                     "trigger": e["trigger"], "start_s": e["start_s"]} for e in plan["entries"]),
                  {"channel": "viseme", "value": "derive_from_speech_text", "start_s": 0.0}]
        result = {"kind": kind, "text": text, "object": obj, "query": job.text if job else None,
                  "command": job.command if job else None, "recognition": recognition,
                  "query_type": _query_type(job, recognition), "sentiment": sentiment.to_dict(),
                  "expression": sentiment.expression, "animation": plan, "events": events, "timings": timings,
                  "chatbot": chat.source if chat is not None else None,
                  "chatbot_error": chat.error if chat is not None else None}
        result.update(extra or {})
        return result


def _query_type(job: Job | None, recognition: dict[str, Any] | None) -> str | None:
    """Paper Table 1 categories: A (recognition loads the anchor map), B (no recognition), C (recognition)."""
    if job is None:
        return None
    if recognition is None or recognition.get("skipped"):
        return "B" if recognition is None else "anchored"
    if recognition.get("error"):
        return None
    return "A" if recognition.get("anchors_loaded") else "C"
