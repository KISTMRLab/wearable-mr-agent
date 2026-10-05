import threading
from concurrent.futures import ThreadPoolExecutor

from wearable_mr.anchors import AnchorManager, InMemoryAnchorProvider, SpatialAnchor
from wearable_mr.animation import AnimationBuilder, AnimationTable
from wearable_mr.chatbot import Chatbot
from wearable_mr.factory import build_runtime, default_args
from wearable_mr.knowledge import DomainKnowledge
from wearable_mr.recognition import Recognition, RecognitionError, SimulatedRecognizer
from wearable_mr.runtime import AgentConfig, AgentRuntime
from wearable_mr.types import Gaze

CHARACTER, NOWHERE = Gaze("character"), Gaze("none")
KNOWLEDGE = DomainKnowledge({
    "rose": {"overview": "This is a rose.", "topics": {"care": "Water roses at the base."}},
    "tulip": {"overview": "This is a tulip.", "topics": {}},
    "lily": {"overview": "This is a lily.", "topics": {"pets": "Warning: lilies are toxic to cats."}},
})


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_runtime(recognizer=None, anchors=None, executor=None, **config):
    clock = Clock()
    table = AnimationTable.from_dict({"phrases": {"this is": "point"}, "words": {"hello": "wave"}})
    runtime = AgentRuntime(Chatbot(KNOWLEDGE), builder=AnimationBuilder(table), recognizer=recognizer, anchors=anchors,
                           config=AgentConfig(**config), executor=executor, clock=clock)
    return runtime, clock


def start_listening(runtime, clock):
    runtime.tick(CHARACTER)
    clock.t += runtime.config.dwell_seconds
    assert "listening_started" in runtime.tick(CHARACTER)["events"]


def test_proactive_greeting_after_silent_dwell_with_sentiment_and_animation():
    runtime, clock = make_runtime()
    start_listening(runtime, clock)
    clock.t += 3.0
    result = runtime.tick(CHARACTER)
    assert result["events"] == ["proactive_greeting"] and result["state"] == "responding"
    greeting = result["speak"][0]
    assert greeting["text"] == "Hello, do you need help?" and greeting["kind"] == "greeting"
    assert greeting["expression"] == {"name": "happiness", "level": 1}
    assert greeting["animation"]["entries"][0]["gesture"] == "wave"
    runtime.response_finished()
    assert runtime.state.value == "listening"


def test_voice_command_recognises_the_gazed_object_and_follow_ups_do_not_need_it():
    recognizer = SimulatedRecognizer({"prop-7": "rose"})
    runtime, clock = make_runtime(recognizer)
    start_listening(runtime, clock)
    submitted = runtime.submit("What is this?", Gaze("object", "prop-7"))
    assert submitted["accepted"] and submitted["command"] == "what is this"
    assert submitted["thinking"]["filler"] and submitted["thinking"]["filler_delay_seconds"] == 0.0
    reply = runtime.job(submitted["job"])["reply"]
    assert reply["text"] == "This is a rose." and reply["recognition"]["source"] == "simulated"
    assert reply["query_type"] == "C" and recognizer.calls == 1
    runtime.response_finished()
    # Follow-up while looking at the character: no recognition, same object.
    follow = runtime.job(runtime.submit("How do I care for it?", CHARACTER)["job"])["reply"]
    assert follow["text"] == "Water roses at the base." and follow["object"] == "rose" and follow["recognition"] is None
    assert recognizer.calls == 1 and follow["query_type"] == "B"
    runtime.response_finished()
    general = runtime.job(runtime.submit("Who are you?", CHARACTER)["job"])["reply"]
    assert "virtual guide" in general["text"]


def test_general_conversation_without_any_object():
    runtime, clock = make_runtime(SimulatedRecognizer({}))
    start_listening(runtime, clock)
    reply = runtime.job(runtime.submit("Hello!", CHARACTER)["job"])["reply"]
    assert reply["object"] is None and reply["kind"] == "answer" and reply["recognition"] is None
    assert reply["sentiment"]["class"] == "Joy"


def test_plain_speech_needs_dwell_but_a_voice_command_starts_a_conversation():
    runtime, clock = make_runtime(SimulatedRecognizer({"p": "lily"}))
    rejected = runtime.submit("tell me a story", NOWHERE)
    assert not rejected["accepted"] and rejected["reason"] == "not_listening"
    reply = runtime.job(runtime.submit("Tell me about this flower", Gaze("object", "p"))["job"])["reply"]
    assert reply["object"] == "lily" and runtime.state.value == "responding"
    runtime.response_finished()
    pets = runtime.job(runtime.submit("Is it safe for pets?", Gaze("object", "p"))["job"])["reply"]
    assert pets["recognition"] is None and pets["sentiment"]["class"] == "Fear"
    assert pets["expression"]["name"] == "fear"


def test_looking_away_silently_ends_the_conversation_and_clears_context():
    runtime, clock = make_runtime(SimulatedRecognizer({"p": "rose"}))
    start_listening(runtime, clock)
    runtime.job(runtime.submit("what's this", Gaze("object", "p"))["job"])
    runtime.response_finished()
    clock.t += 0.5
    assert "conversation_ended" not in runtime.tick(Gaze("object", "p"), user_speaking=True)["events"]
    clock.t += 5.0
    assert runtime.tick(Gaze("object", "p"), user_speaking=False)["events"] == []
    clock.t += 1.5
    ended = runtime.tick(Gaze("object", "p"))
    assert ended["events"] == ["conversation_ended"] and ended["object"] is None


def test_loaded_anchors_skip_recognition():
    provider = InMemoryAnchorProvider([SpatialAnchor("r1-rose", "room-1", "rose", [0, 0, 0]),
                                       SpatialAnchor("r1-tulip", "room-1", "tulip", [0, 0, 0]),
                                       SpatialAnchor("r2-lily", "room-2", "lily", [0, 0, 0])])
    recognizer = SimulatedRecognizer({"r1-rose": "rose", "r1-tulip": "tulip", "r2-lily": "lily"})
    runtime, clock = make_runtime(recognizer, AnchorManager(provider))
    start_listening(runtime, clock)
    first = runtime.job(runtime.submit("what is this", Gaze("object", "r1-rose"))["job"])
    assert first["reply"]["query_type"] == "A" and first["reply"]["recognition"]["anchors_loaded"] == "room-1"
    assert "anchors_loaded" in runtime.tick(CHARACTER)["events"]
    runtime.response_finished()
    submitted = runtime.submit("what is this", Gaze("object", "r1-tulip"))
    assert submitted["thinking"]["recognition"] == "anchor"
    anchored = runtime.job(submitted["job"])["reply"]
    assert anchored["recognition"]["skipped"] and anchored["object"] == "tulip" and recognizer.calls == 1
    runtime.response_finished()
    other_room = runtime.job(runtime.submit("what is this", Gaze("object", "r2-lily"))["job"])["reply"]
    assert other_room["recognition"]["anchors_loaded"] == "room-2" and recognizer.calls == 2


def test_thinking_state_is_asynchronous_while_recognition_is_pending():
    release = threading.Event()

    class SlowRecognizer:
        def recognize(self, target_id, frame):
            release.wait(5)
            return Recognition("rose", .9, "slow")

    with ThreadPoolExecutor(1) as pool:
        runtime, clock = make_runtime(SlowRecognizer(), executor=pool)
        start_listening(runtime, clock)
        submitted = runtime.submit("what is this", Gaze("object", "x"))
        assert submitted["state"] == "thinking" and submitted["thinking"]["animation"] == "think"
        assert runtime.job(submitted["job"]) == {**runtime.snapshot(), "job": submitted["job"], "done": False}
        clock.t += 10
        assert runtime.tick(NOWHERE)["state"] == "thinking"  # looking away while it thinks is fine
        assert not runtime.submit("hello", CHARACTER)["accepted"]
        release.set()
        for _ in range(200):
            if runtime.job(submitted["job"])["done"]:
                break
            threading.Event().wait(.01)
        assert runtime.job(submitted["job"])["reply"]["object"] == "rose"
        assert "response_ready" in runtime.tick(CHARACTER)["events"]


def test_recognition_failure_is_spoken_not_raised():
    class Broken:
        def recognize(self, target_id, frame):
            raise RecognitionError("No object detector is configured.")

    runtime, clock = make_runtime(Broken())
    start_listening(runtime, clock)
    reply = runtime.job(runtime.submit("what is this", Gaze("object", "x"))["job"])["reply"]
    assert "couldn't recognise" in reply["text"] and reply["recognition"]["error"]


def test_bundled_scenario_runtime_answers_with_the_paper_contract():
    runtime = build_runtime(default_args(bank="", recognition_latency=0.0))
    reply = runtime.ask("tell me about this", Gaze("object", "r1-lily"))
    assert reply["object"] == "lily" and reply["recognition"]["anchors_loaded"] == "room-1"
    assert set(reply["sentiment"]) >= {"class", "level", "expression"}
    assert {e["channel"] for e in reply["events"]} >= {"speech", "expression", "viseme"}
