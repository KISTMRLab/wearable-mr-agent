from wearable_mr.anchors import JsonAnchorProvider, SpatialAnchor
from wearable_mr.gaze import select_detection
from wearable_mr.interaction import InteractionState, InteractionStateMachine
from wearable_mr.knowledge import DomainKnowledge
from wearable_mr.runtime import AgentRuntime
from wearable_mr.types import BoundingBox, Detection


def test_dwell_requires_four_continuous_seconds():
    machine = InteractionStateMachine()
    machine.update_gaze("flower:1", 10.0)
    assert machine.update_gaze("flower:1", 13.99) == []
    assert machine.state == InteractionState.DWELLING
    assert machine.update_gaze("flower:1", 14.0) == ["listening_started"]
    assert machine.state == InteractionState.LISTENING
    assert machine.update_gaze(None, 14.1) == ["conversation_ended"]


def test_pointer_prefers_tighter_box_and_reply_is_object_grounded():
    detections = [
        Detection("plant", .9, BoundingBox(0, 0, 100, 100)),
        Detection("rose", .8, BoundingBox(25, 25, 75, 75)),
    ]
    assert select_detection(detections, 50, 50).label == "rose"
    knowledge = DomainKnowledge({"rose": {"overview": "Curator-authored rose description.", "emotion": "joy"}})
    reply = knowledge.answer("Tell me about this", "rose")
    assert reply.object_label == "rose"
    assert {event.channel for event in reply.events} == {"speech", "expression", "gesture", "viseme"}


def test_anchors_are_room_scoped(tmp_path):
    store = JsonAnchorProvider(tmp_path / "anchors.json")
    store.save(SpatialAnchor("a", "room-one", "rose", [1.0] * 16))
    store.save(SpatialAnchor("b", "room-two", "lamp", [2.0] * 16))
    assert [item.anchor_id for item in store.for_room("room-one")] == ["a"]


def test_runtime_keeps_same_untracked_box_across_detector_jitter():
    runtime = AgentRuntime(DomainKnowledge({"rose": {"overview": "Reviewed."}}))
    runtime.observe([Detection("rose", .9, BoundingBox(10, 10, 50, 50))], 25, 25, 0)
    events = runtime.observe([Detection("rose", .9, BoundingBox(11, 10, 51, 50))], 25, 25, 4)
    assert events == ["listening_started"]
