from wearable_mr.anchors import AnchorManager, InMemoryAnchorProvider, JsonAnchorProvider, SpatialAnchor
from wearable_mr.gaze import DetectionGazeTracker, select_detection
from wearable_mr.types import BoundingBox, Detection, Gaze


def test_pointer_prefers_tighter_box():
    detections = [
        Detection("plant", .9, BoundingBox(0, 0, 100, 100)),
        Detection("rose", .8, BoundingBox(25, 25, 75, 75)),
    ]
    assert select_detection(detections, 50, 50).label == "rose"
    assert select_detection(detections, 500, 500) is None


def test_anchors_are_room_scoped(tmp_path):
    store = JsonAnchorProvider(tmp_path / "anchors.json")
    store.save(SpatialAnchor("a", "room-one", "rose", [1.0] * 16))
    store.save(SpatialAnchor("b", "room-two", "lamp", [2.0] * 16))
    assert [item.anchor_id for item in store.for_room("room-one")] == ["a"]
    assert {item.anchor_id for item in store.all()} == {"a", "b"}


def test_recognised_label_selects_room_anchors_and_ambiguity_waits_for_more_evidence():
    manager = AnchorManager(InMemoryAnchorProvider([
        SpatialAnchor("1-rose", "room-1", "rose", [0, 0, 0]), SpatialAnchor("1-fern", "room-1", "fern", [0, 0, 0]),
        SpatialAnchor("2-rose", "room-2", "rose", [0, 0, 0]), SpatialAnchor("2-lily", "room-2", "lily", [0, 0, 0]),
    ]))
    assert manager.resolve("1-fern") is None  # nothing loaded before recognition
    assert manager.observe_recognition("rose") is None  # identical rooms: rose is in both
    assert manager.loaded_room is None
    assert manager.observe_recognition("lily") == "room-2"
    assert manager.resolve("2-rose").object_label == "rose"
    assert manager.observe_recognition("rose") is None  # already in a room containing rose
    assert manager.observe_recognition("fern") == "room-1"
    assert sorted(manager.loaded) == ["1-fern", "1-rose"]


def test_gaze_tracker_keeps_same_untracked_box_across_detector_jitter():
    tracker = DetectionGazeTracker()
    first = tracker.update([Detection("rose", .9, BoundingBox(10, 10, 50, 50))], 25, 25)
    second = tracker.update([Detection("rose", .9, BoundingBox(11, 10, 51, 50))], 25, 25)
    assert first == second and first.kind == "object"
    assert tracker.update([], 25, 25) == Gaze("none")
