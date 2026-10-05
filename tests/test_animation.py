import json
import re
from pathlib import Path

import pytest

from wearable_mr.animation import AnimationBuilder, AnimationTable, ClipLibrary

ROOT = Path(__file__).resolve().parents[1]


def table(**extra):
    return AnimationTable.from_dict({"phrases": {"i think": {"clip": "beat_08", "procedural": "beat"},
                                                 "this is a": {"procedural": "point"}, "this is": {"procedural": "open"}},
                                     "words": {"i": {"clip": "beat_06"}, "this": {"procedural": "point"},
                                               "we": {"clip": "beat_03", "procedural": "open"}, "hello": "wave"},
                                     **extra})


def bank(ids):
    return {"fps": 30, "joint_names": ["Hips"], "axisSigns": [1, 1, 1],
            "clips": [{"id": cid, "text": cid, "positions": [[[0, 0, 0]]] * 30} for cid in ids]}


def test_animations_follow_text_order_not_trigger_order():
    # Regression: the old trigger list checked "this" before "i" for the whole phrase.
    entries = AnimationBuilder(table(min_gap_words=0)).build("Well, I really like this.")
    assert [(e.trigger, e.kind) for e in entries] == [("i", "clip"), ("this", "procedural")]
    assert entries[0].start_char < entries[1].start_char and entries[0].end_char == entries[1].start_char
    assert entries[0].start_s < entries[1].start_s


def test_longest_phrase_wins_then_words_fill_unclaimed_text():
    entries = AnimationBuilder(table(min_gap_words=0)).build("This is a garden and I think we love it.")
    assert [(e.trigger, e.match) for e in entries] == [("this is a", "phrase"), ("i think", "phrase"), ("we", "word")]
    assert entries[0].gesture == "point"  # not the shorter "this is" or the word "this"
    assert entries[1].clip_id == "beat_08"


def test_word_triggers_keep_a_minimum_gap_but_phrases_always_play():
    text = "I think we can, and we will."
    entries = AnimationBuilder(table(min_gap_words=3)).build(text)
    assert [e.trigger for e in entries] == ["i think", "we"]  # first "we" is too close to the phrase
    assert entries[1].start_word == 5


def test_clip_ids_resolve_against_the_prepared_bank_with_ordinal_and_procedural_fallbacks():
    library = ClipLibrary(bank(["take:0-75", "take:75-150", "take:150-225"]))
    builder = AnimationBuilder(table(min_gap_words=0), library)
    entries = builder.build("Hello, we know I can.")
    by_trigger = {e.trigger: e for e in entries}
    assert by_trigger["we"].kind == "clip" and by_trigger["we"].clip_id == "take:150-225"  # beat_03 -> third clip
    assert "i" not in by_trigger  # beat_06 is absent and the entry has no procedural alternative
    assert by_trigger["hello"].kind == "procedural" and by_trigger["hello"].gesture == "wave"
    plan = builder.plan("we")
    assert list(plan["motion"]["clips"]) == ["take:150-225"] and plan["motion"]["fps"] == 30


def windowed_bank():
    """Shared bank shape: ordinal ids, stable window ids in source (or only take/start/end in older banks)."""
    data = bank(["beat_01", "beat_02", "beat_03"])
    data["clips"][0]["source"] = {"take": "1_wayne_0_1_1", "window_id": "1_wayne_0_1_1:1830-1905",
                                  "start_frame": 1830, "end_frame": 1905}
    data["clips"][1]["source"] = {"take": "1_wayne_0_1_1", "window_id": "1_wayne_0_1_1:1005-1080",
                                  "start_frame": 1005, "end_frame": 1080}
    data["clips"][2]["source"] = {"take": "1_wayne_0_1_1", "start_frame": 780, "end_frame": 855}
    return data


def test_window_ids_resolve_regardless_of_bank_order_and_missing_windows_fall_back_to_procedural():
    library = ClipLibrary(windowed_bank())
    assert library.resolve("1_wayne_0_1_1:1830-1905") == "beat_01"
    assert library.resolve("1_wayne_0_1_1:780-855") == "beat_03"  # derived from take/start/end
    assert library.resolve("1_wayne_0_1_1:480-555") is None  # never an unrelated ordinal clip
    assert library.resolve("beat_02") == "beat_02"
    shifted = windowed_bank()
    shifted["clips"].reverse()
    for k, clip in enumerate(shifted["clips"]):
        clip["id"] = f"beat_{k + 1:02d}"
    assert ClipLibrary(shifted).resolve("1_wayne_0_1_1:1830-1905") == "beat_03"  # same window after a rebuild
    words = {"i": {"clip": "1_wayne_0_1_1:1005-1080", "procedural": "beat"},
             "you": {"clip": "1_wayne_0_1_1:480-555", "procedural": "open"}}
    entries = AnimationBuilder(AnimationTable.from_dict({"words": words, "min_gap_words": 0}), library).build("I see you")
    assert [(e.trigger, e.kind, e.clip_id, e.gesture) for e in entries] == [
        ("i", "clip", "beat_02", "beat"), ("you", "procedural", None, "open")]


def test_bundled_table_uses_stable_window_ids_with_procedural_fallbacks():
    payload = json.loads((ROOT / "demo" / "animation-table.json").read_text(encoding="utf-8"))
    specs = list(payload["phrases"].values()) + list(payload["words"].values())
    clips = [spec for spec in specs if isinstance(spec, dict) and spec.get("clip")]
    assert clips and all(re.fullmatch(r"\d+_[a-z]+_\d+_\d+_\d+:\d+-\d+", spec["clip"]) for spec in clips)
    assert all(spec.get("procedural") for spec in clips)
    for spec in clips:  # the description quotes each referenced window
        assert spec["clip"].split(":")[1] in payload["description"]


def test_table_validation_and_the_bundled_table_loads():
    with pytest.raises(ValueError):
        AnimationTable.from_dict({"phrases": {"single": "wave"}})
    with pytest.raises(ValueError):
        AnimationTable.from_dict({"words": {"hi": {"procedural": "moonwalk"}}})
    bundled = AnimationTable.load(ROOT / "demo" / "animation-table.json")
    assert len(bundled.phrases) + len(bundled.words) >= 30
    entries = AnimationBuilder(bundled).build("Hello, do you need help? I think you can look at this rose.")
    assert [e.trigger for e in entries][:3] == ["hello do you need help", "i think", "you can"]
    assert entries[0].gesture == "wave"
    assert all(e.end_s >= e.start_s for e in entries)


def test_prepared_bank_if_present_resolves_the_bundled_table():
    path = ROOT / "outputs" / "beat-library" / "bank.json"
    if not path.is_file():
        pytest.skip("BEAT bank not prepared")
    library = ClipLibrary.load(path)
    bundled = AnimationTable.load(ROOT / "demo" / "animation-table.json")
    # Both default banks (public multi-take and processed BEAT) hold these reviewed windows of the named take;
    # each transcript contains the trigger it serves.
    core = {"1_wayne_0_1_1:1830-1905": "i think", "1_wayne_0_1_1:1080-1155": "i can", "1_wayne_0_1_1:780-855": "walk",
            "1_wayne_0_1_1:1230-1305": "how much", "1_wayne_0_1_1:1005-1080": "friends",
            "1_wayne_0_1_1:1905-1980": "anime", "1_wayne_0_1_1:1680-1755": "movies", "1_wayne_0_1_1:1755-1830": "sleep"}
    for window, phrase in core.items():
        if window.split(":")[0] not in {(c.get("source") or {}).get("take") for c in library.clips.values()}:
            pytest.skip("prepared bank does not use the default named take")
        assert window in bundled.clip_ids
        assert phrase in library.clips[library.resolve(window)]["text"]
    plan = json.loads(json.dumps(AnimationBuilder(bundled, library).plan("I think we can walk.")))
    assert plan["motion"]["clips"]
    assert all(plan["motion"]["clips"][e["clip_id"]]["source"]["start_frame"] == 1830
               for e in plan["entries"] if e["trigger"] == "i think")
