import json
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
    assert all(library.resolve(cid) for cid in bundled.clip_ids)
    assert json.loads(json.dumps(AnimationBuilder(bundled, library).plan("I think we can walk.")))["motion"]["clips"]
