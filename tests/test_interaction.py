import pytest

from wearable_mr.interaction import InteractionConfig, InteractionState, InteractionStateMachine


def listening(config=None, at=0.0):
    machine = InteractionStateMachine(config or InteractionConfig())
    machine.update(at, True)
    machine.update(at + machine.config.dwell_seconds, True)
    assert machine.state == InteractionState.LISTENING
    return machine


def test_dwell_on_the_character_for_four_seconds_starts_listening():
    machine = InteractionStateMachine()
    assert machine.update(10.0, on_character=True) == ["dwell_started"]
    assert machine.update(13.99, True) == []
    assert machine.state == InteractionState.DWELLING
    assert machine.update(14.0, True) == ["listening_started"]
    assert machine.state == InteractionState.LISTENING


def test_dwell_is_configurable_and_resets_when_gaze_leaves_the_character():
    machine = InteractionStateMachine(InteractionConfig(dwell_seconds=2.0))
    machine.update(0.0, True)
    assert machine.update(1.5, False) == ["dwell_cancelled"]
    assert machine.state == InteractionState.IDLE
    machine.update(2.0, True)
    assert machine.update(3.9, True) == []
    assert machine.update(4.0, True) == ["listening_started"]


def test_silence_after_dwell_triggers_one_proactive_greeting():
    machine = listening(at=0.0)  # listening from t=4
    assert machine.update(6.9, True) == []
    assert machine.update(7.0, True) == ["proactive_greeting"]
    machine.begin_response(7.0, 2.0)
    machine.finish_response(9.0)
    assert machine.update(20.0, True) == []  # greets once per conversation


def test_user_speech_postpones_and_an_utterance_prevents_the_greeting():
    machine = listening(at=0.0)
    machine.update(6.0, True, user_speaking=True)
    assert machine.update(8.5, True) == []  # only 2.5 s of silence since speech
    machine.submit_utterance(8.6)
    machine.begin_response(9.0)
    machine.finish_response(10.0)
    assert machine.update(30.0, True) == []


def test_looking_away_while_speaking_keeps_the_conversation():
    machine = listening(at=0.0)
    for t in (5.0, 6.0, 7.0, 8.0):
        assert machine.update(t, on_character=False, user_speaking=True) == []
    assert machine.state == InteractionState.LISTENING
    # Silence while still looking away ends it after the grace period.
    assert machine.update(8.5, False, False) == []
    assert machine.update(10.0, False, False) == ["conversation_ended"]
    assert machine.state == InteractionState.IDLE


def test_strict_paper_rule_ends_immediately_when_gaze_leaves_and_user_is_silent():
    machine = listening(InteractionConfig(lookaway_grace_seconds=0.0))
    assert machine.update(4.1, on_character=False) == ["conversation_ended"]


def test_looking_away_while_the_agent_thinks_or_answers_does_not_end_it():
    machine = listening(at=0.0)
    machine.submit_utterance(5.0)
    assert machine.update(9.0, False) == []
    machine.begin_response(9.0, 1.0)
    assert machine.update(10.0, False) == []
    machine.finish_response(10.0)
    assert machine.update(10.5, True) == []  # follow-up window while looking at the character
    assert machine.state == InteractionState.LISTENING


def test_voice_commands_can_open_a_conversation_but_plain_speech_needs_dwell():
    machine = InteractionStateMachine()
    with pytest.raises(RuntimeError):
        machine.submit_utterance(0.0, is_command=False)
    machine.submit_utterance(0.0, is_command=True)
    assert machine.state == InteractionState.THINKING
    strict = InteractionStateMachine(InteractionConfig(commands_start_conversation=False))
    assert strict.accepts(is_command=True) == (False, "not_listening")


def test_unacknowledged_response_times_out_back_to_listening():
    machine = listening(InteractionConfig(response_timeout_seconds=2.0))
    machine.submit_utterance(5.0)
    machine.begin_response(5.0, duration_seconds=3.0)
    assert machine.update(9.9, True) == []
    assert machine.update(10.0, True) == ["response_timeout"]
    assert machine.state == InteractionState.LISTENING
