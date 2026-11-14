from emotion_chatbot import NEUTRAL, Prediction, RuleBasedClassifier
from emotion_chatbot.chat import NEGATED_RESPONSES, RESPONSES, respond, run_chat


def test_every_emotion_has_both_responses():
    from emotion_chatbot import EMOTIONS

    assert set(EMOTIONS) <= set(RESPONSES)
    assert set(EMOTIONS) == set(NEGATED_RESPONSES)


def test_respond_uses_negated_reply_when_only_negation_found():
    assert respond(Prediction(NEUTRAL, frozenset(), ("sadness",))) == NEGATED_RESPONSES["sadness"]


def test_respond_prefers_real_label():
    assert respond(Prediction("joy", frozenset({"joy"}), ("sadness",))) == RESPONSES["joy"]


def test_chat_loop_runs_and_exits():
    inputs = iter(["Ich bin glücklich", "", "Ich bin nicht traurig", "EXIT"])
    out: list[str] = []
    run_chat(RuleBasedClassifier(), input_fn=lambda _: next(inputs), output_fn=out.append)
    assert any("(joy)" in line for line in out)
    assert any(NEGATED_RESPONSES["sadness"] in line for line in out)
    assert out[-1].startswith("Bot: Bis zum")


def test_chat_loop_survives_eof():
    def raise_eof(_):
        raise EOFError

    out: list[str] = []
    run_chat(RuleBasedClassifier(), input_fn=raise_eof, output_fn=out.append)
    assert out[-1].startswith("Bot: Bis zum")
