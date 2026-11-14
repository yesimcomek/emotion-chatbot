import pytest

from emotion_chatbot import NEUTRAL, RuleBasedClassifier

clf = RuleBasedClassifier()


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Ich bin so glücklich heute!", "joy"),
        ("Das ist wunderbar", "joy"),
        ("Ich bin wütend auf ihn", "anger"),
        ("Das nervt mich total", "anger"),
        ("Ich fühle mich traurig und einsam", "sadness"),
        ("Ich habe Angst vor der Prüfung", "fear"),
        ("Ich bin nervös", "fear"),
        ("Ich liebe dieses Lied", "joy"),
        ("Das ist ekelhaft", "disgust"),
        ("Wow, das ist unglaublich!", "surprise"),
        ("Heute ist Dienstag", NEUTRAL),
        ("", NEUTRAL),
    ],
)
def test_basic_emotions(text, label):
    assert clf.predict(text) == label


def test_case_insensitive():
    assert clf.predict("ICH BIN WÜTEND") == "anger"


def test_inflected_forms():
    assert clf.predict("Eine traurige Geschichte") == "sadness"
    assert clf.predict("Das ärgert mich") == "anger"


def test_no_substring_false_positive():
    # "gut" must not match inside "Gutachten"
    assert clf.predict("Das Gutachten liegt vor") == NEUTRAL


@pytest.mark.parametrize(
    ("text", "negated"),
    [
        ("Ich bin nicht glücklich", "joy"),
        ("Ich bin nicht sehr glücklich", "joy"),
        ("Ich bin nicht traurig", "sadness"),
        ("Ich habe keine Angst", "fear"),
        ("Er ging ohne Angst hinein", "fear"),
    ],
)
def test_negation_is_detected_and_not_counted(text, negated):
    prediction = clf.analyze(text)
    assert prediction.label == NEUTRAL
    assert prediction.labels == frozenset()
    assert negated in prediction.negated


def test_negation_does_not_cross_clauses():
    # The comma ends the clause, so "traurig" is not negated by the earlier "nicht".
    prediction = clf.analyze("Ich bin nicht wütend, ich bin traurig")
    assert prediction.label == "sadness"
    assert prediction.negated == ("anger",)


def test_negation_does_not_cross_contrast_word():
    assert clf.predict("Nicht gut aber ich bin wütend") == "anger"


def test_majority_by_weight_wins():
    assert clf.predict("Ich bin glücklich und froh, aber auch etwas besorgt") == "joy"


def test_multi_label():
    labels = clf.predict_labels("Ich bin glücklich, aber auch besorgt")
    assert labels == frozenset({"joy", "fear"})


def test_tie_is_deterministic():
    text = "Ich bin glücklich und traurig"
    assert len({clf.predict(text) for _ in range(5)}) == 1
