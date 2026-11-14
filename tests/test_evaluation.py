import pytest

np = pytest.importorskip("numpy")

from emotion_chatbot import EMOTIONS  # noqa: E402
from emotion_chatbot.baselines import NeutralBaseline, PriorBaseline  # noqa: E402
from emotion_chatbot.data import Dataset  # noqa: E402
from emotion_chatbot.evaluation import (  # noqa: E402
    bootstrap_ci,
    compute_metrics,
    evaluate,
    macro_f1,
    paired_bootstrap,
    to_matrix,
)
from emotion_chatbot.ml_base import tune_thresholds  # noqa: E402


def _mat(*rows):
    return to_matrix([frozenset(r) for r in rows])


def test_to_matrix_shape_and_order():
    m = _mat({"anger"}, set(), {"surprise", "anger"})
    assert m.shape == (3, len(EMOTIONS))
    assert m[0, EMOTIONS.index("anger")] and not m[1].any()
    assert m[2, EMOTIONS.index("surprise")]


def test_perfect_prediction():
    y = _mat({"joy"}, {"fear", "anger"}, set())
    assert compute_metrics(y, y.copy())["micro_f1"] == 1.0


def test_macro_f1_matches_hand_computation():
    # joy: tp=1 fp=1 fn=0 -> P=.5 R=1 F1=2/3 ; anger: tp=0 fn=1 -> F1=0 ; other 4 labels: F1=0
    y = _mat({"joy"}, {"anger"})
    p = _mat({"joy"}, {"joy"})
    assert macro_f1(y, p) == pytest.approx((2 / 3) / len(EMOTIONS))


def test_neutral_only_counts_as_negative():
    y = _mat(set(), set())
    m = compute_metrics(y, y.copy())
    assert m["exact_match"] == 1.0 and m["macro_f1"] == 0.0  # no positives -> F1 defined as 0


def test_bootstrap_ci_contains_point_estimate_and_is_seeded():
    rng = np.random.default_rng(1)
    y = rng.random((200, 6)) < 0.3
    p = y ^ (rng.random((200, 6)) < 0.15)
    lo, hi = bootstrap_ci(y, p, n_boot=200, seed=3)
    assert lo <= macro_f1(y, p) <= hi
    assert (lo, hi) == bootstrap_ci(y, p, n_boot=200, seed=3)


def test_paired_bootstrap_detects_clear_winner():
    rng = np.random.default_rng(2)
    y = rng.random((300, 6)) < 0.3
    good = y.copy()
    bad = ~y
    res = paired_bootstrap(y, good, bad, n_boot=100)
    assert res["diff"] > 0.5 and res["p_better"] == 1.0


def test_tune_thresholds_picks_separating_value():
    y = np.array([[1], [1], [0], [0]], dtype=bool)
    probs = np.array([[0.9], [0.4], [0.3], [0.1]])
    thr = tune_thresholds(y, probs)
    assert 0.3 < thr[0] <= 0.4  # separates {0.9, 0.4} from {0.3, 0.1}


def test_evaluate_runs_baselines():
    data = Dataset(["a", "b", "c", "d"], [frozenset({"joy"}), frozenset(), frozenset({"fear"}),
                                          frozenset({"joy", "anger"})])
    assert evaluate(NeutralBaseline(), data, n_boot=20)["macro_f1"] == 0.0
    prior = PriorBaseline(seed=1).fit(data)
    result = evaluate(prior, data, n_boot=20)
    assert result["n"] == 4 and 0.0 <= result["macro_f1"] <= 1.0
