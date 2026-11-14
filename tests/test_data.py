import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from emotion_chatbot.data import VAL_FROM_TRAIN, load_split, load_splits  # noqa: E402

COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def _write(path, rows):
    """rows: (id, text, set_of_labels)"""
    table = {
        "id": [r[0] for r in rows],
        "text": [r[1] for r in rows],
        **{c: [int(c in r[2]) for r in rows] for c in COLS},
    }
    pq.write_table(pa.table(table), path)


@pytest.fixture
def cache(tmp_path):
    train = [(f"deu_train_track_a_{i:05d}", f"text {i}", {"joy"} if i % 2 else set())
             for i in range(400)]
    dev = [(f"deu_dev_track_{t}_{i:05d}", f"dev {i}", {"fear"}) for t in "ac" for i in range(10)]
    test = [(f"deu_test_track_{t}_{i:05d}", f"test {i}", {"anger", "disgust"})
            for t in "ac" for i in range(20)] + [("deu_test_track_a_99999", "test 0", set())]
    for name, rows in [("train", train), ("dev", dev), ("test", test)]:
        _write(tmp_path / f"{name}.parquet", rows)
    return tmp_path


def test_track_c_and_duplicates_are_dropped(cache):
    dev = load_split("dev", cache)
    assert len(dev) == 10  # track C copy removed
    test = load_split("test", cache)
    assert len(test) == 20  # track C copy and the extra duplicate text removed
    assert len(set(test.texts)) == len(test)


def test_labels_are_frozensets_of_emotions(cache):
    test = load_split("test", cache)
    assert test.labels[0] == frozenset({"anger", "disgust"})


def test_splits_are_disjoint_and_deterministic(cache):
    train, val, test = load_splits(cache)
    assert len(val) == 10 + VAL_FROM_TRAIN and len(train) == 400 - VAL_FROM_TRAIN
    assert not set(train.texts) & set(val.texts)
    assert load_splits(cache)[1].texts == val.texts
