"""Fine-tuned German transformer (default: deepset/gbert-base) for multi-label emotions.

Requires the ``bert`` extra (torch, transformers).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from .base import EMOTIONS
from .data import Dataset
from .evaluation import macro_f1, to_matrix
from .ml_base import ProbabilisticClassifier, tune_thresholds

DEFAULT_BASE_MODEL = "deepset/gbert-base"
MAX_LENGTH = 128  # median text is 37 tokens, 92.5% fit; keeps memory low on 8 GB machines


def _load_tokenizer(name_or_path):
    """AutoTokenizer, falling back to BertTokenizer for repos that omit ``tokenizer_class``
    (e.g. deepset/gbert-base ships only vocab.txt, which AutoTokenizer cannot resolve)."""
    try:
        return AutoTokenizer.from_pretrained(name_or_path)
    except ValueError:
        return BertTokenizer.from_pretrained(name_or_path)


def _load_config(name_or_path, **kwargs):
    """AutoConfig, falling back to BertConfig for repos without ``model_type`` in config.json
    (deepset/gbert-base predates that requirement)."""
    try:
        return AutoConfig.from_pretrained(name_or_path, **kwargs)
    except ValueError:
        return BertConfig.from_pretrained(name_or_path, **kwargs)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BertClassifier(ProbabilisticClassifier):
    def __init__(self, model, tokenizer, base_model: str = DEFAULT_BASE_MODEL, device=None) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device or pick_device()
        self.model.to(self.device).eval()
        self.name = "gbert" if base_model == DEFAULT_BASE_MODEL else base_model.split("/")[-1]

    @staticmethod
    def _new(base_model: str):
        tokenizer = _load_tokenizer(base_model)
        config = _load_config(
            base_model,
            num_labels=len(EMOTIONS),
            problem_type="multi_label_classification",
            id2label=dict(enumerate(EMOTIONS)),
            label2id={e: i for i, e in enumerate(EMOTIONS)},
        )
        model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config)
        return model, tokenizer

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.model.eval()
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # less padding
        for start in range(0, len(texts), batch_size):
            batch = [texts[i] for i in order[start : start + batch_size]]
            enc = self.tokenizer(
                batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
            ).to(self.device)
            out.append(torch.sigmoid(self.model(**enc).logits).float().cpu().numpy())
        probs = np.concatenate(out) if out else np.zeros((0, len(EMOTIONS)))
        restored = np.empty_like(probs)
        restored[order] = probs
        return restored

    @classmethod
    def train(
        cls,
        train: Dataset,
        val: Dataset,
        base_model: str = DEFAULT_BASE_MODEL,
        epochs: int = 5,
        lr: float = 3e-5,
        batch_size: int = 8,
        grad_accum: int = 2,
        seed: int = 42,
    ) -> BertClassifier:
        """Fine-tune on ``train``; keep the epoch with the best val macro-F1 (thresholds tuned).

        Effective batch size is ``batch_size * grad_accum`` (default 16).
        """
        set_seed(seed)
        model, tokenizer = cls._new(base_model)
        clf = cls(model, tokenizer, base_model)
        device = clf.device
        print(f"  device={device}  base={base_model}  epochs={epochs} lr={lr} "
              f"batch={batch_size}x{grad_accum}", flush=True)

        y_train = torch.tensor(to_matrix(train.labels), dtype=torch.float)
        y_val = to_matrix(val.labels)
        updates_per_epoch = -(-len(train) // (batch_size * grad_accum))
        steps = epochs * updates_per_epoch
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)
        rng = random.Random(seed)

        best_score, best_state, best_thr = -1.0, None, None
        for epoch in range(1, epochs + 1):
            model.train()
            idx = list(range(len(train)))
            rng.shuffle(idx)
            start, total = time.time(), 0.0
            optim.zero_grad()
            for n, b in enumerate(range(0, len(idx), batch_size), start=1):
                ids = idx[b : b + batch_size]
                enc = tokenizer(
                    [train.texts[i] for i in ids], truncation=True, max_length=MAX_LENGTH,
                    padding=True, return_tensors="pt",
                ).to(device)
                loss = model(**enc, labels=y_train[ids].to(device)).loss
                (loss / grad_accum).backward()
                total += loss.item() * len(ids)
                if n % grad_accum == 0 or b + batch_size >= len(idx):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    sched.step()
                    optim.zero_grad()

            probs = clf.predict_proba(val.texts)
            thr = tune_thresholds(y_val, probs)
            score = macro_f1(y_val, probs >= thr)
            print(f"  epoch {epoch}/{epochs}  train loss={total / len(train):.4f}  "
                  f"val macro-F1={score:.4f}  ({time.time() - start:.0f}s)", flush=True)
            if score > best_score:
                best_score, best_thr = score, thr
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        clf.thresholds = best_thr
        clf.model.to(device).eval()
        print(f"  -> kept best epoch (val macro-F1={best_score:.4f})", flush=True)
        return clf

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "thresholds.json").write_text(
            json.dumps({"emotions": EMOTIONS, "thresholds": self.thresholds.tolist(),
                        "base_model": self.base_model})
        )

    @classmethod
    def load(cls, path: str | Path) -> BertClassifier:
        path = Path(path)
        meta = json.loads((path / "thresholds.json").read_text())
        if tuple(meta["emotions"]) != EMOTIONS:
            raise ValueError("Saved model was trained with a different label set.")
        clf = cls(
            AutoModelForSequenceClassification.from_pretrained(path),
            _load_tokenizer(path),
            meta["base_model"],
        )
        clf.thresholds = np.array(meta["thresholds"])
        return clf
