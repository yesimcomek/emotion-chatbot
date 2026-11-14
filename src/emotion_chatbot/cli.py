"""Command-line interface.

    emotion-chatbot                       # chat with the rule-based model
    emotion-chatbot chat --model bert     # chat with a trained model
    emotion-chatbot train tfidf           # train + save a model
    emotion-chatbot evaluate --models rules tfidf bert --split test
    emotion-chatbot demo                  # web demo (pip install .[demo])
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chat import run_chat
from .registry import DEFAULT_PATHS, MODEL_NAMES, load_model

_TRAINABLE = tuple(DEFAULT_PATHS)


def _cmd_chat(args: argparse.Namespace) -> None:
    run_chat(load_model(args.model, args.path))


def _cmd_train(args: argparse.Namespace) -> None:
    from .data import load_splits

    train, val, _ = load_splits()
    out = Path(args.out) if args.out else DEFAULT_PATHS[args.model]
    print(f"Training {args.model} on {len(train)} texts, validating on {len(val)} ...")
    if args.model == "tfidf":
        from .tfidf import TfidfClassifier

        model = TfidfClassifier.train(train, val)
    else:
        from .bert import DEFAULT_BASE_MODEL, BertClassifier

        model = BertClassifier.train(
            train, val, base_model=args.base_model or DEFAULT_BASE_MODEL,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            grad_accum=args.grad_accum, seed=args.seed,
        )
    model.save(out)
    print(f"Saved to {out}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from .data import load_splits
    from .evaluation import evaluate, paired_bootstrap, predict_matrix, to_matrix
    from .report import render_markdown

    _, val, test = load_splits()
    data = {"val": val, "test": test}[args.split]
    results, preds = [], {}
    for name in args.models:
        model = load_model(name)
        print(f"Evaluating {name} on {args.split} ({len(data)} texts) ...")
        res = evaluate(model, data)
        results.append(res)
        preds[res["model"]] = predict_matrix(model, data)[0]
    y = to_matrix(data.labels)
    names = list(preds)
    comparisons = {
        f"{a} vs {b}": paired_bootstrap(y, preds[a], preds[b])
        for i, a in enumerate(names)
        for b in names[:i]
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"split": args.split, "n": len(data), "results": results, "paired": comparisons}
    (out / f"metrics_{args.split}.json").write_text(json.dumps(payload, indent=2))
    md = render_markdown(payload)
    (out / f"results_{args.split}.md").write_text(md)
    print(md)


def _cmd_demo(args: argparse.Namespace) -> None:
    from .demo import main as launch

    launch()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="emotion-chatbot", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="interactive chatbot (default)")
    chat.add_argument("--model", choices=MODEL_NAMES, default="rules")
    chat.add_argument("--path", help="path to a trained model (default: models/...)")

    train = sub.add_parser("train", help="train and save a model")
    train.add_argument("model", choices=_TRAINABLE)
    train.add_argument("--out", help="where to save the model")
    train.add_argument("--base-model", help="[bert] Hugging Face model id")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--lr", type=float, default=3e-5)
    train.add_argument("--batch-size", type=int, default=8, help="[bert] micro-batch size")
    train.add_argument("--grad-accum", type=int, default=2, help="[bert] accumulation steps")
    train.add_argument("--seed", type=int, default=42)

    ev = sub.add_parser("evaluate", help="evaluate models on the validation or test split")
    ev.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["rules"])
    ev.add_argument("--split", choices=("val", "test"), default="val")
    ev.add_argument("--out", default="results")

    sub.add_parser("demo", help="launch the web demo (needs the 'demo' extra)")

    args = parser.parse_args(argv)
    handlers = {"train": _cmd_train, "evaluate": _cmd_evaluate, "demo": _cmd_demo}
    if args.command in handlers:
        handlers[args.command](args)
    else:
        if args.command is None:
            args.model, args.path = "rules", None
        _cmd_chat(args)


if __name__ == "__main__":
    main()
