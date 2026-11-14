# Data

This repository does **not** redistribute any dataset. `emotion-chatbot train/evaluate`
downloads the German subset of BRIGHTER from the Hugging Face Hub into `data/raw/`
(git-ignored) on first use.

| | |
|---|---|
| Dataset | [BRIGHTER emotion categories](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories), config `deu` |
| Used in | SemEval-2025 Task 11, Track A (multi-label emotion detection) |
| License | CC BY 4.0 |
| Labels | `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise` (multi-label; no label = neutral) |
| Text | Short German texts, mostly social-media style |

## Citation

If you use these results or data, please cite the dataset papers:

```bibtex
@inproceedings{muhammad2025brighter,
  title     = {{BRIGHTER}: {BRI}dging the {G}ap in {H}uman-{A}nnotated {T}extual {E}motion
               {R}ecognition {D}atasets for 28 {L}anguages},
  author    = {Muhammad, Shamsuddeen Hassan and Ousidhoum, Nedjma and others},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for
               Computational Linguistics (ACL)},
  year      = {2025},
  note      = {arXiv:2502.11926}
}

@misc{muhammad2025semeval,
  title  = {{SemEval-2025 Task 11}: Bridging the Gap in Text-Based Emotion Detection},
  author = {Muhammad, Shamsuddeen Hassan and Ousidhoum, Nedjma and others},
  year   = {2025},
  note   = {arXiv:2503.07269}
}
```

## How the data is prepared (`src/emotion_chatbot/data.py`)

Findings from inspecting the raw files, handled in code and covered by tests:

* **Every dev and test text appears twice** (track A and the cross-lingual track C, with
  identical labels). Only track A is used; otherwise all test metrics would be double-counted
  and confidence intervals too narrow.
* One further exact-duplicate text in test is dropped.
* **Dev has only 200 unique texts**, too few to tune per-label thresholds reliably. The
  validation set is therefore dev + 300 examples drawn from train with a fixed seed (42).

| Split | Texts | Use |
|---|---|---|
| train | 2,303 | fit models |
| validation | 500 | hyper-parameters, per-label thresholds, epoch selection, lexicon sanity checks |
| test | 2,603 | final report only |

Train, validation and test share no text.

## Caveats

* Anger and disgust in this data are largely **topic-driven** (e.g. political texts about
  war and parties) rather than expressed through explicit emotion words. This explains why a
  lexicon cannot do well on them (see README).
* Labels are subjective; annotator agreement is imperfect. Treat small score differences with
  care and look at the confidence intervals.
