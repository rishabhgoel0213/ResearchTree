# Parameter Golf Baseline

This directory contains the baseline training code that the local search harness mutates and evaluates.

Trimmed contents:

- `train_gpt.py`: baseline trainer used as the search root
- `.gitignore`: local ignores for TreeGit and scorer outputs
- `LICENSE` and `THIRD_PARTY_NOTICES.md`: retained from the upstream source

The example-level data downloader writes artifacts to [`../data/`](../data/). `train_gpt.py` is typically invoked through [`../score.py`](../score.py), which passes explicit `DATA_PATH` and `TOKENIZER_PATH` values pointing at that cache.
