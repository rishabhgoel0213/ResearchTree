# Parameter Golf Baseline

This directory vendors the baseline `train_gpt.py` used by the local search harness in [`..`](../).

Trimmed contents:

- `train_gpt.py`: baseline trainer scored by the example harness
- `.gitignore`: local ignores for TreeGit and scorer outputs
- `LICENSE` and `THIRD_PARTY_NOTICES.md`: retained from the upstream source

The example-level data downloader writes artifacts to [`../data/`](../data/). `train_gpt.py` is invoked through [`../score.py`](../score.py), which passes explicit `DATA_PATH` and `TOKENIZER_PATH` values pointing at that local cache.
