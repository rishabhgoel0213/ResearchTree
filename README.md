# ResearchTree

ResearchTree is a local search harness for iterating on OpenAI's Parameter Golf challenge with TreeGit-managed worktrees and a shared scorer.

The repository keeps four things together:

- a reference `parameter-golf/` checkout that agents branch from
- shared datasets and tokenizer assets in `data/`
- scoring and objective wrappers in the repo root
- TreeGit worktrees, prompts, and run artifacts for search

If you want the challenge rules and the baseline training script details, start with [`parameter-golf/README.md`](parameter-golf/README.md). This root README documents the harness around that repo.

## What This Repo Does

The search loop is:

1. Start from the baseline `parameter-golf/` repo.
2. Expand child worktrees with TreeGit and the prompt in [`mcts/prompt.md`](mcts/prompt.md).
3. Run the objective wrapper in [`objectives/parameter_golf_objective.py`](objectives/parameter_golf_objective.py).
4. Have [`score.py`](score.py) either launch a training run or score an existing log.
5. Store run outputs under `artifacts/` and keep candidate repos under `worktrees/`.

The proxy objective is intentionally simple: lower is better, mostly driven by `val_bpb` and `val_loss`, with huge penalties if the total submission exceeds 16,000,000 bytes or `train_gpt.py` exceeds 1500 lines.

## Layout

- `parameter-golf/`: reference challenge checkout and baseline training code
- `score.py`: train-and-score entrypoint for a candidate repo or worktree
- `objectives/parameter_golf_objective.py`: JSON objective wrapper used by TreeGit
- `mcts/smoke.json`: cheap config for validating the search loop
- `mcts/real.json`: longer-running config for actual exploration
- `mcts/prompt.md`: agent prompt used when expanding worktrees
- `data/`: shared FineWeb shards, tokenizer files, and dataset utilities
- `artifacts/`: per-run outputs written by the objective harness
- `worktrees/`: TreeGit-managed candidate branches

## Requirements

- Linux
- `pixi`
- a local `parameter-golf` checkout or compatible worktree containing `train_gpt.py`
- the shared dataset and tokenizer under `data/`, or explicit overrides passed to the scorer

The checked-in MCTS configs currently contain absolute paths under `/home/r1shabhg/...`. Update [`mcts/smoke.json`](mcts/smoke.json) and [`mcts/real.json`](mcts/real.json) before using this on another machine.

## Quick Start

Create the environment:

```bash
pixi install
```

Run a cheap training-and-score pass against the baseline checkout:

```bash
pixi run python score.py ./parameter-golf \
  --run-id baseline_smoke \
  --env ITERATIONS=2 \
  --env WARMUP_STEPS=0 \
  --env WARMDOWN_ITERS=0 \
  --env MAX_WALLCLOCK_SECONDS=60 \
  --env TRAIN_BATCH_TOKENS=8192 \
  --env VAL_LOSS_EVERY=0 \
  --env TRAIN_LOG_EVERY=1 \
  --env MUON_BACKEND_STEPS=1
```

Score an existing log without launching training:

```bash
pixi run python score.py ./parameter-golf \
  --log-file ./parameter-golf/test_runs/test_20260322_233131/logs/test_20260322_233131.txt
```

Run the JSON objective wrapper directly:

```bash
pixi run python objectives/parameter_golf_objective.py ./parameter-golf \
  --objective-version manual-smoke \
  --output-root ./artifacts/manual-smoke \
  --env ITERATIONS=2 \
  --env WARMUP_STEPS=0 \
  --env WARMDOWN_ITERS=0 \
  --env MAX_WALLCLOCK_SECONDS=60
```

## How Scoring Works

`score.py` supports two modes:

- run mode: launches `torchrun` via the root `pixi.toml`, then parses the produced log
- score-only mode: parses an existing log with `--log-file`

When launching training, the scorer:

- treats the positional argument as the candidate repo root
- defaults the train script to `<repo_root>/train_gpt.py`
- uses `data/` from the candidate repo if present, otherwise falls back to the parent repo's `data/`
- writes run outputs under `<repo_root>/score_runs/` unless you override `--output-root` or `--run-dir`

The final score is a weighted sum of:

- `val_bpb`
- `val_loss`
- total submission size in bytes

Hard penalties are applied if the submission exceeds the artifact cap or the line cap.

## Tree Search Notes

[`mcts/smoke.json`](mcts/smoke.json) is the safer place to validate wiring changes. It uses minimal training settings and a smaller expansion width.

[`mcts/real.json`](mcts/real.json) is the heavier search config. It points the objective wrapper at the shared dataset and tokenizer cache in this repo and writes outputs into `artifacts/`.

Both configs assume:

- TreeGit is available
- the Codex TreeGit policy script exists at the configured absolute path
- candidate branches are created under `worktrees/`

## Related Docs

- [`parameter-golf/README.md`](parameter-golf/README.md): challenge context and baseline repo usage
- [`data/README.md`](data/README.md): dataset export and tokenizer asset details
