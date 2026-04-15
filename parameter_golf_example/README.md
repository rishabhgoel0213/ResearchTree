# Parameter Golf Example

This directory is one concrete example of how to wire this repository around OpenAI's Parameter Golf challenge. The canonical challenge repo now lives in the root-level [`parameter_golf/`](../parameter_golf/) submodule, and TreeGit lives in the root-level [`treegit/`](../treegit/) submodule.

If you want the challenge rules and baseline training details, start with [`../parameter_golf/README.md`](../parameter_golf/README.md). This README only documents the example harness.

## Layout

- `score.py`: train-and-score entrypoint for a candidate repo or TreeGit worktree
- `objectives/parameter_golf_objective.py`: JSON objective wrapper used by TreeGit
- `mcts/smoke.json`: cheaper search config for wiring checks
- `mcts/real.json`: longer-running search config
- `mcts/prompt.md`: agent prompt used for worktree expansion
- `skills/`: helper skills that were used while exploring this example
- `test_runs/`: preserved sample outputs from earlier local runs
- `plot.png` and `search-tree.svg`: example result visualizations

## Requirements

- Linux
- `pixi`
- the root repo cloned with submodules
- FineWeb data and tokenizer assets under `../data/`, or explicit `--data-path` and `--tokenizer-path` overrides

## Quick Start

Create the example environment:

```bash
cd parameter_golf_example
pixi install
```

Run a cheap training-and-score pass against the root `parameter_golf` submodule:

```bash
cd parameter_golf_example
pixi run python score.py ../parameter_golf \
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

Score one of the preserved example logs without launching training:

```bash
cd parameter_golf_example
pixi run python score.py ../parameter_golf \
  --log-file ./test_runs/test_20260322_233131/logs/test_20260322_233131.txt
```

Run the JSON objective wrapper directly:

```bash
cd parameter_golf_example
pixi run python objectives/parameter_golf_objective.py ../parameter_golf \
  --objective-version manual-smoke \
  --output-root ./artifacts/manual-smoke \
  --env ITERATIONS=2 \
  --env WARMUP_STEPS=0 \
  --env WARMDOWN_ITERS=0 \
  --env MAX_WALLCLOCK_SECONDS=60
```

## TreeGit Wiring

Run TreeGit from inside the `parameter_golf/` submodule and point it at the configs in this directory:

```bash
cd parameter_golf
python3 ../treegit/src/treegit/cli.py init
python3 ../treegit/src/treegit/cli.py mcts init --config ../parameter_golf_example/mcts/smoke.json
```

The checked-in configs assume:

- candidate worktrees are created under `parameter_golf_example/worktrees/` or `parameter_golf_example/smoke-worktrees/`
- run artifacts are written under `parameter_golf_example/artifacts/` or `parameter_golf_example/smoke-artifacts/`
- the shared dataset cache lives at the repo-root `data/` directory

## How Scoring Works

`score.py` supports two modes:

- run mode: launches `torchrun` via this directory's `pixi.toml`, then parses the produced log
- score-only mode: parses an existing log with `--log-file`

When launching training, the scorer:

- treats the positional argument as the candidate repo root
- defaults the train script to `<repo_root>/train_gpt.py`
- searches for `data/` near the candidate repo, then near this example, then at the repo root
- writes run outputs under `<repo_root>/score_runs/` unless you override `--output-root` or `--run-dir`

The final score is a weighted sum of `val_bpb`, `val_loss`, and total submission size in bytes, with hard penalties if the submission exceeds the artifact or line caps.

## Example Search Result

This plot shows one best-branch lineage improving `val_bpb` over 63 search steps, from `1.9604` down to `0.9352`.

![Best-branch lineage example](plot.png)
