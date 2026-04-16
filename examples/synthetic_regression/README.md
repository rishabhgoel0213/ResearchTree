# Synthetic Regression Example

This example is a lightweight search target for ResearchTree.

Instead of long language-model training runs, it uses a tiny regression task: fit a fixed synthetic function with a small MLP and score candidates by validation MSE. The point is to keep the full loop fast enough that we can iterate on prompts, TreeGit wiring, objective definitions, and simple model/code mutations without waiting on heavy training.

## Goal

Use this example when we want to:

- validate the end-to-end TreeGit search loop
- test scoring and artifact handling on a cheap objective
- iterate on agent prompts and branch expansion behavior
- try model or optimizer edits that should show up within seconds

## Layout

- `src/`: baseline training code used as the TreeGit root
- `score.py`: launches a run or scores an existing log
- `objectives/synthetic_regression_objective.py`: generic objective wrapper for TreeGit
- `mcts/smoke.json`: very cheap search config
- `mcts/real.json`: slightly less constrained search config

## Quick Start

Set up the environment:

```bash
cd examples/synthetic_regression
pixi install
```

Run a baseline score:

```bash
cd examples/synthetic_regression
pixi run python score.py ./src --json
```

Score an existing log:

```bash
cd examples/synthetic_regression
pixi run python score.py ./src \
  --log-file ./src/score_runs/your_run/logs/your_run.txt \
  --json
```

## TreeGit Wiring

Run TreeGit from inside `src/`:

```bash
cd examples/synthetic_regression/src
python3 ../../../treegit/src/treegit/cli.py init
python3 ../../../treegit/src/treegit/cli.py mcts init --config ../mcts/smoke.json
```

Candidate worktrees live under `examples/synthetic_regression/worktrees/` and artifacts live under `examples/synthetic_regression/artifacts/`.

## Scoring

Lower is better.

The score is mostly validation MSE, with small penalties for runtime and script size so the search still prefers clean, cheap candidates when two models perform similarly.
