# ResearchTree

ResearchTree is a local search harness for running structured experimentation against compact-model training objectives.

The immediate use case in this repository is OpenAI's Parameter Golf challenge: start from a known baseline, generate modified descendants with TreeGit, score them with a consistent local proxy, and keep artifacts and worktrees organized enough to run many iterations.

The goal is not just to train one model once. The goal is to support repeated search over code changes:

- keep a baseline source tree that acts as the search root
- spin out candidate worktrees
- evaluate each candidate with the same scorer and objective wrapper
- preserve run outputs so promising branches can be compared and revisited

## Layout

- [`examples/parameter_golf/`](examples/parameter_golf/): end-to-end example of this workflow for Parameter Golf
- [`examples/synthetic_regression/`](examples/synthetic_regression/): fast synthetic example for cheap search iterations
- [`treegit/`](treegit/): TreeGit checkout used to manage search branches and worktrees

## Search Loop

At a high level, the loop in this repo is:

1. Start from a baseline training script.
2. Create candidate branches and worktrees with TreeGit.
3. Apply one concrete model or training change per candidate.
4. Score that candidate with a fixed local proxy.
5. Keep the best descendants and continue expanding.

## Clone

Clone with submodules so `treegit/` is present immediately:

```bash
git clone --recurse-submodules https://github.com/rishabhgoel0213/ResearchTree.git
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Usage

The runnable examples in this repo live under `examples/`. Start there for the scorer, objective wrapper, MCTS configs, and usage notes:

- [`examples/parameter_golf/README.md`](examples/parameter_golf/README.md)
- [`examples/synthetic_regression/README.md`](examples/synthetic_regression/README.md)
