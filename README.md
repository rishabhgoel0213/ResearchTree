# ResearchTree

ResearchTree is a framework for systematically exploring and improving AI research ideas through structured search and experimentation.

The goal is not just to make one improvement to a codebase. The goal is to support repeated search over code changes:

- keep a baseline source tree that acts as the search root
- spin out candidate worktrees
- evaluate each candidate with the same scorer and objective wrapper
- preserve run outputs so promising branches can be compared and revisited

## Layout

- [`examples/parameter_golf/`](examples/parameter_golf/): end-to-end example of this workflow for OpenAI's Parameter Golf challenge
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

## Containers

This repo includes a first-pass container workflow driven by a `container.toml` in each example directory.

- shared image logic lives in `docker/Dockerfile`
- the host-side launcher lives in `scripts/container.py` (nix is also supported using `scripts/container_nix.py`)
- each example declares its runtime in `examples/*/container.toml`

The current model is intentionally simple:

- the repo is copied into a per-example Docker workspace volume at launch time
- only explicitly configured large directories are bind-mounted from the host into that copied workspace
- the container home directory is also kept in a Docker volume, while host `~/.codex/` is mounted into it for auth/config reuse
- each example's `.pixi/` directory is overlaid with a Docker volume so Pixi environments stay on a Linux filesystem
- OS-level tools such as `git`, `tmux`, `python3`, and `pixi` come from the shared image
- Codex CLI itself is installed in the shared image
- Python/package dependencies still come from each example's `pixi.toml`
- example-local setup hooks such as `pixi run download-data` come from `container.toml`

Build the image for an example:

```bash
python3 scripts/container.py build synthetic_regression
```

Run an example shell after any configured setup steps:

```bash
python3 scripts/container.py shell synthetic_regression
```

Run a specific command inside the example container:

```bash
python3 scripts/container.py run synthetic_regression -- \
  pixi run python score.py ./src --json
```

Prepare the Parameter Golf container and data cache:

```bash
python3 scripts/container.py run parameter_golf --setup-only
```

The container workspace is disposable and isolated from the host checkout, so generated `.treegit/`, worktrees, artifacts, and similar churn stay inside Docker-managed volumes instead of showing up in the base repository.

The checked-in example configs pin `platform = "linux/amd64"` because the current Pixi manifests target `linux-64`.

Codex CLI is installed in the shared image, and the launcher mounts host `~/.codex/` into the container so authenticated Codex-driven MCTS runs can work inside the container as well.
