# ResearchTree

This repository keeps TreeGit as a submodule and places the Parameter Golf example under [`examples/parameter_golf/`](examples/parameter_golf/).

## Layout

- [`examples/parameter_golf/`](examples/parameter_golf/): the example harness and related assets
- [`examples/parameter_golf/src/`](examples/parameter_golf/src/): trimmed vendored baseline source used by the example harness
- [`treegit/`](treegit/): submodule pointing at the TreeGit repository used by the example harness

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

The example harness lives entirely under [`examples/parameter_golf/`](examples/parameter_golf/). Start there for scoring, MCTS configs, and local run documentation:

- [`examples/parameter_golf/README.md`](examples/parameter_golf/README.md)
