# ResearchTree

This repository now keeps its upstream dependencies as submodules and places the Parameter Golf harness under [`parameter_golf_example/`](parameter_golf_example/).

## Layout

- [`parameter_golf/`](parameter_golf/): submodule pointing at the upstream OpenAI Parameter Golf repository
- [`treegit/`](treegit/): submodule pointing at the TreeGit repository used by the example harness
- [`parameter_golf_example/`](parameter_golf_example/): this repo's example implementation for running TreeGit against Parameter Golf

## Clone

Clone with submodules so both referenced projects are present immediately:

```bash
git clone --recurse-submodules https://github.com/rishabhgoel0213/ResearchTree.git
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Usage

The example harness lives entirely under [`parameter_golf_example/`](parameter_golf_example/). Start there for scoring, MCTS configs, and local run documentation:

- [`parameter_golf_example/README.md`](parameter_golf_example/README.md)
