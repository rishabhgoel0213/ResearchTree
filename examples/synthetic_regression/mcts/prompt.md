# Synthetic Regression Search Prompt

You are editing a very small regression baseline.

## Objective

Improve the local synthetic regression score. Lower is better.

The score mostly reflects validation MSE, with smaller penalties for runtime and script size.

## Constraints

- Work only inside this worktree.
- Prefer a single focused improvement.
- Keep the script readable and reasonably compact.
- Do not add external downloads or new heavy dependencies.

## Good Change Categories

- better activations
- mild architectural changes
- small optimizer or schedule changes
- better normalization or initialization
- batching or sampling improvements

## Before You Edit

- Read `README.md`
- Read `.treegit/mcts/change_history.md`
- Read `.treegit/mcts/score_history.md`
- Read `train.py`

## Before You Stop

- If you made risky Python edits, a quick `python3 -m py_compile train.py` is acceptable.
- Write your branch note to `.treegit/mcts/current_change.md`.
- Do not edit `.treegit/mcts/change_history.md` or `.treegit/mcts/score_history.md` directly.
