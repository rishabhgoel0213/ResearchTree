You are editing a TreeGit worktree, not a Git checkout.

Context:
- Agent: {agent_name} (slot {agent_slot})
- Parent branch: {parent_branch}
- Child branch: {branch}
- Worktree: {worktree}
- Repo root: {repo_root}

TreeGit context files:
- Read `.treegit/mcts/change_history.md` before editing. It contains the aggregated branch-change notes for the lineage leading into the current parent branch.
- Read `.treegit/mcts/score_history.md` before editing. It contains the recorded evaluation outputs for the same lineage, including the parent branch score when one exists.
- Write your own branch note into `.treegit/mcts/current_change.md` before you stop. The harness aggregates that file into future descendants' `change_history.md`.

First read `README.md` in the repo root so you are optimizing for the actual challenge goal, not just making a random change.

Exact goal:
- The real Parameter Golf goal is to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated on FineWeb validation compression quality in bits per byte.
- Lower `val_bpb` is better. Lower `val_loss` is also useful, but the challenge is fundamentally about parameter-limited model quality under strict artifact and runtime constraints.
- The artifact budget includes code bytes plus compressed model bytes. Avoid ideas that obviously explode parameter count, artifact size, or evaluation complexity.
- In this local search harness, the immediate proxy objective is the `score.py` score: lower is better, it is dominated by `val_bpb` and `val_loss`, and it applies huge penalties if the total submission size exceeds 16,000,000 bytes or if `train_gpt.py` becomes too large.

Task:
- Make exactly one concrete improvement intended to improve the Parameter Golf objective and the local `score.py` proxy.
- Be exploratory. Prefer a strong, defensible hypothesis over a timid micro-edit.
- Favor ideas that are plausible from ML literature or strong prior art in small-model training, compression, quantization, parameter tying, recurrence, low-rank structure, efficient attention, optimizer behavior, tokenizer-aware modeling, or evaluation-aware architecture choices.
- Prefer a single coherent idea with a few related edits over unrelated tweaks.
- Keep the code runnable.
- Do not use git commands.
- Do not create commits or branches.
- Stop after making the edit.

Research:
- Before editing, spend a short amount of time looking for a useful idea in the repo and in the installed research skills.
- Use `$academic-research` (`/home/r1shabhg/.codex/skills/academic-research/SKILL.md`) and `$arxiv` (`/home/r1shabhg/.codex/skills/arxiv/SKILL.md`) if they help you ground the change in papers or strong ML prior art.
- Keep the research brief and targeted. You do not need a full literature review; you need one promising idea that can be translated into one concrete code change.
- If you cite an idea from prior work, adapt it pragmatically to this codebase instead of copying jargon into the code.

Validation:
- You may run lightweight checks if useful.
- Do not launch training or GPU-heavy validation runs from the agent itself.
- If you run anything, bias toward cheap static or syntax checks only.
- `python3 -m py_compile train_gpt.py` is acceptable if you made a risky Python edit and want a fast syntax sanity check.
- Runtime or training failures are handled by the objective evaluation itself, so do not add your own smoke-training loop.

Implementation guidance:
- Start by reading `README.md`, `.treegit/mcts/change_history.md`, `.treegit/mcts/score_history.md`, and `train_gpt.py`.
- Prefer edits to `train_gpt.py`, because that is where the challenge expects most counted code to live.
- Weird but well-motivated ideas are acceptable if they are still implementable and testable in this repo.
- Avoid changes that mainly optimize for cleanliness or style unless they clearly support the objective.

Output:
- Apply the code change directly in the worktree and then exit.
- Before you exit, fill in `.treegit/mcts/current_change.md` using the exact section structure already present there.
- Keep `Summary` and `Hypothesis` to one concise sentence each.
- List every edited repo file under `Files Changed`.
- Under `Validation`, record either a cheap static check you ran or `- not run`.
- Put any extra caveats or follow-up context under `Notes`.
- Do not edit `.treegit/mcts/change_history.md` or `.treegit/mcts/score_history.md` directly.
