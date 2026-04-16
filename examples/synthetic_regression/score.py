#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent

TRAIN_SCRIPT_LINE_CAP = 800
TRAIN_SCRIPT_SIZE_CAP_BYTES = 64_000

SCORE_WEIGHTS = {
    "val_mse": 1_000_000.0,
    "train_mse": 100_000.0,
    "elapsed_seconds": 1_000.0,
    "train_script_bytes": 0.01,
}

AUTO_FAIL_BASE_PENALTY = 1_000_000_000.0
AUTO_FAIL_EXCESS_WEIGHTS = {
    "train_script_lines": 1_000_000.0,
    "train_script_bytes": 1_000.0,
}
EXECUTION_FAILURE_PENALTY = 10_000_000_000.0

FINAL_METRICS_RE = re.compile(
    r"final_metrics "
    r"train_mse:(?P<train_mse>\S+) "
    r"val_mse:(?P<val_mse>\S+) "
    r"best_val_mse:(?P<best_val_mse>\S+) "
    r"elapsed_seconds:(?P<elapsed_seconds>\S+) "
    r"parameter_count:(?P<parameter_count>\d+)"
)


@dataclass
class ScoreInputs:
    train_mse: float
    val_mse: float
    best_val_mse: float
    elapsed_seconds: float
    parameter_count: int
    train_script_lines: int
    train_script_bytes: int


@dataclass
class ScoreBreakdown:
    score: float
    passes_line_cap: bool
    passes_size_cap: bool
    base_components: dict[str, float]
    penalties: dict[str, float]
    inputs: ScoreInputs


@dataclass
class ScoreConfig:
    repo_root: Path
    train_script: Path
    output_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or score the synthetic regression example.")
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=".",
        help="Folder containing the candidate repo to score.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=None,
        help="Path to the train.py script to run. Defaults to <repo_root>/train.py.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier for launched training. Defaults to a timestamped score_* id.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory under which per-run folders are created. Defaults to <repo_root>/score_runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory to score or create. Overrides --output-root when launching a run.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Score this exact training log file instead of launching a new run.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment override for train.py. May be passed multiple times.",
    )
    parser.add_argument("--json", action="store_true", help="Print the final result as JSON only.")
    return parser.parse_args()


def resolve_repo_root(raw_repo_root: str) -> Path:
    repo_root = Path(raw_repo_root)
    if not repo_root.is_absolute():
        repo_root = SCRIPT_ROOT / repo_root
    return repo_root.resolve()


def build_config(args: argparse.Namespace) -> ScoreConfig:
    repo_root = resolve_repo_root(args.repo_root)
    return ScoreConfig(
        repo_root=repo_root,
        train_script=(args.train_script if args.train_script is not None else repo_root / "train.py").resolve(),
        output_root=(args.output_root if args.output_root is not None else repo_root / "score_runs").resolve(),
    )


def make_run_id(explicit_run_id: str | None) -> str:
    if explicit_run_id is not None:
        return explicit_run_id
    return datetime.now().strftime("score_%Y%m%d_%H%M%S")


def parse_env_overrides(raw_items: list[str]) -> dict[str, str]:
    env_overrides: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --env value {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --env value {item!r}; empty key")
        env_overrides[key] = value
    return env_overrides


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def parse_log(log_path: Path, train_script: Path) -> ScoreInputs:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    match = None
    for line in reversed(lines):
        match = FINAL_METRICS_RE.search(line)
        if match is not None:
            break
    if match is None:
        raise ValueError(f"Could not find final_metrics line in {log_path}")

    return ScoreInputs(
        train_mse=float(match.group("train_mse")),
        val_mse=float(match.group("val_mse")),
        best_val_mse=float(match.group("best_val_mse")),
        elapsed_seconds=float(match.group("elapsed_seconds")),
        parameter_count=int(match.group("parameter_count")),
        train_script_lines=count_lines(train_script),
        train_script_bytes=train_script.stat().st_size,
    )


def compute_score(inputs: ScoreInputs) -> ScoreBreakdown:
    base_components = {
        "val_mse": inputs.val_mse * SCORE_WEIGHTS["val_mse"],
        "train_mse": inputs.train_mse * SCORE_WEIGHTS["train_mse"],
        "elapsed_seconds": inputs.elapsed_seconds * SCORE_WEIGHTS["elapsed_seconds"],
        "train_script_bytes": inputs.train_script_bytes * SCORE_WEIGHTS["train_script_bytes"],
    }
    passes_line_cap = inputs.train_script_lines <= TRAIN_SCRIPT_LINE_CAP
    passes_size_cap = inputs.train_script_bytes <= TRAIN_SCRIPT_SIZE_CAP_BYTES
    penalties: dict[str, float] = {}

    if not passes_line_cap or not passes_size_cap:
        penalties["base_penalty"] = AUTO_FAIL_BASE_PENALTY
        if not passes_line_cap:
            penalties["line_cap_excess"] = (
                inputs.train_script_lines - TRAIN_SCRIPT_LINE_CAP
            ) * AUTO_FAIL_EXCESS_WEIGHTS["train_script_lines"]
        if not passes_size_cap:
            penalties["size_cap_excess"] = (
                inputs.train_script_bytes - TRAIN_SCRIPT_SIZE_CAP_BYTES
            ) * AUTO_FAIL_EXCESS_WEIGHTS["train_script_bytes"]

    return ScoreBreakdown(
        score=sum(base_components.values()) + sum(penalties.values()),
        passes_line_cap=passes_line_cap,
        passes_size_cap=passes_size_cap,
        base_components=base_components,
        penalties=penalties,
        inputs=inputs,
    )


def launch_training(
    config: ScoreConfig,
    *,
    run_id: str,
    run_dir: Path,
    env_overrides: dict[str, str],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.txt"

    env = os.environ.copy()
    env.update(env_overrides)
    env.setdefault("RUN_ID", run_id)

    command = [sys.executable, str(config.train_script)]
    completed = subprocess.run(
        command,
        cwd=config.repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"train.py exited with code {completed.returncode}")
    return log_path


def build_result_payload(
    *,
    config: ScoreConfig,
    breakdown: ScoreBreakdown,
    run_id: str | None,
    run_dir: Path | None,
    log_path: Path,
    score_json_path: Path | None,
    mode: str,
) -> dict[str, object]:
    return {
        "score": breakdown.score,
        "passes_line_cap": breakdown.passes_line_cap,
        "passes_size_cap": breakdown.passes_size_cap,
        "base_components": breakdown.base_components,
        "penalties": breakdown.penalties,
        "inputs": asdict(breakdown.inputs),
        "caps": {
            "train_script_lines": TRAIN_SCRIPT_LINE_CAP,
            "train_script_bytes": TRAIN_SCRIPT_SIZE_CAP_BYTES,
        },
        "weights": SCORE_WEIGHTS,
        "repo_root": str(config.repo_root),
        "run_id": run_id,
        "run_dir": None if run_dir is None else str(run_dir),
        "log_path": str(log_path),
        "score_json_path": None if score_json_path is None else str(score_json_path),
        "mode": mode,
    }


def emit_execution_failure(
    *,
    config: ScoreConfig,
    log_path: Path | None,
    run_dir: Path | None,
    run_id: str | None,
    reason: str,
    error_type: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    inputs = ScoreInputs(
        train_mse=1.0,
        val_mse=1.0,
        best_val_mse=1.0,
        elapsed_seconds=0.0,
        parameter_count=0,
        train_script_lines=count_lines(config.train_script) if config.train_script.exists() else TRAIN_SCRIPT_LINE_CAP + 1,
        train_script_bytes=config.train_script.stat().st_size if config.train_script.exists() else TRAIN_SCRIPT_SIZE_CAP_BYTES + 1,
    )
    breakdown = ScoreBreakdown(
        score=EXECUTION_FAILURE_PENALTY,
        passes_line_cap=False,
        passes_size_cap=False,
        base_components={},
        penalties={"execution_failure": EXECUTION_FAILURE_PENALTY},
        inputs=inputs,
    )
    return {
        **build_result_payload(
            config=config,
            breakdown=breakdown,
            run_id=run_id,
            run_dir=run_dir,
            log_path=log_path if log_path is not None else config.repo_root / "missing.log",
            score_json_path=None,
            mode="score-only" if args.log_file is not None else "run",
        ),
        "error": {
            "type": error_type,
            "reason": reason,
        },
    }


def main() -> int:
    args = parse_args()
    config = build_config(args)

    try:
        env_overrides = parse_env_overrides(args.env)
    except ValueError as exc:
        result = emit_execution_failure(
            config=config,
            log_path=None,
            run_dir=None,
            run_id=None,
            reason=str(exc),
            error_type="invalid_args",
            args=args,
        )
        print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
        return 0

    run_id: str | None = None
    run_dir: Path | None = None
    log_path: Path
    score_json_path: Path | None = None
    mode = "score-only" if args.log_file is not None else "run"

    try:
        if args.log_file is not None:
            log_path = args.log_file.resolve()
        else:
            run_id = make_run_id(args.run_id)
            run_dir = (args.run_dir.resolve() if args.run_dir is not None else (config.output_root / run_id))
            log_path = launch_training(config, run_id=run_id, run_dir=run_dir, env_overrides=env_overrides)

        breakdown = compute_score(parse_log(log_path, config.train_script))
        result = build_result_payload(
            config=config,
            breakdown=breakdown,
            run_id=run_id,
            run_dir=run_dir,
            log_path=log_path,
            score_json_path=None,
            mode=mode,
        )

        if run_dir is not None:
            score_json_path = run_dir / "score.json"
            result["score_json_path"] = str(score_json_path)
            score_json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        result = emit_execution_failure(
            config=config,
            log_path=log_path if "log_path" in locals() else None,
            run_dir=run_dir,
            run_id=run_id,
            reason=str(exc),
            error_type=type(exc).__name__,
            args=args,
        )

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
