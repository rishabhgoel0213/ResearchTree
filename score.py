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

SUBMISSION_SIZE_CAP_BYTES = 16_000_000
TRAIN_SCRIPT_LINE_CAP = 1500

SCORE_WEIGHTS = {
    "val_bpb": 1_000_000.0,
    "val_loss": 100_000.0,
    "submission_bytes": 0.001,
}

AUTO_FAIL_BASE_PENALTY = 1_000_000_000_000.0
AUTO_FAIL_EXCESS_WEIGHTS = {
    "submission_size_bytes": 1_000_000.0,
    "train_script_lines": 1_000_000_000.0,
}
EXECUTION_FAILURE_PENALTY = 10_000_000_000_000.0

FINAL_EXACT_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>\S+) val_bpb:(?P<val_bpb>\S+)"
)
TOTAL_SIZE_RE = re.compile(r"Total submission size int8\+zlib: (?P<bytes>\d+) bytes")
COMPRESSED_SIZE_RE = re.compile(r"Serialized model int8\+zlib: (?P<bytes>\d+) bytes")
CODE_SIZE_RE = re.compile(r"Code size: (?P<bytes>\d+) bytes")


@dataclass
class ScoreInputs:
    val_loss: float
    val_bpb: float
    total_submission_size_bytes: int
    compressed_model_size_bytes: int | None
    code_size_bytes: int | None
    train_script_lines: int


@dataclass
class ScoreBreakdown:
    score: float
    passes_submission_size_cap: bool
    passes_line_cap: bool
    base_components: dict[str, float]
    penalties: dict[str, float]
    inputs: ScoreInputs


@dataclass
class ScoreConfig:
    repo_root: Path
    pixi_root: Path
    data_root: Path
    train_script: Path
    data_path: Path
    tokenizer_path: Path
    output_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the documented 1xH100 train_gpt.py command by default, then compute a simple weighted score from its log."
        )
    )
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=".",
        help=(
            "Folder containing the repo to score. Relative paths are resolved from the directory containing score.py."
        ),
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=None,
        help="Path to the train_gpt.py script to run and score. Defaults to <repo_root>/train_gpt.py.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "Path to the FineWeb dataset directory used by the 1xH100 command. "
            "Defaults to <repo_root>/../data/datasets/fineweb10B_sp1024."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help=(
            "Path to the SentencePiece tokenizer model used by the 1xH100 command. "
            "Defaults to <repo_root>/../data/tokenizers/fineweb_1024_bpe.model."
        ),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1024,
        help="VOCAB_SIZE for train_gpt.py.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="RUN_ID for a launched training run. Defaults to a timestamped score_* id.",
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
        "--nproc-per-node",
        type=int,
        default=1,
        help="torchrun --nproc_per_node value. Defaults to the documented 1xH100 setting.",
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
        help="Extra environment override for train_gpt.py. May be passed multiple times.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final result as JSON only.",
    )
    return parser.parse_args()


def resolve_repo_root(raw_repo_root: str) -> Path:
    repo_root = Path(raw_repo_root)
    if not repo_root.is_absolute():
        repo_root = SCRIPT_ROOT / repo_root
    return repo_root.resolve()


def build_config(args: argparse.Namespace) -> ScoreConfig:
    repo_root = resolve_repo_root(args.repo_root)
    pixi_root = SCRIPT_ROOT
    if not (pixi_root / "pixi.toml").exists():
        raise FileNotFoundError(f"Global pixi manifest not found: {pixi_root / 'pixi.toml'}")
    data_root = repo_root / "data"
    if not data_root.exists():
        data_root = repo_root.parent / "data"

    return ScoreConfig(
        repo_root=repo_root,
        pixi_root=pixi_root,
        data_root=data_root,
        train_script=(args.train_script if args.train_script is not None else repo_root / "train_gpt.py").resolve(),
        data_path=(
            args.data_path
            if args.data_path is not None
            else data_root / "datasets" / "fineweb10B_sp1024"
        ).resolve(),
        tokenizer_path=(
            args.tokenizer_path
            if args.tokenizer_path is not None
            else data_root / "tokenizers" / "fineweb_1024_bpe.model"
        ).resolve(),
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
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def last_match(lines: list[str], pattern: re.Pattern[str]) -> re.Match[str] | None:
    for line in reversed(lines):
        match = pattern.search(line)
        if match is not None:
            return match
    return None


def parse_log(log_path: Path, train_script: Path) -> ScoreInputs:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    final_match = last_match(lines, FINAL_EXACT_RE)
    total_size_match = last_match(lines, TOTAL_SIZE_RE)

    if final_match is None:
        raise ValueError(f"Could not find final_int8_zlib_roundtrip_exact line in {log_path}")
    if total_size_match is None:
        raise ValueError(f"Could not find Total submission size int8+zlib line in {log_path}")

    compressed_size_match = last_match(lines, COMPRESSED_SIZE_RE)
    code_size_match = last_match(lines, CODE_SIZE_RE)

    return ScoreInputs(
        val_loss=float(final_match.group("val_loss")),
        val_bpb=float(final_match.group("val_bpb")),
        total_submission_size_bytes=int(total_size_match.group("bytes")),
        compressed_model_size_bytes=(
            int(compressed_size_match.group("bytes")) if compressed_size_match is not None else None
        ),
        code_size_bytes=int(code_size_match.group("bytes")) if code_size_match is not None else None,
        train_script_lines=count_lines(train_script),
    )


def compute_score(inputs: ScoreInputs) -> ScoreBreakdown:
    base_components = {
        "val_bpb": inputs.val_bpb * SCORE_WEIGHTS["val_bpb"],
        "val_loss": inputs.val_loss * SCORE_WEIGHTS["val_loss"],
        "submission_bytes": inputs.total_submission_size_bytes * SCORE_WEIGHTS["submission_bytes"],
    }

    penalties: dict[str, float] = {}

    submission_size_over = max(inputs.total_submission_size_bytes - SUBMISSION_SIZE_CAP_BYTES, 0)
    if submission_size_over > 0:
        penalties["submission_size_cap"] = (
            AUTO_FAIL_BASE_PENALTY
            + submission_size_over * AUTO_FAIL_EXCESS_WEIGHTS["submission_size_bytes"]
        )

    train_script_lines_over = max(inputs.train_script_lines - TRAIN_SCRIPT_LINE_CAP, 0)
    if train_script_lines_over > 0:
        penalties["train_script_line_cap"] = (
            AUTO_FAIL_BASE_PENALTY
            + train_script_lines_over * AUTO_FAIL_EXCESS_WEIGHTS["train_script_lines"]
        )

    return ScoreBreakdown(
        score=sum(base_components.values()) + sum(penalties.values()),
        passes_submission_size_cap=submission_size_over == 0,
        passes_line_cap=train_script_lines_over == 0,
        base_components=base_components,
        penalties=penalties,
        inputs=inputs,
    )


def build_execution_failure_payload(
    *,
    config: ScoreConfig | None,
    reason: str,
    run_id: str | None,
    run_dir: Path | None,
    log_path: Path | None,
    command: list[str] | None,
    stdout: str | None,
    stderr: str | None,
    mode: str,
) -> dict[str, object]:
    train_script = None if config is None else config.train_script
    train_script_lines: int | None = None
    if train_script is not None and train_script.exists():
        train_script_lines = count_lines(train_script)
    return {
        "score": EXECUTION_FAILURE_PENALTY,
        "passes_submission_size_cap": False,
        "passes_line_cap": False,
        "base_components": {},
        "penalties": {"execution_failure": EXECUTION_FAILURE_PENALTY},
        "inputs": {},
        "repo_root": None if config is None else str(config.repo_root),
        "log_path": None if log_path is None else str(log_path),
        "run_id": run_id,
        "run_dir": None if run_dir is None else str(run_dir),
        "weights": SCORE_WEIGHTS,
        "caps": {
            "submission_size_bytes": SUBMISSION_SIZE_CAP_BYTES,
            "train_script_lines": TRAIN_SCRIPT_LINE_CAP,
        },
        "mode": mode,
        "failure": {
            "reason": reason,
            "command": command,
            "stdout": stdout,
            "stderr": stderr,
            "train_script": None if train_script is None else str(train_script),
            "train_script_lines": train_script_lines,
        },
    }


def build_run_dir(args: argparse.Namespace, run_id: str) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()
    return (args.output_root / run_id).resolve()


def run_training(args: argparse.Namespace, config: ScoreConfig, run_id: str, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": run_id,
            "DATA_PATH": str(config.data_path),
            "TOKENIZER_PATH": str(config.tokenizer_path),
            "VOCAB_SIZE": str(args.vocab_size),
        }
    )
    env.update(parse_env_overrides(args.env))

    command = [
        "pixi",
        "run",
        "--manifest-path",
        str(config.pixi_root),
        "--executable",
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(config.train_script),
    ]

    print(f"run_dir: {run_dir}")
    print(f"run_id: {run_id}")
    print("command:", " ".join(command))

    completed = subprocess.run(command, cwd=run_dir, env=env)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)

    log_path = run_dir / "logs" / f"{run_id}.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"Expected training log was not created: {log_path}")
    return log_path


def print_human_summary(
    score: ScoreBreakdown,
    log_path: Path,
    run_id: str | None,
    run_dir: Path | None,
) -> None:
    print(f"score: {score.score:.6f}")
    print(f"val_bpb: {score.inputs.val_bpb:.8f}")
    print(f"val_loss: {score.inputs.val_loss:.8f}")
    print(f"total_submission_size_bytes: {score.inputs.total_submission_size_bytes}")
    if score.inputs.compressed_model_size_bytes is not None:
        print(f"compressed_model_size_bytes: {score.inputs.compressed_model_size_bytes}")
    if score.inputs.code_size_bytes is not None:
        print(f"code_size_bytes: {score.inputs.code_size_bytes}")
    print(f"train_script_lines: {score.inputs.train_script_lines}")
    print(f"passes_submission_size_cap: {score.passes_submission_size_cap}")
    print(f"passes_line_cap: {score.passes_line_cap}")
    if score.penalties:
        for key, value in score.penalties.items():
            print(f"penalty_{key}: {value:.6f}")
    print(f"log_path: {log_path}")
    if run_id is not None:
        print(f"run_id: {run_id}")
    if run_dir is not None:
        print(f"run_dir: {run_dir}")


def print_failure_summary(payload: dict[str, object]) -> None:
    print(f"score: {payload['score']:.6f}")
    print("failure: true")
    failure = payload.get("failure", {})
    if isinstance(failure, dict):
        reason = failure.get("reason")
        if reason is not None:
            print(f"failure_reason: {reason}")
        train_script = failure.get("train_script")
        if train_script is not None:
            print(f"train_script: {train_script}")
        train_script_lines = failure.get("train_script_lines")
        if train_script_lines is not None:
            print(f"train_script_lines: {train_script_lines}")
    log_path = payload.get("log_path")
    if log_path is not None:
        print(f"log_path: {log_path}")
    run_id = payload.get("run_id")
    if run_id is not None:
        print(f"run_id: {run_id}")
    run_dir = payload.get("run_dir")
    if run_dir is not None:
        print(f"run_dir: {run_dir}")


def main() -> int:
    args = parse_args()
    config: ScoreConfig | None = None
    run_id: str | None = None
    run_dir: Path | None = None
    log_path: Path | None = None
    payload: dict[str, object]
    try:
        config = build_config(args)
        args.output_root = config.output_root
        if args.log_file is not None:
            log_path = args.log_file.resolve()
            mode = "score-only"
        else:
            run_id = make_run_id(args.run_id)
            run_dir = build_run_dir(args, run_id)
            log_path = run_training(args, config, run_id, run_dir)
            mode = "run"

        result = compute_score(parse_log(log_path, config.train_script))
        payload = {
            "score": result.score,
            "passes_submission_size_cap": result.passes_submission_size_cap,
            "passes_line_cap": result.passes_line_cap,
            "base_components": result.base_components,
            "penalties": result.penalties,
            "inputs": asdict(result.inputs),
            "repo_root": str(config.repo_root),
            "log_path": str(log_path),
            "run_id": run_id,
            "run_dir": str(run_dir) if run_dir is not None else None,
            "weights": SCORE_WEIGHTS,
            "caps": {
                "submission_size_bytes": SUBMISSION_SIZE_CAP_BYTES,
                "train_script_lines": TRAIN_SCRIPT_LINE_CAP,
            },
            "mode": mode,
        }
    except subprocess.CalledProcessError as exc:
        payload = build_execution_failure_payload(
            config=config,
            reason=f"Training command failed with exit code {exc.returncode}",
            run_id=run_id,
            run_dir=run_dir,
            log_path=log_path,
            command=list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else [str(exc.cmd)],
            stdout=getattr(exc, "stdout", None),
            stderr=getattr(exc, "stderr", None),
            mode="execution-failure",
        )
    except Exception as exc:
        payload = build_execution_failure_payload(
            config=config,
            reason=str(exc),
            run_id=run_id,
            run_dir=run_dir,
            log_path=log_path,
            command=None,
            stdout=None,
            stderr=None,
            mode="execution-failure",
        )

    if run_dir is not None:
        score_json_path = run_dir / "score.json"
        score_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["score_json_path"] = str(score_json_path)

    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        if "failure" in payload:
            print_failure_summary(payload)
        else:
            result = compute_score(parse_log(Path(payload["log_path"]), config.train_script))
            print_human_summary(result, Path(payload["log_path"]), run_id, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
