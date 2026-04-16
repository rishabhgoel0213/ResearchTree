#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_SCRIPT = SCRIPT_ROOT / "score.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit a generic objective JSON payload for synthetic regression.")
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=".",
        help="Repo root to score. Relative paths are resolved from the current working directory.",
    )
    parser.add_argument(
        "--score-script",
        type=Path,
        default=DEFAULT_SCORE_SCRIPT,
        help="Path to the synthetic regression score.py entrypoint.",
    )
    parser.add_argument("--train-script", type=Path, default=None)
    parser.add_argument("--objective-version", default="v1")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    command = [
        sys.executable,
        str(args.score_script.resolve()),
        str(repo_root),
        "--json",
    ]
    if args.train_script is not None:
        command.extend(["--train-script", str(args.train_script.resolve())])
    if args.run_id is not None:
        command.extend(["--run-id", args.run_id])
    if args.output_root is not None:
        command.extend(["--output-root", str(args.output_root.resolve())])
    if args.run_dir is not None:
        command.extend(["--run-dir", str(args.run_dir.resolve())])
    if args.log_file is not None:
        command.extend(["--log-file", str(args.log_file.resolve())])
    for item in args.env:
        command.extend(["--env", item])

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return _emit_failure(
            args.objective_version,
            {
                "reason": f"score.py exited with code {completed.returncode}",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "command": command,
            },
        )

    payload = _extract_json_payload(completed.stdout)
    if payload is None:
        return _emit_failure(
            args.objective_version,
            {
                "reason": "score.py did not emit valid JSON",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "command": command,
            },
        )
    if not isinstance(payload, dict):
        return _emit_failure(
            args.objective_version,
            {"reason": "score.py JSON payload must be an object", "command": command},
        )

    raw_score = payload.get("score")
    raw_score = float(raw_score) if raw_score is not None else None
    success = bool(payload.get("passes_line_cap")) and bool(payload.get("passes_size_cap"))
    artifacts = {
        key: value
        for key, value in {
            "log_path": payload.get("log_path"),
            "run_dir": payload.get("run_dir"),
            "score_json_path": payload.get("score_json_path"),
        }.items()
        if isinstance(value, str)
    }
    objective_payload = {
        "success": success,
        "objective_id": "synthetic_regression",
        "objective_version": args.objective_version,
        "direction": "minimize",
        "raw_score": raw_score,
        "utility": None if raw_score is None else -raw_score,
        "metrics": payload.get("inputs", {}),
        "artifacts": artifacts,
        "payload": payload,
    }
    json.dump(objective_payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _emit_failure(objective_version: str, payload: dict[str, object]) -> int:
    json.dump(
        {
            "success": False,
            "objective_id": "synthetic_regression",
            "objective_version": objective_version,
            "direction": "minimize",
            "raw_score": None,
            "utility": None,
            "metrics": {},
            "artifacts": {},
            "payload": payload,
        },
        sys.stdout,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


def _extract_json_payload(stdout: str) -> dict[str, object] | None:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        pass
    else:
        return payload if isinstance(payload, dict) else None

    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if not stdout[index + end :].strip():
            return payload if isinstance(payload, dict) else None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
