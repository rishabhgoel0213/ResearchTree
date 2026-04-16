#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from container_spec import default_command, load_container_spec, resolve_container_relative_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run setup steps inside a ResearchTree example container.")
    parser.add_argument("config_path", type=Path, help="Absolute path to the container.toml inside the container.")
    parser.add_argument("--setup-only", action="store_true", help="Run setup hooks and exit.")
    parser.add_argument("container_command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_container_spec(args.config_path, REPO_ROOT)
    command = _normalize_override_command(args.container_command)
    env = os.environ.copy()
    env.update(spec.runtime.env)

    for step in spec.setup:
        marker_path = None
        if step.once is not None:
            marker_path = resolve_container_relative_path(spec.container_example_dir, step.once)
            if marker_path.exists():
                print(f"[container] skipping setup step {step.name} (marker exists)")
                continue
        print(f"[container] setup step {step.name}: {step.run}")
        subprocess.run(
            ["bash", "-lc", step.run],
            cwd=spec.container_workdir,
            env=env,
            check=True,
        )
        if marker_path is not None:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(
                f"{datetime.now(timezone.utc).isoformat()} {step.name}\n",
                encoding="utf-8",
            )

    if args.setup_only:
        return 0

    if command is None:
        command = default_command(spec)

    print(f"[container] launching: {' '.join(command)}")
    completed = subprocess.run(command, cwd=spec.container_workdir, env=env, check=False)
    return completed.returncode


def _normalize_override_command(raw: list[str]) -> list[str] | None:
    if not raw:
        return None
    if raw[0] == "--":
        raw = raw[1:]
    if not raw:
        return None
    return raw


if __name__ == "__main__":
    raise SystemExit(main())
