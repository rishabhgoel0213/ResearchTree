#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VENV_DIR = REPO_ROOT / ".venv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a local virtual environment for ResearchTree helper scripts."
    )
    parser.add_argument(
        "--venv-dir",
        default=str(DEFAULT_VENV_DIR),
        help=f"Virtual environment directory to create. Default: {DEFAULT_VENV_DIR}",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for the virtual environment. Default: current interpreter.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the virtual environment if it already exists.",
    )
    parser.add_argument(
        "--skip-pip-upgrade",
        action="store_true",
        help="Do not upgrade pip/setuptools/wheel inside the virtual environment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    python_executable = Path(args.python).expanduser().resolve()
    if not python_executable.exists():
        parser.exit(status=1, message=f"error: python executable not found: {python_executable}\n")

    if shutil.which("docker") is None:
        print("warning: docker is not installed or not on PATH; scripts/container.py will not run until it is available")

    venv_dir = Path(args.venv_dir).expanduser().resolve()
    if args.recreate and venv_dir.exists():
        shutil.rmtree(venv_dir)

    create_virtualenv(python_executable, venv_dir)
    venv_python = venv_python_path(venv_dir)

    if not args.skip_pip_upgrade:
        run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    packages = required_packages(venv_python)
    if packages:
        run([str(venv_python), "-m", "pip", "install", *packages])

    print(f"venv: {venv_dir}")
    print(f"python: {venv_python}")
    if packages:
        print(f"installed: {' '.join(packages)}")
    else:
        print("installed: none")
    print("container.py dependencies: stdlib only, plus tomli on Python < 3.11")
    print(f"activate: source {venv_dir / 'bin' / 'activate'}")
    print(f"test: {venv_python} {REPO_ROOT / 'scripts' / 'container.py'} --help")
    return 0


def create_virtualenv(python_executable: Path, venv_dir: Path) -> None:
    if venv_dir.exists() and (venv_dir / "pyvenv.cfg").exists():
        return
    if python_executable.resolve() == Path(sys.executable).resolve():
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_dir)
        return
    run([str(python_executable), "-m", "venv", str(venv_dir)])


def venv_python_path(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def required_packages(venv_python: Path) -> list[str]:
    completed = subprocess.run(
        [
            str(venv_python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    major, minor = (int(part) for part in completed.stdout.strip().split(".", 1))
    packages: list[str] = []
    if (major, minor) < (3, 11):
        packages.append("tomli")
    return packages


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
