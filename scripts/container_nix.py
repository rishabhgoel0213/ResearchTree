#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _ensure_compatible_python() -> None:
    if sys.version_info >= (3, 11):
        return
    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    target = VENV_PYTHON.resolve()
    if current == target:
        return
    os.execv(str(target), [str(target), __file__, *sys.argv[1:]])


_ensure_compatible_python()

from container_spec import (
    WORKSPACE_ROOT,
    default_command,
    load_container_spec,
    resolve_container_relative_path,
    resolve_example_config,
)

DEFAULT_INSTALLABLES = [
    "nixpkgs#bashInteractive",
    "nixpkgs#bubblewrap",
    "nixpkgs#cacert",
    "nixpkgs#coreutils",
    "nixpkgs#curl",
    "nixpkgs#findutils",
    "nixpkgs#gawk",
    "nixpkgs#gcc",
    "nixpkgs#git",
    "nixpkgs#git-lfs",
    "nixpkgs#gnumake",
    "nixpkgs#gnugrep",
    "nixpkgs#gnused",
    "nixpkgs#gnutar",
    "nixpkgs#gzip",
    "nixpkgs#pixi",
    "nixpkgs#pkg-config",
    "nixpkgs#procps",
    "nixpkgs#python3",
    "nixpkgs#python3Packages.pip",
    "nixpkgs#python3Packages.requests",
    "nixpkgs#python3Packages.tomli",
    "nixpkgs#ripgrep",
    "nixpkgs#tmux",
    "nixpkgs#util-linux",
    "nixpkgs#uv",
    "nixpkgs#which",
]

TRANSIENT_DIR_NAMES = {
    ".container",
    ".container_state",
    ".git",
    ".pixi",
    ".treegit",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "smoke-artifacts",
    "container-smoke-artifacts",
    "score_runs",
    "worktrees",
    "smoke-worktrees",
    "container-smoke-worktrees",
}

SYSTEM_PACKAGE_MAP = {
    "bash": "nixpkgs#bashInteractive",
    "build-essential": "nixpkgs#gcc",
    "bubblewrap": "nixpkgs#bubblewrap",
    "ca-certificates": "nixpkgs#cacert",
    "curl": "nixpkgs#curl",
    "git": "nixpkgs#git",
    "git-lfs": "nixpkgs#git-lfs",
    "gnupg": "nixpkgs#gnupg",
    "procps": "nixpkgs#procps",
    "python3": "nixpkgs#python3",
    "python3-pip": "nixpkgs#python3Packages.pip",
    "python3-tomli": "nixpkgs#python3Packages.tomli",
    "python3-venv": "nixpkgs#python3",
    "tmux": "nixpkgs#tmux",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ResearchTree examples in a Nix shell.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Realize the Nix environment for an example.")
    _add_example_argument(build_parser)
    build_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Recreate the persistent Nix workspace snapshot before realizing the environment.",
    )

    run_parser = subparsers.add_parser("run", help="Run an example command inside a Nix shell.")
    _add_example_argument(run_parser)
    run_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Recreate the persistent Nix workspace snapshot before running.",
    )
    run_parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Run configured setup steps and exit without launching the default command.",
    )
    run_parser.add_argument("container_command", nargs=argparse.REMAINDER)

    shell_parser = subparsers.add_parser("shell", help="Open an interactive shell for an example.")
    _add_example_argument(shell_parser)
    shell_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Recreate the persistent Nix workspace snapshot before opening the shell.",
    )

    treegit_parser = subparsers.add_parser(
        "treegit",
        help="Run a treegit command in the example's src/ directory inside the Nix workspace.",
    )
    _add_example_argument(treegit_parser)
    treegit_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Recreate the persistent Nix workspace snapshot before running treegit.",
    )
    treegit_parser.add_argument("treegit_args", nargs=argparse.REMAINDER)
    return parser


def _add_example_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "example",
        help="Example name under examples/, an example directory path, or a container.toml path.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _require_nix(parser)

    config_path = resolve_example_config(args.example, REPO_ROOT)
    spec = load_container_spec(config_path, REPO_ROOT)
    installables = _installables_for_spec(spec)
    workspace_root = ensure_workspace_snapshot(spec, fresh=getattr(args, "fresh", False))

    if args.command == "build":
        print(f"[nix] realizing environment for {spec.name}")
        return _run_nix_shell(installables, ["bash", "-lc", ":"], cwd=workspace_root, env=os.environ.copy())

    host_workdir = _default_workdir(spec, workspace_root)
    env = _runtime_env(spec, workspace_root)
    _run_setup(spec, workspace_root, installables, env)

    if args.command == "shell":
        print(f"[nix] launching shell in {host_workdir}")
        return _run_nix_shell(installables, _interactive_shell_command(spec, workspace_root), cwd=host_workdir, env=env)

    if args.command == "treegit":
        command = _treegit_command(spec, workspace_root, _normalize_override_command(args.treegit_args))
        print(f"[nix] launching: {' '.join(command)}")
        return _run_nix_shell(installables, command, cwd=host_workdir, env=env)

    if args.setup_only:
        return 0

    command = _normalize_override_command(args.container_command)
    if command is None:
        command = _translate_command(spec, workspace_root, default_command(spec))
    else:
        command = _translate_command(spec, workspace_root, command)
    print(f"[nix] launching: {' '.join(command)}")
    return _run_nix_shell(installables, command, cwd=host_workdir, env=env)


def _require_nix(parser: argparse.ArgumentParser) -> None:
    if not shutil_which("nix"):
        parser.exit(status=1, message="error: nix is not installed or not on PATH\n")


def _run_setup(spec, workspace_root: Path, installables: list[str], env: dict[str, str]) -> None:
    for step in spec.setup:
        marker_path = None
        if step.once is not None:
            marker_container_path = resolve_container_relative_path(spec.container_example_dir, step.once)
            marker_path = _host_path_for_container_path(spec, workspace_root, marker_container_path)
            if marker_path.exists():
                print(f"[nix] skipping setup step {step.name} (marker exists)")
                continue
        run_command = _translate_container_string(spec, workspace_root, step.run)
        print(f"[nix] setup step {step.name}: {run_command}")
        exit_code = _run_nix_shell(
            installables,
            ["bash", "-lc", run_command],
            cwd=_host_workdir(spec, workspace_root),
            env=env,
        )
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, run_command)
        if marker_path is not None:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(
                f"{datetime.now(timezone.utc).isoformat()} {step.name}\n",
                encoding="utf-8",
            )


def _runtime_env(spec, workspace_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    treegit_bin_dir = _ensure_treegit_wrapper(workspace_root)
    env["PATH"] = str(treegit_bin_dir) + os.pathsep + env.get("PATH", "")
    for key, value in spec.runtime.env.items():
        env[key] = _translate_env_value(spec, workspace_root, value)
    return env


def _host_workdir(spec, workspace_root: Path) -> Path:
    workdir = spec.project.workdir.strip()
    if not workdir or workdir == ".":
        return workspace_root / spec.example_rel
    if workdir.startswith("/"):
        return _host_path_for_container_path(spec, workspace_root, Path(workdir))
    return (workspace_root / spec.example_rel / workdir).resolve()


def _default_workdir(spec, workspace_root: Path) -> Path:
    source_dir = workspace_root / spec.example_rel / "src"
    if source_dir.is_dir():
        return source_dir.resolve()
    return _host_workdir(spec, workspace_root)


def _translate_env_value(spec, workspace_root: Path, value: str) -> str:
    parts = value.split(os.pathsep)
    translated = [_translate_container_string(spec, workspace_root, part) for part in parts]
    return os.pathsep.join(translated)


def _translate_command(spec, workspace_root: Path, command: list[str]) -> list[str]:
    translated: list[str] = []
    for item in command:
        if item in {"bash", "sh"}:
            translated.append(item)
            continue
        translated.append(_translate_container_string(spec, workspace_root, item))
    return translated


def _translate_container_string(spec, workspace_root: Path, value: str) -> str:
    workspace_prefix = str(WORKSPACE_ROOT)
    if value == workspace_prefix:
        return str(workspace_root)
    if not value.startswith(workspace_prefix + "/"):
        return value
    return str(workspace_root / value[len(workspace_prefix) + 1 :])


def _host_path_for_container_path(spec, workspace_root: Path, path: Path) -> Path:
    path_str = str(path)
    workspace_prefix = str(WORKSPACE_ROOT)
    if path_str == workspace_prefix:
        return workspace_root
    if path_str.startswith(workspace_prefix + "/"):
        relative = Path(path_str[len(workspace_prefix) + 1 :])
        return (workspace_root / relative).resolve()
    return path


def ensure_workspace_snapshot(spec, *, fresh: bool) -> Path:
    state_dir = nix_workspace_state_dir(spec)
    workspace_root = state_dir / "workspace"
    if fresh and state_dir.exists():
        print(f"[nix] removing workspace snapshot for {spec.name}")
        shutil.rmtree(state_dir)
    if workspace_root.exists():
        print(f"[nix] reusing workspace snapshot for {spec.name}: {workspace_root}")
        return workspace_root
    print(f"[nix] creating workspace snapshot for {spec.name}: {workspace_root}")
    workspace_root.parent.mkdir(parents=True, exist_ok=True)
    _copy_repo_snapshot(spec, workspace_root)
    _materialize_mounts(spec, workspace_root)
    return workspace_root


def nix_workspace_state_dir(spec) -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return (cache_home / "researchtree-container-nix" / spec.slug).resolve()


def _ensure_treegit_wrapper(workspace_root: Path) -> Path:
    bin_dir = workspace_root.parent / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = bin_dir / "treegit"
    treegit_pythonpath = workspace_root / "treegit" / "src"
    wrapper_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'export PYTHONPATH="{treegit_pythonpath}"${{PYTHONPATH:+:$PYTHONPATH}}',
                'exec python3 -m treegit "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    wrapper_path.chmod(0o755)
    return bin_dir


def _write_shell_rcfile(spec, workspace_root: Path) -> Path:
    state_dir = nix_workspace_state_dir(spec)
    rcfile = state_dir / "shellrc"
    treegit_bin_dir = state_dir / "bin"
    treegit_pythonpath = workspace_root / "treegit" / "src"
    rcfile.write_text(
        "\n".join(
            [
                "# Auto-generated by container_nix.py",
                'if [ -f "$HOME/.bashrc" ]; then',
                '  . "$HOME/.bashrc"',
                "fi",
                "unalias treegit 2>/dev/null || true",
                f'export PATH="{treegit_bin_dir}:$PATH"',
                f'export PYTHONPATH="{treegit_pythonpath}"${{PYTHONPATH:+:$PYTHONPATH}}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return rcfile


def _copy_repo_snapshot(spec, workspace_root: Path) -> None:
    excluded_roots = {spec.resolve_host_mount_source(mount) for mount in spec.runtime.mounts}
    workspace_root.mkdir(parents=True, exist_ok=True)
    for source_root, relative_root in _snapshot_roots(spec):
        source_root = source_root.resolve()
        if not source_root.exists():
            continue
        if source_root.is_file():
            destination_path = workspace_root / relative_root
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root, destination_path)
            continue
        for root, dirs, files in os.walk(source_root, topdown=True):
            root_path = Path(root)
            dirs[:] = [
                name
                for name in dirs
                if not _is_excluded_path(root_path / name, excluded_roots, source_root=source_root)
            ]
            destination_dir = workspace_root / relative_root / root_path.relative_to(source_root)
            destination_dir.mkdir(parents=True, exist_ok=True)
            for file_name in files:
                source_path = root_path / file_name
                if _is_excluded_path(source_path, excluded_roots, source_root=source_root):
                    continue
                destination_path = destination_dir / file_name
                if source_path.is_symlink():
                    destination_path.symlink_to(os.readlink(source_path))
                    continue
                shutil.copy2(source_path, destination_path)


def _snapshot_roots(spec) -> list[tuple[Path, Path]]:
    # Keep the staged workspace narrow: the example itself plus the shared
    # TreeGit code that the example configs invoke via /workspace/treegit/...
    roots: list[tuple[Path, Path]] = [
        (spec.example_dir, spec.example_rel),
        (REPO_ROOT / "treegit", Path("treegit")),
    ]
    return roots


def _is_excluded_path(path: Path, excluded_roots: set[Path], *, source_root: Path) -> bool:
    if path == source_root:
        return False
    if path.name == ".DS_Store":
        return True
    relative_parts = path.relative_to(source_root).parts
    if any(part in TRANSIENT_DIR_NAMES for part in relative_parts):
        return True
    for excluded_root in excluded_roots:
        if path == excluded_root or excluded_root in path.parents:
            return True
    return False


def _materialize_mounts(spec, workspace_root: Path) -> None:
    for mount in spec.runtime.mounts:
        host_source = spec.resolve_host_mount_source(mount)
        container_target = spec.resolve_container_mount_target(mount)
        target_path = _host_path_for_container_path(spec, workspace_root, container_target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() or target_path.is_symlink():
            if target_path.is_dir() and not target_path.is_symlink():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
        target_path.symlink_to(host_source, target_is_directory=host_source.is_dir())


def _normalize_override_command(raw: list[str]) -> list[str] | None:
    if not raw:
        return None
    if raw[0] == "--":
        raw = raw[1:]
    if not raw:
        return None
    return raw


def _treegit_command(spec, workspace_root: Path, raw_args: list[str] | None) -> list[str]:
    source_dir = workspace_root / spec.example_rel / "src"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Example {spec.example_dir} does not contain a src/ directory for treegit")
    treegit_args = raw_args or ["--help"]
    treegit_pythonpath = workspace_root / "treegit" / "src"
    command = [
        "bash",
        "-lc",
        (
            f"export PYTHONPATH={shlex_quote(str(treegit_pythonpath))}"
            f"${{PYTHONPATH:+:$PYTHONPATH}} && "
            f"exec python3 -m treegit {' '.join(shlex_quote(arg) for arg in treegit_args)}"
        ),
    ]
    return command


def shlex_quote(value: str) -> str:
    return shlex.quote(value)


def _interactive_shell_command(spec, workspace_root: Path) -> list[str]:
    rcfile = _write_shell_rcfile(spec, workspace_root)
    return ["bash", "--rcfile", str(rcfile), "-i"]


def _installables_for_spec(spec) -> list[str]:
    installables = list(DEFAULT_INSTALLABLES)
    seen = set(installables)
    for package in spec.image.system_packages:
        mapped = SYSTEM_PACKAGE_MAP.get(package)
        if mapped is None or mapped in seen:
            continue
        installables.append(mapped)
        seen.add(mapped)
    return installables


def _run_nix_shell(installables: list[str], command: list[str], *, cwd: Path, env: dict[str, str]) -> int:
    nix_command = [
        "nix",
        "--extra-experimental-features",
        "nix-command",
        "--extra-experimental-features",
        "flakes",
        "shell",
        *installables,
        "--command",
        *command,
    ]
    completed = subprocess.run(nix_command, cwd=cwd, env=env, check=False)
    return completed.returncode


def shutil_which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
