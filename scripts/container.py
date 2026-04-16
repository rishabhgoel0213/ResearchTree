#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from container_spec import WORKSPACE_ROOT, load_container_spec, resolve_example_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
DEFAULT_IN_CONTAINER_SCRIPT = Path("/workspace/docker/in_container.py")
HOST_CODEX_DIR = Path.home() / ".codex"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and run example containers for ResearchTree.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the image for an example.")
    _add_example_argument(build_parser)
    build_parser.add_argument("--rebuild", action="store_true", help="Always rebuild the image.")

    run_parser = subparsers.add_parser("run", help="Run an example container.")
    _add_example_argument(run_parser)
    run_parser.add_argument("--rebuild", action="store_true", help="Rebuild the image before running.")
    run_parser.add_argument("--skip-build", action="store_true", help="Skip image build checks.")
    run_parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Run configured setup steps and exit without launching the default command.",
    )
    run_parser.add_argument("container_command", nargs=argparse.REMAINDER)

    shell_parser = subparsers.add_parser("shell", help="Open a shell in an example container.")
    _add_example_argument(shell_parser)
    shell_parser.add_argument("--rebuild", action="store_true", help="Rebuild the image before running.")
    shell_parser.add_argument("--skip-build", action="store_true", help="Skip image build checks.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if shutil.which("docker") is None:
        parser.exit(status=1, message="error: docker is not installed or not on PATH\n")

    config_path = resolve_example_config(args.example, REPO_ROOT)
    spec = load_container_spec(config_path, REPO_ROOT)
    image_tag = image_tag_for_spec(spec)

    if args.command == "build":
        build_image(spec, image_tag=image_tag, rebuild=args.rebuild)
        print(image_tag)
        return 0

    if not getattr(args, "skip_build", False):
        build_image(spec, image_tag=image_tag, rebuild=getattr(args, "rebuild", False))

    if args.command == "shell":
        return run_container(spec, image_tag=image_tag, override_command=["bash"])

    override_command = _normalize_override_command(args.container_command)
    return run_container(
        spec,
        image_tag=image_tag,
        override_command=override_command,
        setup_only=args.setup_only,
    )


def build_image(spec, *, image_tag: str, rebuild: bool) -> None:
    if not rebuild and image_exists(image_tag):
        print(f"[container] using existing image {image_tag}")
        return

    command = [
        "docker",
        "build",
        "-f",
        str(DOCKERFILE),
        "-t",
        image_tag,
        "--build-arg",
        f"BASE_IMAGE={spec.image.base}",
    ]
    if spec.image.platform:
        command.extend(["--platform", spec.image.platform])
    if spec.image.system_packages:
        command.extend(["--build-arg", f"SYSTEM_PACKAGES={' '.join(spec.image.system_packages)}"])
    command.append(str(REPO_ROOT))
    print(f"[container] building {image_tag}")
    subprocess.run(command, check=True)


def run_container(
    spec,
    *,
    image_tag: str,
    override_command: list[str] | None = None,
    setup_only: bool = False,
) -> int:
    workspace_volume = f"researchtree-{spec.slug}-workspace"
    home_volume = f"researchtree-{spec.slug}-home"
    pixi_volume = f"researchtree-{spec.slug}-pixi"
    state_volume = f"researchtree-{spec.slug}-state"
    sync_workspace(spec, image_tag=image_tag, workspace_volume=workspace_volume)
    ensure_volume_permissions(spec, image_tag=image_tag, volume_name=home_volume)
    ensure_volume_permissions(spec, image_tag=image_tag, volume_name=pixi_volume)
    ensure_volume_permissions(spec, image_tag=image_tag, volume_name=state_volume)

    command = ["docker", "run", "--rm", "--init"]
    if spec.image.platform:
        command.extend(["--platform", spec.image.platform])
    if sys.stdin.isatty() and sys.stdout.isatty():
        command.extend(["-it"])
    command.extend(["-v", f"{workspace_volume}:{WORKSPACE_ROOT}"])
    command.extend(["-v", f"{home_volume}:/home/researchtree"])
    command.extend(["-v", f"{pixi_volume}:{spec.container_example_dir / '.pixi'}"])
    command.extend(["-v", f"{state_volume}:{spec.container_example_dir / '.container_state'}"])
    for mount in spec.runtime.mounts:
        source = spec.resolve_host_mount_source(mount)
        target = spec.resolve_container_mount_target(mount)
        command.extend(["-v", f"{source}:{target}"])
    if HOST_CODEX_DIR.exists():
        command.extend(["-v", f"{HOST_CODEX_DIR}:/home/researchtree/.codex"])
    command.extend(["-w", str(WORKSPACE_ROOT)])
    command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

    home_env = {
        "HOME": "/home/researchtree",
        "PIXI_HOME": "/home/researchtree/.pixi",
        "PIP_CACHE_DIR": "/home/researchtree/.cache/pip",
        "XDG_CACHE_HOME": "/home/researchtree/.cache",
        "USER": "researchtree",
        "LOGNAME": "researchtree",
    }
    for key, value in {**home_env, **spec.runtime.env}.items():
        command.extend(["-e", f"{key}={value}"])
    command.extend(spec.runtime.docker_args)
    command.append(image_tag)
    command.extend(["python3", str(DEFAULT_IN_CONTAINER_SCRIPT), str(spec.container_config_path)])
    if setup_only:
        command.append("--setup-only")
    if override_command is not None:
        command.append("--")
        command.extend(override_command)

    print(f"[container] running {spec.name} in {image_tag}")
    completed = subprocess.run(command, check=False)
    return completed.returncode


def image_exists(image_tag: str) -> bool:
    completed = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def sync_workspace(spec, *, image_tag: str, workspace_volume: str) -> None:
    clear_volume(spec, image_tag=image_tag, volume_name=workspace_volume, target=WORKSPACE_ROOT)

    extract_command = ["docker", "run", "--rm", "-i"]
    if spec.image.platform:
        extract_command.extend(["--platform", spec.image.platform])
    extract_command.extend(
        [
            "-v",
            f"{workspace_volume}:{WORKSPACE_ROOT}",
            image_tag,
            "bash",
            "-lc",
            f"mkdir -p {WORKSPACE_ROOT} && tar -xf - -C {WORKSPACE_ROOT}",
        ]
    )

    print(f"[container] syncing workspace snapshot for {spec.name}")
    with subprocess.Popen(extract_command, stdin=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
            add_repo_to_tar(tar, spec)
        proc.stdin.close()
        returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, extract_command)
    ensure_workspace_mount_targets(spec, image_tag=image_tag, workspace_volume=workspace_volume)
    ensure_volume_permissions(spec, image_tag=image_tag, volume_name=workspace_volume)


def add_repo_to_tar(tar: tarfile.TarFile, spec) -> None:
    excluded_roots = {spec.resolve_host_mount_source(mount) for mount in spec.runtime.mounts}

    def is_excluded(path: Path) -> bool:
        if path == REPO_ROOT:
            return False
        if path.name == ".DS_Store":
            return True
        relative_parts = path.relative_to(REPO_ROOT).parts
        if any(part in TRANSIENT_DIR_NAMES for part in relative_parts):
            return True
        for excluded_root in excluded_roots:
            if path == excluded_root or excluded_root in path.parents:
                return True
        return False

    for root, dirs, files in os.walk(REPO_ROOT, topdown=True):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if not is_excluded(root_path / name)]
        rel_root = root_path.relative_to(REPO_ROOT)
        if rel_root != Path("."):
            tar.add(root_path, arcname=str(rel_root), recursive=False)
        for file_name in files:
            file_path = root_path / file_name
            if is_excluded(file_path):
                continue
            tar.add(file_path, arcname=str(file_path.relative_to(REPO_ROOT)), recursive=False)


def ensure_volume_permissions(spec, *, image_tag: str, volume_name: str) -> None:
    command = ["docker", "run", "--rm"]
    if spec.image.platform:
        command.extend(["--platform", spec.image.platform])
    command.extend(
        [
            "-v",
            f"{volume_name}:/target",
            image_tag,
            "bash",
            "-lc",
            f"mkdir -p /target && chown -R {os.getuid()}:{os.getgid()} /target",
        ]
    )
    subprocess.run(command, check=True)


def clear_volume(spec, *, image_tag: str, volume_name: str, target: Path) -> None:
    command = ["docker", "run", "--rm"]
    if spec.image.platform:
        command.extend(["--platform", spec.image.platform])
    command.extend(
        [
            "-v",
            f"{volume_name}:{target}",
            image_tag,
            "bash",
            "-lc",
            (
                f"shopt -s dotglob nullglob && mkdir -p {target} && "
                f"for path in {target}/*; do rm -rf \"$path\"; done"
            ),
        ]
    )
    subprocess.run(command, check=True)


def ensure_workspace_mount_targets(spec, *, image_tag: str, workspace_volume: str) -> None:
    if not spec.runtime.mounts:
        return
    mkdir_targets = " ".join(str(spec.resolve_container_mount_target(mount)) for mount in spec.runtime.mounts)
    command = ["docker", "run", "--rm"]
    if spec.image.platform:
        command.extend(["--platform", spec.image.platform])
    command.extend(
        [
            "-v",
            f"{workspace_volume}:{WORKSPACE_ROOT}",
            image_tag,
            "bash",
            "-lc",
            f"mkdir -p {mkdir_targets}",
        ]
    )
    subprocess.run(command, check=True)


def image_tag_for_spec(spec) -> str:
    payload = {
        "base": spec.image.base,
        "platform": spec.image.platform,
        "system_packages": spec.image.system_packages,
        "dockerfile_sha256": hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"researchtree-{spec.slug}:{digest}"


def _normalize_override_command(raw: list[str]) -> list[str] | None:
    if not raw:
        return None
    if raw[0] == "--":
        raw = raw[1:]
    if not raw:
        return None
    return raw


def _add_example_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "example",
        help="Example name under examples/, an example directory path, or a container.toml path.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
