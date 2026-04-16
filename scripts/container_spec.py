#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


WORKSPACE_ROOT = Path("/workspace")


@dataclass(frozen=True)
class ImageConfig:
    base: str = "ubuntu:22.04"
    platform: str | None = None
    system_packages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProjectConfig:
    workdir: str = "."
    default_command: str | list[str] | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    docker_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    mounts: list["MountConfig"] = field(default_factory=list)


@dataclass(frozen=True)
class MountConfig:
    source: str
    target: str | None = None


@dataclass(frozen=True)
class SetupStep:
    name: str
    run: str
    once: str | None = None


@dataclass(frozen=True)
class ContainerSpec:
    name: str
    config_path: Path
    repo_root: Path
    example_dir: Path
    image: ImageConfig
    project: ProjectConfig
    runtime: RuntimeConfig
    setup: list[SetupStep]
    schema: int = 1

    @property
    def example_rel(self) -> Path:
        return self.example_dir.relative_to(self.repo_root)

    @property
    def slug(self) -> str:
        return _slugify(self.name)

    @property
    def container_config_path(self) -> Path:
        return WORKSPACE_ROOT / self.config_path.relative_to(self.repo_root)

    @property
    def container_example_dir(self) -> Path:
        return WORKSPACE_ROOT / self.example_rel

    @property
    def container_workdir(self) -> Path:
        workdir = self.project.workdir.strip()
        if not workdir or workdir == ".":
            return self.container_example_dir
        if workdir.startswith("/"):
            return Path(workdir)
        return self.container_example_dir / workdir

    def resolve_host_mount_source(self, mount: MountConfig) -> Path:
        source = Path(mount.source).expanduser()
        if source.is_absolute():
            return source.resolve()
        return (self.example_dir / source).resolve()

    def resolve_container_mount_target(self, mount: MountConfig) -> Path:
        raw_target = mount.target if mount.target is not None else mount.source
        target = Path(raw_target)
        if target.is_absolute():
            return target
        return self.container_example_dir / target


def load_container_spec(config_path: Path, repo_root: Path) -> ContainerSpec:
    resolved_repo_root = repo_root.resolve()
    resolved_config_path = config_path.resolve()
    raw = tomllib.loads(resolved_config_path.read_text(encoding="utf-8"))
    schema = int(raw.get("schema", 1))
    if schema != 1:
        raise ValueError(f"Unsupported container schema {schema} in {resolved_config_path}")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"container config {resolved_config_path} must define a non-empty name")

    image_raw = _ensure_table(raw.get("image", {}), "image", resolved_config_path)
    runtime_raw = _ensure_table(raw.get("runtime", {}), "runtime", resolved_config_path)
    project_raw = _ensure_table(raw.get("project", {}), "project", resolved_config_path)

    image = ImageConfig(
        base=_as_string(image_raw.get("base", "ubuntu:22.04"), "image.base", resolved_config_path),
        platform=_as_optional_string(image_raw.get("platform"), "image.platform", resolved_config_path),
        system_packages=_as_string_list(
            image_raw.get("system_packages", []),
            "image.system_packages",
            resolved_config_path,
        ),
    )
    project = ProjectConfig(
        workdir=_as_string(project_raw.get("workdir", "."), "project.workdir", resolved_config_path),
        default_command=_as_command(
            project_raw.get("default_command"),
            "project.default_command",
            resolved_config_path,
        ),
    )
    runtime = RuntimeConfig(
        docker_args=_as_string_list(runtime_raw.get("docker_args", []), "runtime.docker_args", resolved_config_path),
        env=_as_string_map(runtime_raw.get("env", {}), "runtime.env", resolved_config_path),
        mounts=_parse_mounts(runtime_raw.get("mounts", []), resolved_config_path),
    )
    setup = _parse_setup(raw.get("setup", []), resolved_config_path)

    return ContainerSpec(
        name=name.strip(),
        config_path=resolved_config_path,
        repo_root=resolved_repo_root,
        example_dir=resolved_config_path.parent,
        image=image,
        project=project,
        runtime=runtime,
        setup=setup,
        schema=schema,
    )


def resolve_example_config(example: str, repo_root: Path) -> Path:
    raw = Path(example)
    if raw.is_absolute():
        candidate = raw
    elif "/" in example or example.startswith("."):
        candidate = (repo_root / raw).resolve()
    else:
        candidate = (repo_root / "examples" / example).resolve()

    if candidate.is_dir():
        candidate = candidate / "container.toml"
    if candidate.name != "container.toml":
        candidate = candidate / "container.toml"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not find container.toml for {example!r} under {candidate}")
    return candidate


def default_command(spec: ContainerSpec) -> list[str]:
    value = spec.project.default_command
    if value is None:
        return ["bash"]
    if isinstance(value, str):
        return ["bash", "-lc", value]
    return list(value)


def resolve_container_relative_path(base_dir: Path, raw_path: str) -> Path:
    raw = raw_path.strip()
    if raw.startswith("/"):
        return Path(raw)
    return base_dir / raw


def _parse_setup(raw_setup: Any, config_path: Path) -> list[SetupStep]:
    if raw_setup is None:
        return []
    if not isinstance(raw_setup, list):
        raise ValueError(f"{config_path}: setup must be an array of tables")
    steps: list[SetupStep] = []
    for index, item in enumerate(raw_setup):
        if not isinstance(item, dict):
            raise ValueError(f"{config_path}: setup[{index}] must be a table")
        name = _as_string(item.get("name", f"step-{index + 1}"), f"setup[{index}].name", config_path)
        run = _as_string(item.get("run"), f"setup[{index}].run", config_path)
        once = item.get("once")
        if once is not None:
            once = _as_string(once, f"setup[{index}].once", config_path)
        steps.append(SetupStep(name=name, run=run, once=once))
    return steps


def _parse_mounts(raw_mounts: Any, config_path: Path) -> list[MountConfig]:
    if raw_mounts is None:
        return []
    if not isinstance(raw_mounts, list):
        raise ValueError(f"{config_path}: runtime.mounts must be an array of tables")
    mounts: list[MountConfig] = []
    for index, item in enumerate(raw_mounts):
        if not isinstance(item, dict):
            raise ValueError(f"{config_path}: runtime.mounts[{index}] must be a table")
        source = _as_string(item.get("source"), f"runtime.mounts[{index}].source", config_path)
        target = item.get("target")
        if target is not None:
            target = _as_string(target, f"runtime.mounts[{index}].target", config_path)
        mounts.append(MountConfig(source=source, target=target))
    return mounts


def _ensure_table(value: Any, field_name: str, config_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{config_path}: {field_name} must be a table")
    return value


def _as_bool(value: Any, field_name: str, config_path: Path) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{config_path}: {field_name} must be a boolean")
    return value


def _as_string(value: Any, field_name: str, config_path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{config_path}: {field_name} must be a non-empty string")
    return value


def _as_optional_string(value: Any, field_name: str, config_path: Path) -> str | None:
    if value is None:
        return None
    return _as_string(value, field_name, config_path)


def _as_string_list(value: Any, field_name: str, config_path: Path) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{config_path}: {field_name} must be an array of strings")
    return list(value)


def _as_string_map(value: Any, field_name: str, config_path: Path) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{config_path}: {field_name} must be a table")
    output: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{config_path}: {field_name} must map strings to strings")
        output[key] = item
    return output


def _as_command(value: Any, field_name: str, config_path: Path) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return list(value)
    raise ValueError(f"{config_path}: {field_name} must be a string or array of strings")


def _slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")
