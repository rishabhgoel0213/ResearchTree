from __future__ import annotations

import os
from pathlib import Path
import pty
import select
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


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
    os.execv(str(target), [str(target), "-m", "unittest", *sys.argv[1:]])


_ensure_compatible_python()


def _python_executable() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _shell_test_script() -> str:
    return "\n".join(
        [
            "pwd",
            "test -f hello.py && echo FILE_OK",
            "printf 'snapshot only\\n' > snapshot_only.txt",
            "test -f snapshot_only.txt && echo SNAPSHOT_OK",
            "treegit init",
            "treegit status",
            "exit",
            "",
        ]
    )


def _assert_shell_behavior(
    testcase: unittest.TestCase,
    completed: subprocess.CompletedProcess[str],
    *,
    source_src_dir: Path,
) -> Path:
    combined_output = completed.stdout + completed.stderr
    testcase.assertEqual(completed.returncode, 0, combined_output)
    testcase.assertIn("FILE_OK", combined_output)
    testcase.assertIn("SNAPSHOT_OK", combined_output)
    testcase.assertIn("snapshot_only.txt", combined_output)
    testcase.assertIn("hello.py", combined_output)
    testcase.assertIn("Initialized empty TreeGit repository", combined_output)

    pwd_lines = [line.strip() for line in combined_output.splitlines() if line.strip().endswith("/src")]
    testcase.assertTrue(pwd_lines, combined_output)
    staged_src = Path(pwd_lines[0])
    testcase.assertNotEqual(staged_src.resolve(), source_src_dir.resolve())
    testcase.assertFalse((source_src_dir / "snapshot_only.txt").exists())
    testcase.assertEqual(
        (source_src_dir / "hello.py").read_text(encoding="utf-8"),
        "print('hello from source fixture')\n",
    )
    return staged_src


def _run_with_pty(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    script: str,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    master_fd, slave_fd = pty.openpty()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=False,
            close_fds=True,
        )
    finally:
        os.close(slave_fd)

    os.write(master_fd, script.encode("utf-8"))

    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                raise subprocess.TimeoutExpired(command, timeout)

            ready, _, _ = select.select([master_fd], [], [], min(0.1, remaining))
            if ready:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    data = b""
                if data:
                    chunks.append(data)

            if process.poll() is not None:
                while True:
                    try:
                        data = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not data:
                        break
                    chunks.append(data)
                break
    finally:
        os.close(master_fd)

    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=b"".join(chunks).decode("utf-8", errors="replace"),
        stderr="",
    )


def _docker_is_usable() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "docker is not installed"
    completed = subprocess.run(
        ["docker", "info"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "docker info failed"
        return False, detail
    return True, ""


def _nix_is_usable() -> tuple[bool, str]:
    if shutil.which("nix") is None:
        return False, "nix is not installed"
    completed = subprocess.run(
        ["nix", "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "nix --version failed"
        return False, detail
    return True, ""


class _ExternalExampleFixture(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.temp_root = Path(self.tempdir.name)
        self.external_example_dir = self.temp_root / "external-example"
        self._create_external_example()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _create_external_example(self) -> None:
        unique_name = self.temp_root.name.replace("-", " ").strip() or "fixture"
        (self.external_example_dir / "src").mkdir(parents=True, exist_ok=True)
        (self.external_example_dir / "src" / "hello.py").write_text(
            "print('hello from source fixture')\n",
            encoding="utf-8",
        )
        (self.external_example_dir / "container.toml").write_text(
            textwrap.dedent(
                f"""
                schema = 1
                name = "External TreeGit Fixture {unique_name}"

                [image]
                base = "ubuntu:22.04"

                [project]
                workdir = "src"
                default_command = "bash"

                [runtime]
                docker_args = []
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )


class ContractContainerLauncherShellTests(_ExternalExampleFixture):
    def setUp(self) -> None:
        super().setUp()
        self.fake_repo_root = self.temp_root / "repo"
        self.fake_bin_dir = self.temp_root / "fake-bin"
        self.fake_docker_state_dir = self.temp_root / "fake-docker-state"
        self.fake_cache_home = self.temp_root / "fake-cache"
        self.fake_home = self.temp_root / "home"
        self._create_fake_repo()
        self._create_fake_backends()

    def test_container_py_shell_works_for_external_example(self) -> None:
        self._assert_shell_launcher_works("container.py")

    def test_container_nix_shell_works_for_external_example(self) -> None:
        self._assert_shell_launcher_works("container_nix.py")

    def _assert_shell_launcher_works(self, launcher_filename: str) -> None:
        launcher_path = self.fake_repo_root / "scripts" / launcher_filename
        completed = subprocess.run(
            [_python_executable(), str(launcher_path), "shell", str(self.external_example_dir)],
            cwd=self.fake_repo_root,
            env=self._launcher_env(),
            input=_shell_test_script(),
            text=True,
            capture_output=True,
            check=False,
        )

        staged_src = _assert_shell_behavior(
            self,
            completed,
            source_src_dir=self.external_example_dir / "src",
        )
        self.assertTrue((staged_src / "snapshot_only.txt").exists())

    def _create_fake_repo(self) -> None:
        shutil.copytree(
            REPO_ROOT / "scripts",
            self.fake_repo_root / "scripts",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(
            REPO_ROOT / "docker",
            self.fake_repo_root / "docker",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(
            REPO_ROOT / "treegit",
            self.fake_repo_root / "treegit",
            ignore=shutil.ignore_patterns(
                ".git",
                ".mypy_cache",
                ".pytest_cache",
                "__pycache__",
                "*.pyc",
            ),
        )

    def _create_fake_backends(self) -> None:
        self.fake_bin_dir.mkdir(parents=True, exist_ok=True)
        (self.fake_home / ".bashrc").parent.mkdir(parents=True, exist_ok=True)
        (self.fake_home / ".bashrc").write_text("# test shell rc\n", encoding="utf-8")
        self._write_fake_nix()
        self._write_fake_docker()

    def _write_fake_nix(self) -> None:
        nix_path = self.fake_bin_dir / "nix"
        nix_path.write_text(
            textwrap.dedent(
                """\
                #!__PYTHON__
                from __future__ import annotations

                import subprocess
                import sys


                def main(argv: list[str]) -> int:
                    if "--command" not in argv:
                        return 1
                    command = argv[argv.index("--command") + 1 :]
                    completed = subprocess.run(command, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, check=False)
                    return completed.returncode


                if __name__ == "__main__":
                    raise SystemExit(main(sys.argv[1:]))
                """
            ).replace("__PYTHON__", _python_executable()),
            encoding="utf-8",
        )
        nix_path.chmod(0o755)

    def _write_fake_docker(self) -> None:
        docker_path = self.fake_bin_dir / "docker"
        docker_path.write_text(
            textwrap.dedent(
                """\
                #!__PYTHON__
                from __future__ import annotations

                import hashlib
                import json
                import os
                from pathlib import Path
                import shutil
                import shlex
                import subprocess
                import sys
                import tarfile
                import tempfile
                import textwrap
                import importlib.util


                STATE_DIR = Path(os.environ["FAKE_DOCKER_STATE_DIR"])
                REPO_ROOT = Path(os.environ["FAKE_REPO_ROOT"])


                def _load_module(path: Path, name: str):
                    spec = importlib.util.spec_from_file_location(name, path)
                    assert spec is not None and spec.loader is not None
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    return module


                container_spec = _load_module(REPO_ROOT / "scripts" / "container_spec.py", "fake_container_spec")


                def main(argv: list[str]) -> int:
                    STATE_DIR.mkdir(parents=True, exist_ok=True)
                    if not argv:
                        return 1
                    command = argv[0]
                    if command == "buildx" and argv[1:] == ["version"]:
                        return 1
                    if command == "build":
                        return handle_build(argv[1:])
                    if command == "image" and len(argv) >= 2 and argv[1] == "inspect":
                        return handle_image_inspect(argv[2:])
                    if command == "info":
                        print("[]")
                        return 0
                    if command == "inspect":
                        return handle_inspect(argv[1:])
                    if command == "rm":
                        return handle_rm(argv[1:])
                    if command == "run":
                        return handle_run(argv[1:])
                    if command == "create":
                        return handle_create(argv[1:])
                    if command == "start":
                        return handle_start(argv[1:])
                    if command == "exec":
                        return handle_exec(argv[1:])
                    return 1


                def images_path() -> Path:
                    return STATE_DIR / "images.json"


                def containers_dir() -> Path:
                    path = STATE_DIR / "containers"
                    path.mkdir(parents=True, exist_ok=True)
                    return path


                def volumes_dir() -> Path:
                    path = STATE_DIR / "volumes"
                    path.mkdir(parents=True, exist_ok=True)
                    return path


                def load_json(path: Path, default):
                    if not path.exists():
                        return default
                    return json.loads(path.read_text(encoding="utf-8"))


                def save_json(path: Path, payload) -> None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


                def parse_volume_source(raw: str) -> Path:
                    source = Path(raw)
                    if source.is_absolute():
                        return source
                    path = volumes_dir() / raw
                    path.mkdir(parents=True, exist_ok=True)
                    return path


                def map_container_path(mounts: list[dict[str, str]], container_path: str) -> Path:
                    for mount in sorted(mounts, key=lambda item: len(item["target"]), reverse=True):
                        target = mount["target"]
                        if container_path == target:
                            return Path(mount["source"])
                        prefix = target + "/"
                        if container_path.startswith(prefix):
                            return Path(mount["source"]) / container_path[len(prefix) :]
                    raise KeyError(container_path)


                def ensure_treegit_wrapper(workspace_root: Path) -> Path:
                    bin_dir = workspace_root.parent / ".fake-docker-bin"
                    bin_dir.mkdir(parents=True, exist_ok=True)
                    wrapper = bin_dir / "treegit"
                    wrapper.write_text(
                        textwrap.dedent(
                            f\"\"\"\
                            #!/usr/bin/env bash
                            set -euo pipefail
                            export PYTHONPATH={shlex.quote(str(workspace_root / "treegit" / "src"))}${{PYTHONPATH:+:$PYTHONPATH}}
                            exec {shlex.quote(sys.executable)} -m treegit "$@"
                            \"\"\"
                        ),
                        encoding="utf-8",
                    )
                    wrapper.chmod(0o755)
                    return bin_dir


                def handle_build(argv: list[str]) -> int:
                    tag = argv[argv.index("-t") + 1]
                    images = load_json(images_path(), {})
                    images[tag] = f"sha256:{hashlib.sha256(tag.encode('utf-8')).hexdigest()}"
                    save_json(images_path(), images)
                    return 0


                def handle_image_inspect(argv: list[str]) -> int:
                    images = load_json(images_path(), {})
                    if not argv:
                        return 1
                    if argv[0] == "-f":
                        tag = argv[2]
                    else:
                        tag = argv[0]
                    if tag not in images:
                        return 1
                    if argv[0] == "-f":
                        print(images[tag])
                    else:
                        print("{}")
                    return 0


                def handle_inspect(argv: list[str]) -> int:
                    if len(argv) != 3 or argv[0] != "-f":
                        return 1
                    container_name = argv[2]
                    state_path = containers_dir() / f"{container_name}.json"
                    if not state_path.exists():
                        return 1
                    payload = load_json(state_path, {})
                    print(f"{payload['state']}\\t{payload['image_id']}")
                    return 0


                def handle_rm(argv: list[str]) -> int:
                    container_name = argv[-1]
                    state_path = containers_dir() / f"{container_name}.json"
                    if state_path.exists():
                        state_path.unlink()
                    return 0


                def handle_run(argv: list[str]) -> int:
                    mounts, image_tag, command = parse_runtime_args(argv)
                    del image_tag
                    shell_command = " ".join(command)
                    if "tar -xf - -C" in shell_command:
                        target = command[-1].split("tar -xf - -C ", 1)[1]
                        host_target = map_container_path(mounts, target)
                        host_target.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(delete=False) as handle:
                            data = sys.stdin.buffer.read()
                            handle.write(data)
                            temp_name = handle.name
                        with tarfile.open(temp_name, mode="r") as archive:
                            try:
                                archive.extractall(host_target, filter="fully_trusted")
                            except TypeError:
                                archive.extractall(host_target)
                        Path(temp_name).unlink()
                        return 0
                    if "for path in " in shell_command and "rm -rf" in shell_command:
                        target = command[-1].split("mkdir -p ", 1)[1].split(" && ", 1)[0]
                        host_target = map_container_path(mounts, target)
                        host_target.mkdir(parents=True, exist_ok=True)
                        for child in host_target.iterdir():
                            if child.is_dir() and not child.is_symlink():
                                shutil.rmtree(child)
                            else:
                                child.unlink()
                        return 0
                    if command[:2] == ["bash", "-lc"] and command[-1].startswith("mkdir -p "):
                        mkdir_targets = command[-1][len("mkdir -p ") :].split(" && ", 1)[0].split()
                        for target in mkdir_targets:
                            map_container_path(mounts, target).mkdir(parents=True, exist_ok=True)
                        return 0
                    return 0


                def handle_create(argv: list[str]) -> int:
                    images = load_json(images_path(), {})
                    container_name = argv[argv.index("--name") + 1]
                    mounts, image_tag, command = parse_runtime_args(argv)
                    del command
                    env = {}
                    workdir = "."
                    for index, item in enumerate(argv):
                        if item == "-e":
                            key, value = argv[index + 1].split("=", 1)
                            env[key] = value
                        if item == "-w":
                            workdir = argv[index + 1]
                    payload = {
                        "state": "created",
                        "image_id": images[image_tag],
                        "image_tag": image_tag,
                        "mounts": mounts,
                        "env": env,
                        "workdir": workdir,
                    }
                    save_json(containers_dir() / f"{container_name}.json", payload)
                    return 0


                def handle_start(argv: list[str]) -> int:
                    container_name = argv[-1]
                    path = containers_dir() / f"{container_name}.json"
                    payload = load_json(path, {})
                    payload["state"] = "running"
                    save_json(path, payload)
                    return 0


                def handle_exec(argv: list[str]) -> int:
                    while argv and argv[0] in {"-i", "-t", "-it"}:
                        argv = argv[1:]
                    container_name = argv[0]
                    path = containers_dir() / f"{container_name}.json"
                    payload = load_json(path, {})
                    mounts = payload["mounts"]
                    workspace_root = map_container_path(mounts, "/workspace")
                    treegit_bin_dir = ensure_treegit_wrapper(workspace_root)
                    env = os.environ.copy()
                    env.update(payload["env"])
                    env["PATH"] = str(treegit_bin_dir) + os.pathsep + env.get("PATH", "")

                    command = argv[1:]
                    if command[:2] != ["python3", "/workspace/docker/in_container.py"]:
                        completed = subprocess.run(command, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, env=env, check=False)
                        return completed.returncode

                    config_host_path = map_container_path(mounts, command[2])
                    spec = container_spec.load_container_spec(config_host_path, workspace_root)
                    host_workdir = workspace_root / spec.example_rel / spec.project.workdir
                    command_args = command[3:]
                    if command_args and command_args[0] == "--":
                        command_args = command_args[1:]
                    if not command_args:
                        command_args = ["bash"]
                    completed = subprocess.run(
                        command_args,
                        cwd=host_workdir,
                        env=env,
                        stdin=sys.stdin,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                        check=False,
                    )
                    return completed.returncode


                def parse_runtime_args(argv: list[str]) -> tuple[list[dict[str, str]], str, list[str]]:
                    mounts: list[dict[str, str]] = []
                    index = 0
                    while index < len(argv):
                        item = argv[index]
                        if item == "-v":
                            source, target = argv[index + 1].split(":", 1)
                            mounts.append({"source": str(parse_volume_source(source)), "target": target})
                            index += 2
                            continue
                        if item in {"--rm", "-i", "--init"}:
                            index += 1
                            continue
                        if item in {"--platform", "--name", "--user", "-w", "-e"}:
                            index += 2
                            continue
                        if item.startswith("-"):
                            index += 1
                            continue
                        return mounts, item, argv[index + 1 :]
                    raise RuntimeError("Could not parse docker runtime arguments")


                if __name__ == "__main__":
                    raise SystemExit(main(sys.argv[1:]))
                """
            ).replace("__PYTHON__", _python_executable()),
            encoding="utf-8",
        )
        docker_path.chmod(0o755)

    def _launcher_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PATH"] = str(self.fake_bin_dir) + os.pathsep + env.get("PATH", "")
        env["FAKE_REPO_ROOT"] = str(self.fake_repo_root)
        env["FAKE_DOCKER_STATE_DIR"] = str(self.fake_docker_state_dir)
        env["XDG_CACHE_HOME"] = str(self.fake_cache_home)
        env["HOME"] = str(self.fake_home)
        return env


class RealContainerLauncherShellIntegrationTests(_ExternalExampleFixture):
    def test_container_py_shell_works_for_external_example(self) -> None:
        usable, reason = _docker_is_usable()
        if not usable:
            self.skipTest(reason)

        launcher_path = REPO_ROOT / "scripts" / "container.py"
        try:
            completed = _run_with_pty(
                [_python_executable(), str(launcher_path), "shell", "--fresh", str(self.external_example_dir)],
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                script=_shell_test_script(),
                timeout=900,
            )
        finally:
            subprocess.run(
                [_python_executable(), str(launcher_path), "stop", str(self.external_example_dir)],
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

        staged_src = _assert_shell_behavior(
            self,
            completed,
            source_src_dir=self.external_example_dir / "src",
        )
        self.assertTrue(str(staged_src).startswith("/workspace/"), completed.stdout + completed.stderr)

    def test_container_nix_shell_works_for_external_example(self) -> None:
        usable, reason = _nix_is_usable()
        if not usable:
            self.skipTest(reason)

        launcher_path = REPO_ROOT / "scripts" / "container_nix.py"
        completed = subprocess.run(
            [_python_executable(), str(launcher_path), "shell", "--fresh", str(self.external_example_dir)],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            input=_shell_test_script(),
            text=True,
            capture_output=True,
            check=False,
            timeout=900,
        )

        staged_src = _assert_shell_behavior(
            self,
            completed,
            source_src_dir=self.external_example_dir / "src",
        )
        self.assertIn("researchtree-container-nix", str(staged_src))


if __name__ == "__main__":
    unittest.main()
