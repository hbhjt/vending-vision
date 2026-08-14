from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).parents[1]


def test_parallel_install_launches_both_isolated_closures_before_waiting(monkeypatch):
    from scripts import bootstrap_build_envs

    started: list[str] = []
    polled_after: list[tuple[str, ...]] = []
    commands: list[list[str]] = []

    class Process:
        def __init__(self, command):
            self.command = command
            self.name = "core" if "core-wheels" in command else "ai"
            self.returncode = None

        def poll(self):
            polled_after.append(tuple(started))
            self.returncode = 0
            return self.returncode

        def terminate(self):
            raise AssertionError("successful closure must not be terminated")

        def wait(self, timeout=None):
            raise AssertionError("successful closure must not be reaped")

    def popen(command):
        commands.append(command)
        started.append("core" if "core-wheels" in command else "ai")
        return Process(command)

    monkeypatch.setattr(bootstrap_build_envs.subprocess, "Popen", popen)
    monkeypatch.setattr(bootstrap_build_envs.time, "sleep", lambda _: None)

    bootstrap_build_envs._install_isolated_closures(
        ("core", Path("core-python"), Path("core-wheels"), Path("core-lock")),
        ("ai", Path("ai-python"), Path("ai-wheels"), Path("ai-lock")),
    )

    assert started == ["core", "ai"]
    assert polled_after and all(entry == ("core", "ai") for entry in polled_after)
    assert commands == [
        [
            "core-python", "-m", "pip", "install", "--disable-pip-version-check",
            "--no-index", "--find-links", "core-wheels", "--require-hashes", "-r", "core-lock",
        ],
        [
            "ai-python", "-m", "pip", "install", "--disable-pip-version-check",
            "--no-index", "--find-links", "ai-wheels", "--require-hashes", "-r", "ai-lock",
        ],
    ]


@pytest.mark.parametrize("failed_name", ("core", "ai"))
def test_parallel_install_failure_terminates_and_reaps_other_closure(monkeypatch, failed_name):
    from scripts import bootstrap_build_envs

    processes: dict[str, object] = {}

    class Process:
        def __init__(self, name, returncode):
            self.name = name
            self.returncode = returncode
            self.terminated = False
            self.waits: list[float | None] = []

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            raise AssertionError("terminated sibling should be reaped without kill")

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return self.returncode

    def popen(command):
        name = "core" if "core-wheels" in command else "ai"
        process = Process(name, 9 if name == failed_name else None)
        processes[name] = process
        return process

    monkeypatch.setattr(bootstrap_build_envs.subprocess, "Popen", popen)
    monkeypatch.setattr(bootstrap_build_envs.time, "sleep", lambda _: None)

    with pytest.raises(
        bootstrap_build_envs.BuildBootstrapError,
        match=rf"{failed_name}_install_failed:exit_9",
    ):
        bootstrap_build_envs._install_isolated_closures(
            ("core", Path("core-python"), Path("core-wheels"), Path("core-lock")),
            ("ai", Path("ai-python"), Path("ai-wheels"), Path("ai-lock")),
        )

    other_name = "ai" if failed_name == "core" else "core"
    assert processes[other_name].terminated
    assert processes[other_name].waits == [bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS]


def test_parallel_install_kills_a_sibling_that_does_not_reap_in_time(monkeypatch):
    from scripts import bootstrap_build_envs

    events: list[str] = []

    class Process:
        def __init__(self, name, returncode):
            self.name = name
            self.returncode = returncode
            self.wait_count = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("terminate-ai")

        def kill(self):
            events.append("kill-ai")
            self.returncode = -9

        def wait(self, timeout=None):
            self.wait_count += 1
            events.append(f"wait-{self.name}-{timeout}")
            if self.name == "ai" and self.wait_count == 1:
                raise subprocess.TimeoutExpired(self.name, timeout)
            return self.returncode

    def popen(command):
        name = "core" if "core-wheels" in command else "ai"
        return Process(name, 7 if name == "core" else None)

    monkeypatch.setattr(bootstrap_build_envs.subprocess, "Popen", popen)
    monkeypatch.setattr(bootstrap_build_envs.time, "sleep", lambda _: None)

    with pytest.raises(bootstrap_build_envs.BuildBootstrapError, match="core_install_failed:exit_7"):
        bootstrap_build_envs._install_isolated_closures(
            ("core", Path("core-python"), Path("core-wheels"), Path("core-lock")),
            ("ai", Path("ai-python"), Path("ai-wheels"), Path("ai-lock")),
        )

    assert events == [
        "terminate-ai",
        f"wait-ai-{bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS}",
        "kill-ai",
        f"wait-ai-{bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS}",
    ]


def test_parallel_install_reaps_the_first_closure_when_the_second_launch_fails(monkeypatch):
    from scripts import bootstrap_build_envs

    class FirstProcess:
        returncode = None
        alive = True
        waits: list[float | None] = []

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15
            self.alive = False

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return self.returncode

    first = FirstProcess()
    launches = 0

    def popen(_command):
        nonlocal launches
        launches += 1
        if launches == 2:
            raise OSError("second launch failed")
        return first

    monkeypatch.setattr(bootstrap_build_envs.subprocess, "Popen", popen)

    with pytest.raises(
        bootstrap_build_envs.BuildBootstrapError,
        match="build_env_install_launch_failed:OSError",
    ):
        bootstrap_build_envs._install_isolated_closures(
            ("core", Path("core-python"), Path("core-wheels"), Path("core-lock")),
            ("ai", Path("ai-python"), Path("ai-wheels"), Path("ai-lock")),
        )

    assert not first.alive
    assert first.waits == [bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS]


def test_parallel_install_kills_and_reaps_after_terminate_oserror(monkeypatch):
    from scripts import bootstrap_build_envs

    events: list[str] = []

    class Process:
        def __init__(self, name, returncode):
            self.name = name
            self.returncode = returncode
            self.alive = returncode is None

        def poll(self):
            events.append(f"poll-{self.name}")
            return self.returncode

        def terminate(self):
            events.append(f"terminate-{self.name}")
            raise OSError("terminate failed")

        def kill(self):
            events.append(f"kill-{self.name}")
            self.returncode = -9
            self.alive = False

        def wait(self, timeout=None):
            events.append(f"wait-{self.name}-{timeout}")
            return self.returncode

    processes = {
        "core": Process("core", 7),
        "ai": Process("ai", None),
    }

    def popen(command):
        return processes["core" if "core-wheels" in command else "ai"]

    monkeypatch.setattr(bootstrap_build_envs.subprocess, "Popen", popen)

    with pytest.raises(bootstrap_build_envs.BuildBootstrapError, match="core_install_failed:exit_7"):
        bootstrap_build_envs._install_isolated_closures(
            ("core", Path("core-python"), Path("core-wheels"), Path("core-lock")),
            ("ai", Path("ai-python"), Path("ai-wheels"), Path("ai-lock")),
        )

    assert not processes["ai"].alive
    assert events[-3:] == [
        "poll-ai",
        "kill-ai",
        f"wait-ai-{bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS}",
    ]


def test_parallel_install_kills_and_reaps_after_wait_oserror(monkeypatch):
    from scripts import bootstrap_build_envs

    events: list[str] = []

    class Process:
        def __init__(self, name, returncode):
            self.name = name
            self.returncode = returncode
            self.wait_count = 0
            self.alive = returncode is None

        def poll(self):
            events.append(f"poll-{self.name}")
            return self.returncode

        def terminate(self):
            events.append(f"terminate-{self.name}")

        def kill(self):
            events.append(f"kill-{self.name}")
            self.returncode = -9
            self.alive = False

        def wait(self, timeout=None):
            self.wait_count += 1
            events.append(f"wait-{self.name}-{timeout}")
            if self.name == "ai" and self.wait_count == 1:
                raise OSError("wait failed")
            return self.returncode

    processes = {
        "core": Process("core", 8),
        "ai": Process("ai", None),
    }
    monkeypatch.setattr(
        bootstrap_build_envs.subprocess,
        "Popen",
        lambda command: processes["core" if "core-wheels" in command else "ai"],
    )

    with pytest.raises(bootstrap_build_envs.BuildBootstrapError, match="core_install_failed:exit_8"):
        bootstrap_build_envs._install_isolated_closures(
            ("core", Path("core-python"), Path("core-wheels"), Path("core-lock")),
            ("ai", Path("ai-python"), Path("ai-wheels"), Path("ai-lock")),
        )

    assert not processes["ai"].alive
    assert events[-3:] == [
        "poll-ai",
        "kill-ai",
        f"wait-ai-{bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS}",
    ]


def test_parallel_install_enforces_one_total_deadline_and_reaps_both_closures(monkeypatch):
    from scripts import bootstrap_build_envs

    now = 100.0

    class Process:
        def __init__(self):
            self.returncode = None
            self.alive = True
            self.waits: list[float | None] = []

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15
            self.alive = False

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return self.returncode

    processes = [Process(), Process()]

    def popen(_command):
        return processes.pop(0)

    launched: list[Process] = []

    def tracked_popen(command):
        process = popen(command)
        launched.append(process)
        return process

    def sleep(seconds):
        nonlocal now
        now += seconds
        if now > 101.0:
            raise AssertionError("install coordinator ignored its total deadline")

    monkeypatch.setattr(bootstrap_build_envs.subprocess, "Popen", tracked_popen)
    monkeypatch.setattr(bootstrap_build_envs.time, "monotonic", lambda: now)
    monkeypatch.setattr(bootstrap_build_envs.time, "sleep", sleep)
    monkeypatch.setattr(bootstrap_build_envs, "INSTALL_TIMEOUT_SECONDS", 0.1, raising=False)

    with pytest.raises(
        bootstrap_build_envs.BuildBootstrapError,
        match="build_env_install_deadline_exceeded",
    ):
        bootstrap_build_envs._install_isolated_closures(
            ("core", Path("core-python"), Path("core-wheels"), Path("core-lock")),
            ("ai", Path("ai-python"), Path("ai-wheels"), Path("ai-lock")),
        )

    assert len(launched) == 2
    assert all(not process.alive for process in launched)
    assert all(
        process.waits == [bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS]
        for process in launched
    )


def test_parallel_install_reports_bounded_kill_reap_timeout(monkeypatch):
    from scripts import bootstrap_build_envs

    class Process:
        def __init__(self, returncode):
            self.returncode = returncode
            self.waits: list[float | None] = []
            self.killed = False

        def poll(self):
            return self.returncode

        def terminate(self):
            pass

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waits.append(timeout)
            raise subprocess.TimeoutExpired("pip", timeout)

    core = Process(6)
    ai = Process(None)
    monkeypatch.setattr(
        bootstrap_build_envs.subprocess,
        "Popen",
        lambda command: core if "core-wheels" in command else ai,
    )

    with pytest.raises(
        bootstrap_build_envs.BuildBootstrapError,
        match="core_install_failed:exit_6;ai_install_kill_reap_timeout",
    ):
        bootstrap_build_envs._install_isolated_closures(
            ("core", Path("core-python"), Path("core-wheels"), Path("core-lock")),
            ("ai", Path("ai-python"), Path("ai-wheels"), Path("ai-lock")),
        )

    assert ai.killed
    assert ai.waits == [
        bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS,
        bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS,
    ]


def test_parallel_install_reaps_a_sibling_when_cleanup_poll_raises(monkeypatch):
    from scripts import bootstrap_build_envs

    class Process:
        def __init__(self, name, returncode):
            self.name = name
            self.returncode = returncode
            self.poll_count = 0
            self.alive = returncode is None

        def poll(self):
            self.poll_count += 1
            if self.name == "core" and self.poll_count == 2:
                raise OSError("poll failed")
            return self.returncode

        def terminate(self):
            self.returncode = -15
            self.alive = False

        def kill(self):
            self.returncode = -9
            self.alive = False

        def wait(self, timeout=None):
            assert timeout == bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS
            return self.returncode

    core = Process("core", None)
    ai = Process("ai", 5)
    monkeypatch.setattr(
        bootstrap_build_envs.subprocess,
        "Popen",
        lambda command: core if "core-wheels" in command else ai,
    )

    with pytest.raises(bootstrap_build_envs.BuildBootstrapError, match="ai_install_failed:exit_5"):
        bootstrap_build_envs._install_isolated_closures(
            ("core", Path("core-python"), Path("core-wheels"), Path("core-lock")),
            ("ai", Path("ai-python"), Path("ai-wheels"), Path("ai-lock")),
        )

    assert not core.alive


def test_parallel_install_reaps_every_closure_when_active_poll_raises(monkeypatch):
    from scripts import bootstrap_build_envs

    class Process:
        def __init__(self, poll_error=False):
            self.poll_error = poll_error
            self.returncode = None
            self.alive = True

        def poll(self):
            if self.poll_error:
                self.poll_error = False
                raise OSError("poll failed")
            return self.returncode

        def terminate(self):
            self.returncode = -15
            self.alive = False

        def kill(self):
            self.returncode = -9
            self.alive = False

        def wait(self, timeout=None):
            assert timeout == bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS
            return self.returncode

    core = Process(poll_error=True)
    ai = Process()
    monkeypatch.setattr(
        bootstrap_build_envs.subprocess,
        "Popen",
        lambda command: core if "core-wheels" in command else ai,
    )

    with pytest.raises(
        bootstrap_build_envs.BuildBootstrapError,
        match="core_install_poll_failed:OSError",
    ):
        bootstrap_build_envs._install_isolated_closures(
            ("core", Path("core-python"), Path("core-wheels"), Path("core-lock")),
            ("ai", Path("ai-python"), Path("ai-wheels"), Path("ai-lock")),
        )

    assert not core.alive
    assert not ai.alive


def _write_wheel(
    wheelhouse: Path,
    distribution: str,
    module_value: str,
    *,
    version: str = "1.0.0",
    module: str = "cv2",
) -> tuple[Path, str]:
    normalized = distribution.replace("-", "_")
    filename = f"{normalized}-{version}-py3-none-any.whl"
    path = wheelhouse / filename
    dist_info = f"{normalized}-{version}.dist-info"
    files = {
        f"{module}/__init__.py": f"FLAVOR = {module_value!r}\n".encode(),
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n"
        ).encode(),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: stage22-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = "".join(f"{name},,\n" for name in files) + f"{dist_info}/RECORD,,\n"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
        archive.writestr(f"{dist_info}/RECORD", record)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, f"{distribution}=={version} --hash=sha256:{digest}\n"


def _pip_install(python: Path, wheelhouse: Path, requirements: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            str(python), "-m", "pip", "install", "--disable-pip-version-check",
            "--no-index", "--find-links", str(wheelhouse), "--require-hashes", "-r", str(requirements),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _inspect(python: Path) -> dict:
    code = (
        "import cv2, importlib.metadata as m, json, sys; "
        "print(json.dumps({'python':sys.executable,'prefix':sys.prefix,'flavor':cv2.FLAVOR,"
        "'packages':sorted(d.metadata['Name'] for d in m.distributions() if d.metadata.get('Name'))}))"
    )
    completed = subprocess.run(
        [str(python), "-c", code], capture_output=True, text=True, check=True
    )
    return json.loads(completed.stdout)


def test_offline_build_bootstrap_keeps_core_and_ai_distributions_in_distinct_clean_envs(tmp_path):
    core_wheelhouse = tmp_path / "core-wheels"
    ai_wheelhouse = tmp_path / "ai-wheels"
    core_wheelhouse.mkdir()
    ai_wheelhouse.mkdir()
    _, core_lock = _write_wheel(core_wheelhouse, "opencv-contrib-python", "core-contrib")
    _, ai_lock = _write_wheel(ai_wheelhouse, "opencv-python-headless", "ai-headless")
    core_requirements = tmp_path / "requirements-core.txt"
    ai_requirements = tmp_path / "requirements-ai.txt"
    core_requirements.write_text(core_lock, "utf-8")
    ai_requirements.write_text(ai_lock, "utf-8")
    core_env = tmp_path / "core-env"
    ai_env = tmp_path / "ai-env"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "bootstrap_build_envs.py"),
            "--base-python",
            sys.executable,
            "--core-env",
            str(core_env),
            "--core-wheelhouse",
            str(core_wheelhouse),
            "--core-requirements",
            str(core_requirements),
            "--ai-env",
            str(ai_env),
            "--ai-wheelhouse",
            str(ai_wheelhouse),
            "--ai-requirements",
            str(ai_requirements),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    python_relative = Path("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    core = _inspect(core_env / python_relative)
    ai = _inspect(ai_env / python_relative)
    assert core["flavor"] == "core-contrib"
    assert ai["flavor"] == "ai-headless"
    assert "opencv-contrib-python" in core["packages"]
    assert "opencv-python-headless" not in core["packages"]
    assert "opencv-python-headless" in ai["packages"]
    assert "opencv-contrib-python" not in ai["packages"]
    assert Path(core["python"]).resolve() == (core_env / python_relative).resolve()
    assert Path(ai["python"]).resolve() == (ai_env / python_relative).resolve()
    assert core["prefix"] != ai["prefix"]


def test_same_build_env_control_reproduces_core_lock_conflict_and_cv2_overwrite(tmp_path):
    core_wheelhouse = tmp_path / "core-wheels"
    ai_wheelhouse = tmp_path / "ai-wheels"
    core_wheelhouse.mkdir()
    ai_wheelhouse.mkdir()
    _, core_cv2 = _write_wheel(core_wheelhouse, "opencv-contrib-python", "core-contrib")
    _, core_shared = _write_wheel(
        core_wheelhouse, "packaging", "core-shared", version="1.0.0", module="stage22_shared"
    )
    _, ai_cv2 = _write_wheel(ai_wheelhouse, "opencv-python-headless", "ai-headless")
    _, ai_shared = _write_wheel(
        ai_wheelhouse, "packaging", "ai-shared", version="2.0.0", module="stage22_shared"
    )
    core_requirements = tmp_path / "requirements-core.txt"
    ai_requirements = tmp_path / "requirements-ai.txt"
    core_requirements.write_text(core_cv2 + core_shared, "utf-8")
    ai_requirements.write_text(ai_cv2 + ai_shared, "utf-8")
    shared_env = tmp_path / "shared-env"
    subprocess.run([sys.executable, "-m", "venv", str(shared_env)], check=True)
    python_relative = Path("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    shared_python = shared_env / python_relative

    assert _pip_install(shared_python, core_wheelhouse, core_requirements).returncode == 0
    assert _inspect(shared_python)["flavor"] == "core-contrib"
    assert _pip_install(shared_python, ai_wheelhouse, ai_requirements).returncode == 0
    assert _inspect(shared_python)["flavor"] == "ai-headless"

    lock_check = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "dependency_lock.py"),
            "--requirements-lock",
            str(core_requirements),
            "--wheelhouse",
            str(core_wheelhouse),
            "--python",
            str(shared_python),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert lock_check.returncode != 0
    assert "installed dependency has wrong version for packaging: 2.0.0" in (
        lock_check.stdout + lock_check.stderr
    )


def test_ai_build_tool_lock_is_an_exact_hashed_subset_without_core_runtime_packages(tmp_path):
    from scripts.render_ai_build_requirements import render_ai_build_requirements

    output = tmp_path / "requirements-ai-build-tools.txt"
    render_ai_build_requirements(ROOT / "requirements.txt", output)
    rendered = output.read_text("utf-8")
    names = {
        line.split("==", 1)[0]
        for line in rendered.splitlines()
        if line and not line.startswith(" ")
    }

    assert names == {
        "altgraph",
        "pefile",
        "pyinstaller",
        "pyinstaller-hooks-contrib",
        "pywin32-ctypes",
        "setuptools",
    }
    assert "--hash=sha256:" in rendered
    assert "packaging==26.2" not in rendered
    assert "opencv-contrib-python" not in rendered
