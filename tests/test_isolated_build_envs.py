import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_core_install_uses_exact_offline_hashed_command_and_reaps(monkeypatch):
    from scripts import bootstrap_build_envs

    class Process:
        def __init__(self):
            self.waits = []

        def poll(self):
            return 0

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return 0

    process = Process()
    commands = []
    monkeypatch.setattr(
        bootstrap_build_envs.subprocess,
        "Popen",
        lambda command: commands.append(command) or process,
    )

    bootstrap_build_envs._install_core_closure(
        Path("core-python"), Path("core-wheels"), Path("core-lock")
    )

    assert commands == [[
        "core-python", "-m", "pip", "install", "--disable-pip-version-check",
        "--no-index", "--find-links", "core-wheels", "--require-hashes",
        "-r", "core-lock",
    ]]
    assert process.waits == [bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS]


def test_core_install_nonzero_exit_is_reaped_and_fails_closed(monkeypatch):
    from scripts import bootstrap_build_envs

    class Process:
        def __init__(self):
            self.waits = []

        def poll(self):
            return 7

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return 7

    process = Process()
    monkeypatch.setattr(bootstrap_build_envs.subprocess, "Popen", lambda _command: process)

    with pytest.raises(
        bootstrap_build_envs.BuildBootstrapError,
        match="core_install_failed:exit_7",
    ):
        bootstrap_build_envs._install_core_closure(
            Path("core-python"), Path("core-wheels"), Path("core-lock")
        )

    assert process.waits == [bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS]


def test_core_install_launch_failure_is_not_reported_as_success(monkeypatch):
    from scripts import bootstrap_build_envs

    def fail_launch(_command):
        raise OSError("launch failed")

    monkeypatch.setattr(bootstrap_build_envs.subprocess, "Popen", fail_launch)

    with pytest.raises(
        bootstrap_build_envs.BuildBootstrapError,
        match="core_install_launch_failed:OSError",
    ):
        bootstrap_build_envs._install_core_closure(
            Path("core-python"), Path("core-wheels"), Path("core-lock")
        )


def test_core_install_total_deadline_terminates_and_reaps(monkeypatch):
    from scripts import bootstrap_build_envs

    now = 10.0

    class Process:
        def __init__(self):
            self.terminated = False
            self.waits = []

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return -15

    process = Process()

    def sleep(seconds):
        nonlocal now
        now += seconds

    monkeypatch.setattr(bootstrap_build_envs.subprocess, "Popen", lambda _command: process)
    monkeypatch.setattr(bootstrap_build_envs.time, "monotonic", lambda: now)
    monkeypatch.setattr(bootstrap_build_envs.time, "sleep", sleep)
    monkeypatch.setattr(bootstrap_build_envs, "INSTALL_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(
        bootstrap_build_envs.BuildBootstrapError,
        match="core_install_deadline_exceeded",
    ):
        bootstrap_build_envs._install_core_closure(
            Path("core-python"), Path("core-wheels"), Path("core-lock")
        )

    assert process.terminated
    assert process.waits == [bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS]


def test_core_cleanup_waits_again_after_kill_raises(monkeypatch):
    from scripts import bootstrap_build_envs

    class Process:
        def __init__(self):
            self.waits = []

        def terminate(self):
            return None

        def kill(self):
            raise OSError("kill raced with exit")

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                raise subprocess.TimeoutExpired("pip", timeout)
            return -9

    process = Process()
    diagnostic = bootstrap_build_envs._terminate_and_reap(process)

    assert diagnostic == "kill_OSError"
    assert process.waits == [
        bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS,
        bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS,
    ]


def test_core_cleanup_reports_bounded_timeout_after_kill(monkeypatch):
    from scripts import bootstrap_build_envs

    class Process:
        def __init__(self):
            self.killed = False
            self.waits = []

        def terminate(self):
            return None

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waits.append(timeout)
            raise subprocess.TimeoutExpired("pip", timeout)

    process = Process()
    diagnostic = bootstrap_build_envs._terminate_and_reap(process)

    assert process.killed
    assert diagnostic == "kill_reap_timeout"
    assert process.waits == [
        bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS,
        bootstrap_build_envs.INSTALL_REAP_TIMEOUT_SECONDS,
    ]


@pytest.mark.parametrize(
    ("missing", "diagnostic"),
    [
        ("base", "base_python_missing"),
        ("wheelhouse", "core_wheelhouse_missing"),
        ("requirements", "core_requirements_missing"),
    ],
)
def test_core_bootstrap_requires_every_input(tmp_path, missing, diagnostic):
    from scripts.bootstrap_build_envs import BuildBootstrapError, bootstrap_build_envs

    base_python = tmp_path / "python"
    base_python.write_bytes(b"python")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("", "utf-8")
    if missing == "base":
        base_python.unlink()
    elif missing == "wheelhouse":
        wheelhouse.rmdir()
    else:
        requirements.unlink()

    with pytest.raises(BuildBootstrapError, match=diagnostic):
        bootstrap_build_envs(
            base_python=base_python,
            core_env=tmp_path / "core-env",
            core_wheelhouse=wheelhouse,
            core_requirements=requirements,
        )


def test_core_bootstrap_rejects_a_preexisting_environment(tmp_path):
    from scripts.bootstrap_build_envs import BuildBootstrapError, bootstrap_build_envs

    core_env = tmp_path / "core-env"
    core_env.mkdir()
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("", "utf-8")

    with pytest.raises(BuildBootstrapError, match="build_env_must_not_exist"):
        bootstrap_build_envs(
            base_python=Path(sys.executable),
            core_env=core_env,
            core_wheelhouse=wheelhouse,
            core_requirements=requirements,
        )


def _write_wheel(
    wheelhouse: Path,
    distribution: str,
    module_value: str,
    *,
    version: str = "1.0.0",
    module: str = "cv2",
) -> tuple[Path, str]:
    normalized = distribution.replace("-", "_")
    path = wheelhouse / f"{normalized}-{version}-py3-none-any.whl"
    dist_info = f"{normalized}-{version}.dist-info"
    files = {
        f"{module}/__init__.py": f"FLAVOR = {module_value!r}\n".encode(),
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: core-build-test\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    record = "".join(f"{name},,\n" for name in files) + f"{dist_info}/RECORD,,\n"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
        archive.writestr(f"{dist_info}/RECORD", record)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, f"{distribution}=={version} --hash=sha256:{digest}\n"


def _pip_install(
    python: Path, wheelhouse: Path, requirements: Path
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            str(python), "-m", "pip", "install", "--disable-pip-version-check",
            "--no-index", "--find-links", str(wheelhouse), "--require-hashes",
            "-r", str(requirements),
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


def test_offline_build_bootstrap_creates_only_one_clean_core_environment(tmp_path):
    core_wheelhouse = tmp_path / "core-wheels"
    core_wheelhouse.mkdir()
    _, core_lock = _write_wheel(
        core_wheelhouse, "opencv-contrib-python", "core-contrib"
    )
    core_requirements = tmp_path / "requirements-core.txt"
    core_requirements.write_text(core_lock, "utf-8")
    core_env = tmp_path / "core-env"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "bootstrap_build_envs.py"),
            "--base-python", sys.executable,
            "--core-env", str(core_env),
            "--core-wheelhouse", str(core_wheelhouse),
            "--core-requirements", str(core_requirements),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    python_relative = Path(
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    core = _inspect(core_env / python_relative)
    assert result == {"corePython": str((core_env / python_relative).absolute())}
    assert core["flavor"] == "core-contrib"
    assert "opencv-contrib-python" in core["packages"]
    assert "opencv-python-headless" not in core["packages"]


def test_dependency_lock_rejects_a_core_environment_overwritten_by_another_lock(
    tmp_path,
):
    core_wheelhouse = tmp_path / "core-wheels"
    replacement_wheelhouse = tmp_path / "replacement-wheels"
    core_wheelhouse.mkdir()
    replacement_wheelhouse.mkdir()
    _, core_cv2 = _write_wheel(
        core_wheelhouse, "opencv-contrib-python", "core-contrib"
    )
    _, core_shared = _write_wheel(
        core_wheelhouse, "packaging", "core-shared",
        version="1.0.0", module="build_shared",
    )
    _, replacement_cv2 = _write_wheel(
        replacement_wheelhouse, "opencv-python-headless", "replacement-headless"
    )
    _, replacement_shared = _write_wheel(
        replacement_wheelhouse, "packaging", "replacement-shared",
        version="2.0.0", module="build_shared",
    )
    core_requirements = tmp_path / "requirements-core.txt"
    replacement_requirements = tmp_path / "requirements-replacement.txt"
    core_requirements.write_text(core_cv2 + core_shared, "utf-8")
    replacement_requirements.write_text(
        replacement_cv2 + replacement_shared, "utf-8"
    )
    shared_env = tmp_path / "shared-env"
    subprocess.run([sys.executable, "-m", "venv", str(shared_env)], check=True)
    python_relative = Path(
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    shared_python = shared_env / python_relative

    assert _pip_install(shared_python, core_wheelhouse, core_requirements).returncode == 0
    assert _pip_install(
        shared_python, replacement_wheelhouse, replacement_requirements
    ).returncode == 0

    lock_check = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "dependency_lock.py"),
            "--requirements-lock", str(core_requirements),
            "--wheelhouse", str(core_wheelhouse),
            "--python", str(shared_python),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert lock_check.returncode != 0
    assert "installed dependency has wrong version for packaging: 2.0.0" in (
        lock_check.stdout + lock_check.stderr
    )
