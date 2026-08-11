from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).parents[1]


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
