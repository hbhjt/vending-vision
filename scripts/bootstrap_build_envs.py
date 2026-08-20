"""Create and populate the clean, offline release build environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


class BuildBootstrapError(RuntimeError):
    pass


INSTALL_POLL_SECONDS = 0.05
INSTALL_REAP_TIMEOUT_SECONDS = 30.0
INSTALL_TIMEOUT_SECONDS = 3600.0


def _venv_python(root: Path) -> Path:
    relative = Path("Scripts/python.exe") if sys.platform == "win32" else Path("bin/python")
    return root / relative


def _required_file(path: Path, diagnostic: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise BuildBootstrapError(diagnostic)
    return resolved


def _required_directory(path: Path, diagnostic: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise BuildBootstrapError(diagnostic)
    return resolved


def _offline_install_command(python: Path, wheelhouse: Path, requirements: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-index",
        "--find-links",
        str(wheelhouse),
        "--require-hashes",
        "-r",
        str(requirements),
    ]


def _terminate_and_reap(process: subprocess.Popen[str]) -> str | None:
    """Bound cleanup and return a stable diagnostic instead of leaking a child."""
    diagnostics = []
    try:
        process.terminate()
    except OSError as exc:
        diagnostics.append(f"terminate_{exc.__class__.__name__}")
    try:
        process.wait(timeout=INSTALL_REAP_TIMEOUT_SECONDS)
        return ";".join(diagnostics) or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
    except OSError as exc:
        diagnostics.append(f"kill_{exc.__class__.__name__}")
    try:
        process.wait(timeout=INSTALL_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        diagnostics.append("kill_reap_timeout")
    except OSError as exc:
        diagnostics.append(f"kill_reap_{exc.__class__.__name__}")
    return ";".join(diagnostics) or None


def _install_core_closure(
    python: Path, wheelhouse: Path, requirements: Path
) -> None:
    command = _offline_install_command(python, wheelhouse, requirements)
    try:
        process = subprocess.Popen(command)
    except OSError as exc:
        raise BuildBootstrapError(
            f"core_install_launch_failed:{exc.__class__.__name__}"
        ) from exc
    deadline = time.monotonic() + INSTALL_TIMEOUT_SECONDS
    while True:
        try:
            returncode = process.poll()
        except OSError as exc:
            cleanup = _terminate_and_reap(process)
            suffix = f";core_install_{cleanup}" if cleanup else ""
            raise BuildBootstrapError(
                f"core_install_poll_failed:{exc.__class__.__name__}{suffix}"
            ) from exc
        if returncode is not None:
            try:
                process.wait(timeout=INSTALL_REAP_TIMEOUT_SECONDS)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise BuildBootstrapError(
                    f"core_install_reap_failed:{exc.__class__.__name__}"
                ) from exc
            if returncode != 0:
                raise BuildBootstrapError(f"core_install_failed:exit_{returncode}")
            return
        if time.monotonic() >= deadline:
            cleanup = _terminate_and_reap(process)
            suffix = f";core_install_{cleanup}" if cleanup else ""
            raise BuildBootstrapError(f"core_install_deadline_exceeded{suffix}")
        time.sleep(INSTALL_POLL_SECONDS)


def bootstrap_build_envs(
    *,
    base_python: Path,
    core_env: Path,
    core_wheelhouse: Path,
    core_requirements: Path,
) -> dict[str, str]:
    base_python = _required_file(base_python, "base_python_missing")
    core_wheelhouse = _required_directory(core_wheelhouse, "core_wheelhouse_missing")
    core_requirements = _required_file(core_requirements, "core_requirements_missing")
    core_env = core_env.resolve()
    if core_env.exists():
        raise BuildBootstrapError("build_env_must_not_exist")

    try:
        subprocess.run(
            [str(base_python), "-m", "venv", str(core_env)], check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BuildBootstrapError(
            f"core_env_create_failed:{exc.__class__.__name__}"
        ) from exc

    core_python = _venv_python(core_env)
    _install_core_closure(core_python, core_wheelhouse, core_requirements)
    return {"corePython": str(core_python.absolute())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-python", required=True, type=Path)
    parser.add_argument("--core-env", required=True, type=Path)
    parser.add_argument("--core-wheelhouse", required=True, type=Path)
    parser.add_argument("--core-requirements", required=True, type=Path)
    args = parser.parse_args()
    result = bootstrap_build_envs(**vars(args))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
