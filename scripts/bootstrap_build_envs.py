"""Create and populate the two clean, offline release build environments."""

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


def _reap_install_process(name: str, process: subprocess.Popen[str]) -> str | None:
    """Terminate and reap an unfinished sibling without masking its failure."""
    diagnostics: list[str] = []
    try:
        if process.poll() is not None:
            return None
    except OSError as exc:
        diagnostics.append(f"poll_{exc.__class__.__name__}")
    try:
        process.terminate()
        process.wait(timeout=INSTALL_REAP_TIMEOUT_SECONDS)
        return ";".join(f"{name}_install_{item}" for item in diagnostics) or None
    except OSError as exc:
        diagnostics.append(f"terminate_{exc.__class__.__name__}")
    except subprocess.TimeoutExpired:
        pass

    try:
        if process.poll() is not None:
            return ";".join(f"{name}_install_{item}" for item in diagnostics) or None
    except OSError as exc:
        diagnostics.append(f"poll_{exc.__class__.__name__}")

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
    return ";".join(f"{name}_install_{item}" for item in diagnostics) or None


def _cleanup_install_processes(
    processes: list[tuple[str, subprocess.Popen[str]]], *, exclude: str | None = None
) -> list[str]:
    return [
        diagnostic
        for name, process in processes
        if name != exclude
        and (diagnostic := _reap_install_process(name, process)) is not None
    ]


def _install_isolated_closures(
    *installs: tuple[str, Path, Path, Path],
) -> None:
    """Run the independently locked offline closures concurrently and fail closed."""
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    deadline = time.monotonic() + INSTALL_TIMEOUT_SECONDS
    try:
        for name, python, wheelhouse, requirements in installs:
            processes.append(
                (name, subprocess.Popen(_offline_install_command(python, wheelhouse, requirements)))
            )
    except OSError as exc:
        cleanup = _cleanup_install_processes(processes)
        suffix = f";{';'.join(cleanup)}" if cleanup else ""
        raise BuildBootstrapError(
            f"build_env_install_launch_failed:{exc.__class__.__name__}{suffix}"
        ) from exc

    pending = dict(processes)
    while pending:
        for name, process in tuple(pending.items()):
            try:
                returncode = process.poll()
            except OSError as exc:
                cleanup = _cleanup_install_processes(processes)
                suffix = f";{';'.join(cleanup)}" if cleanup else ""
                raise BuildBootstrapError(
                    f"{name}_install_poll_failed:{exc.__class__.__name__}{suffix}"
                ) from exc
            if returncode is None:
                continue
            del pending[name]
            if returncode == 0:
                continue
            cleanup = _cleanup_install_processes(processes, exclude=name)
            suffix = f";{';'.join(cleanup)}" if cleanup else ""
            raise BuildBootstrapError(f"{name}_install_failed:exit_{returncode}{suffix}")
        if pending:
            if time.monotonic() >= deadline:
                cleanup = _cleanup_install_processes(processes)
                suffix = f";{';'.join(cleanup)}" if cleanup else ""
                raise BuildBootstrapError(f"build_env_install_deadline_exceeded{suffix}")
            time.sleep(INSTALL_POLL_SECONDS)


def bootstrap_build_envs(
    *,
    base_python: Path,
    core_env: Path,
    core_wheelhouse: Path,
    core_requirements: Path,
    ai_env: Path,
    ai_wheelhouse: Path,
    ai_requirements: Path,
) -> dict[str, str]:
    base_python = _required_file(base_python, "base_python_missing")
    core_wheelhouse = _required_directory(core_wheelhouse, "core_wheelhouse_missing")
    ai_wheelhouse = _required_directory(ai_wheelhouse, "ai_wheelhouse_missing")
    core_requirements = _required_file(core_requirements, "core_requirements_missing")
    ai_requirements = _required_file(ai_requirements, "ai_requirements_missing")
    core_env = core_env.resolve()
    ai_env = ai_env.resolve()
    if core_env == ai_env:
        raise BuildBootstrapError("build_envs_must_be_distinct")
    if core_env.exists() or ai_env.exists():
        raise BuildBootstrapError("build_env_must_not_exist")

    for environment in (core_env, ai_env):
        subprocess.run([str(base_python), "-m", "venv", str(environment)], check=True)

    core_python = _venv_python(core_env)
    ai_python = _venv_python(ai_env)
    installs = (
        ("core", core_python, core_wheelhouse, core_requirements),
        ("ai", ai_python, ai_wheelhouse, ai_requirements),
    )
    _install_isolated_closures(*installs)
    return {"corePython": str(core_python.resolve()), "aiPython": str(ai_python.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-python", required=True, type=Path)
    parser.add_argument("--core-env", required=True, type=Path)
    parser.add_argument("--core-wheelhouse", required=True, type=Path)
    parser.add_argument("--core-requirements", required=True, type=Path)
    parser.add_argument("--ai-env", required=True, type=Path)
    parser.add_argument("--ai-wheelhouse", required=True, type=Path)
    parser.add_argument("--ai-requirements", required=True, type=Path)
    args = parser.parse_args()
    result = bootstrap_build_envs(**vars(args))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
