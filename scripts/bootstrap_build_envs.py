"""Create and populate the two clean, offline release build environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


class BuildBootstrapError(RuntimeError):
    pass


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
        (core_python, core_wheelhouse, core_requirements),
        (ai_python, ai_wheelhouse, ai_requirements),
    )
    for python, wheelhouse, requirements in installs:
        subprocess.run(
            [
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
            ],
            check=True,
        )
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
