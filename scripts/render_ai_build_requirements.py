"""Render the hash-locked PyInstaller-only subset for the isolated AI builder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import tempfile


AI_BUILD_TOOLS = {
    "altgraph",
    "pefile",
    "pyinstaller",
    "pyinstaller-hooks-contrib",
    "pywin32-ctypes",
    "setuptools",
}
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s]+)(?:\s+(.+))?$")
_HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}")


class AiBuildRequirementsError(RuntimeError):
    pass


def _logical_lines(path: Path):
    pending = ""
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            pending += line[:-1].strip() + " "
            continue
        if pending:
            line = pending + line
            pending = ""
        if line and not line.startswith("#"):
            yield " ".join(line.split())
    if pending:
        raise AiBuildRequirementsError("core_requirements_unfinished_continuation")


def render_ai_build_requirements(core_requirements: Path, output: Path) -> None:
    selected: dict[str, str] = {}
    for line in _logical_lines(core_requirements):
        match = _PIN.fullmatch(line)
        if match is None:
            raise AiBuildRequirementsError("core_requirements_entry")
        name = match.group(1).lower().replace("_", "-")
        if name not in AI_BUILD_TOOLS:
            continue
        if name in selected or not _HASH.search(line):
            raise AiBuildRequirementsError("ai_build_tool_lock")
        options = match.group(3) or ""
        if " ".join(_HASH.findall(options)) != options:
            raise AiBuildRequirementsError("ai_build_tool_lock_option")
        selected[name] = line
    if set(selected) != AI_BUILD_TOOLS:
        raise AiBuildRequirementsError("ai_build_tool_lock_incomplete")
    value = "\n".join(selected[name] for name in sorted(selected)) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}-", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-requirements", default="requirements.txt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render_ai_build_requirements(args.core_requirements, args.output)
    print("AI build-tool requirements rendered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
