"""Canonical runtime build identity shared by config and delivery verification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


BUILD_IDENTITY_SCHEMA = "vending-vision-build-identity/v1"
_APP_VERSION_RE = re.compile(r'^APP_VERSION = "([0-9A-Za-z.-]+)"\n$')
_SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class _DuplicateIdentityKey(ValueError):
    pass


@dataclass(frozen=True)
class BuildIdentity:
    app_version: str
    source_commit: str | None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise _DuplicateIdentityKey(key)
        value[key] = nested
    return value


def _canonical_json(value: dict[str, str]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _read_app_version(runtime_root: Path) -> str:
    try:
        raw = (runtime_root / "_build_version.py").read_text("utf-8")
    except OSError as error:
        raise RuntimeError("build version marker is unavailable") from error
    match = _APP_VERSION_RE.fullmatch(raw)
    if match is None:
        raise RuntimeError("build version marker is invalid")
    return match.group(1)


def load_build_identity(
    runtime_root: Path | None = None, *, require_source_commit: bool = False
) -> BuildIdentity:
    """Read the single-line version and optional packaged commit identity as data."""
    root = Path(__file__).resolve().parent if runtime_root is None else Path(runtime_root)
    app_version = _read_app_version(root)
    identity_path = root / "_build_identity.json"
    if not identity_path.is_file():
        if require_source_commit:
            raise RuntimeError("packaged build identity is unavailable")
        return BuildIdentity(app_version=app_version, source_commit=None)
    try:
        raw = identity_path.read_text("utf-8")
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, ValueError, _DuplicateIdentityKey) as error:
        raise RuntimeError("packaged build identity is invalid") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"appVersion", "schemaVersion", "sourceCommit"}
        or payload.get("schemaVersion") != BUILD_IDENTITY_SCHEMA
        or payload.get("appVersion") != app_version
        or not isinstance(payload.get("sourceCommit"), str)
        or _SOURCE_COMMIT_RE.fullmatch(payload["sourceCommit"]) is None
        or raw != _canonical_json(payload) + "\n"
    ):
        raise RuntimeError("packaged build identity is invalid")
    return BuildIdentity(app_version=app_version, source_commit=payload["sourceCommit"])


def write_packaged_build_identity(
    version_marker: Path, identity_output: Path, source_commit: str
) -> None:
    """Materialize commit identity beside an unchanged, single-line version marker."""
    app_version = _read_app_version(Path(version_marker).parent)
    if _SOURCE_COMMIT_RE.fullmatch(source_commit) is None:
        raise RuntimeError("packaged build source commit is invalid")
    payload = {
        "appVersion": app_version,
        "schemaVersion": BUILD_IDENTITY_SCHEMA,
        "sourceCommit": source_commit,
    }
    output = Path(identity_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical_json(payload) + "\n", encoding="utf-8", newline="\n")
