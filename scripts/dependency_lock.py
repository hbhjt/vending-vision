"""Verify the one Python 3.11 release lock against selected wheels and installs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from packaging.markers import Marker, default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename


HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})$")


class DependencyLockError(ValueError):
    """The runtime dependency closure is incomplete or unverified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_lines(path: Path):
    pending = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            pending += line[:-1].strip() + " "
            continue
        if pending:
            line = pending + line
            pending = ""
        yield line
    if pending:
        raise DependencyLockError("requirements lock ends with an unfinished continuation")


def read_hash_locked_requirements(path: str | Path) -> dict[str, dict]:
    """Read a flat pip-compatible exact-pin lock with at least one hash each.

    Markers are retained in the returned inventory so a Windows lock can hold
    Windows-only transitive dependencies while Linux CI validates its own
    installable subset from the same immutable file.
    """
    entries: dict[str, dict] = {}
    for line in _logical_lines(Path(path)):
        if line.startswith("--index-url ") or line.startswith("--trusted-host "):
            continue
        requirement_text, separator, hash_text = line.partition(" --hash=")
        if not separator:
            raise DependencyLockError(f"lock entry must include SHA-256 hashes: {line}")
        try:
            requirement = Requirement(requirement_text.strip())
        except Exception as exc:
            raise DependencyLockError(f"lock entry must be an exact package pin: {line}") from exc
        specifiers = list(requirement.specifier)
        if requirement.url or requirement.extras or len(specifiers) != 1 or specifiers[0].operator != "==" or "*" in specifiers[0].version:
            raise DependencyLockError(f"lock entry must be an exact package pin: {line}")
        hashes = []
        for part in ("--hash=" + hash_text).split():
            hash_match = HASH.fullmatch(part)
            if hash_match is None:
                raise DependencyLockError(f"lock entry has unsupported option: {line}")
            hashes.append(hash_match.group(1))
        if not hashes:
            raise DependencyLockError(f"lock entry must include at least one SHA-256: {line}")
        name = requirement.name
        version = specifiers[0].version
        normalized = canonicalize_name(name)
        entry = {
            "name": name,
            "version": version,
            "hashes": sorted(set(hashes)),
            "marker": str(requirement.marker) if requirement.marker else None,
        }
        previous = entries.get(normalized)
        if previous and previous != entry:
            raise DependencyLockError(f"lock repeats {name} with a conflicting version or hash set")
        entries[normalized] = entry
    if not entries:
        raise DependencyLockError("requirements lock has no package entries")
    return entries


def active_hash_locked_requirements(
    lock: dict[str, dict], marker_environment: dict[str, str] | None = None
) -> dict[str, dict]:
    """Select precisely the lock entries that pip would use for one target."""
    environment = default_environment()
    if marker_environment:
        environment.update(marker_environment)
    result = {}
    for normalized, entry in lock.items():
        marker = entry.get("marker")
        if marker is None or Marker(marker).evaluate(environment):
            result[normalized] = entry
    return result


def selected_wheels(lock: dict[str, dict], wheelhouse: str | Path) -> dict[str, dict]:
    """Return the exact downloaded wheel for every lock package after digest checks."""
    wheelhouse = Path(wheelhouse)
    if not wheelhouse.is_dir():
        raise DependencyLockError(f"offline wheelhouse is missing: {wheelhouse}")
    found: dict[str, list[Path]] = {}
    for wheel in sorted(wheelhouse.glob("*.whl")):
        try:
            distribution, version, _, _ = parse_wheel_filename(wheel.name)
        except Exception as exc:
            raise DependencyLockError(f"wheel filename is invalid: {wheel.name}") from exc
        normalized = canonicalize_name(str(distribution))
        if normalized not in lock:
            raise DependencyLockError(f"offline wheelhouse contains unlocked wheel: {wheel.name}")
        if str(version) != lock[normalized]["version"]:
            raise DependencyLockError(f"offline wheel has wrong version for {lock[normalized]['name']}: {wheel.name}")
        found.setdefault(normalized, []).append(wheel)

    selected = {}
    for normalized, entry in lock.items():
        candidates = found.get(normalized, [])
        if len(candidates) != 1:
            raise DependencyLockError(
                f"offline wheelhouse must contain exactly one wheel for {entry['name']}, found {len(candidates)}"
            )
        wheel = candidates[0]
        digest = sha256_file(wheel)
        if digest not in entry["hashes"]:
            raise DependencyLockError(f"offline wheel digest is not locked for {entry['name']}: {wheel.name}")
        selected[normalized] = {"filename": wheel.name, "sha256": digest}
    return selected


def installed_distributions(python: str) -> dict[str, dict]:
    command = (
        "import importlib.metadata as m, json; "
        "print(json.dumps({d.metadata['Name'].lower().replace('_','-'):{"
        "'name':d.metadata['Name'],'version':d.version} "
        "for d in m.distributions() if d.metadata.get('Name')}, sort_keys=True))"
    )
    completed = subprocess.run([python, "-c", command], check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise DependencyLockError(f"could not inspect installed dependencies: {completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DependencyLockError("installed dependency inventory was not JSON") from exc


def verify_dependency_closure(
    lock_path: str | Path,
    wheelhouse: str | Path,
    python: str,
    marker_environment: dict[str, str] | None = None,
) -> list[dict]:
    lock = active_hash_locked_requirements(
        read_hash_locked_requirements(lock_path), marker_environment
    )
    wheels = selected_wheels(lock, wheelhouse)
    installed = installed_distributions(python)
    result = []
    for normalized, entry in sorted(lock.items()):
        distribution = installed.get(normalized)
        if distribution is None:
            raise DependencyLockError(f"locked dependency is not installed: {entry['name']}")
        if distribution["version"] != entry["version"]:
            raise DependencyLockError(
                f"installed dependency has wrong version for {entry['name']}: {distribution['version']}"
            )
        result.append({
            **entry,
            "wheel": wheels[normalized],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description="verify the fully locked Vision Python dependency closure")
    parser.add_argument("--requirements-lock", default="requirements.txt")
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--target-sys-platform", choices=("linux", "win32"))
    args = parser.parse_args()
    marker_environment = (
        {"sys_platform": args.target_sys_platform} if args.target_sys_platform else None
    )
    inventory = verify_dependency_closure(
        args.requirements_lock, args.wheelhouse, args.python, marker_environment
    )
    print(json.dumps({"dependencies": inventory}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
