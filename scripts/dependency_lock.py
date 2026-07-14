"""Verify the one Python 3.11 release lock against selected wheels and installs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from packaging.utils import canonicalize_name, parse_wheel_filename


HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})$")
PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s]+)$")


class DependencyLockError(ValueError):
    """The candidate dependency closure is incomplete or unverified."""


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
    """Read a flat pip-compatible exact-pin lock with at least one hash each."""
    entries: dict[str, dict] = {}
    for line in _logical_lines(Path(path)):
        if line.startswith("--index-url ") or line.startswith("--trusted-host "):
            continue
        parts = line.split()
        if not parts:
            continue
        match = PIN.fullmatch(parts[0])
        if not match:
            raise DependencyLockError(f"lock entry must be an exact package pin: {line}")
        hashes = []
        for part in parts[1:]:
            hash_match = HASH.fullmatch(part)
            if hash_match is None:
                raise DependencyLockError(f"lock entry has unsupported option: {line}")
            hashes.append(hash_match.group(1))
        if not hashes:
            raise DependencyLockError(f"lock entry must include at least one SHA-256: {line}")
        name, version = match.groups()
        normalized = canonicalize_name(name)
        entry = {"name": name, "version": version, "hashes": sorted(set(hashes))}
        previous = entries.get(normalized)
        if previous and previous != entry:
            raise DependencyLockError(f"lock repeats {name} with a conflicting version or hash set")
        entries[normalized] = entry
    if not entries:
        raise DependencyLockError("requirements lock has no package entries")
    return entries


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
        "'name':d.metadata['Name'],'version':d.version,"
        "'licenseExpression':d.metadata.get('License-Expression') or '',"
        "'license':d.metadata.get('License') or '',"
        "'classifiers':d.metadata.get_all('Classifier') or []} "
        "for d in m.distributions() if d.metadata.get('Name')}, sort_keys=True))"
    )
    completed = subprocess.run([python, "-c", command], check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise DependencyLockError(f"could not inspect installed dependencies: {completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DependencyLockError("installed dependency inventory was not JSON") from exc


# These are SPDX expressions for the exact locked package releases.  New lock
# entries must receive an explicit legal review instead of silently emitting
# NOASSERTION.  cv2-enumerate-cameras is intentionally GPLv3.
LICENSE_OVERRIDES = {
    "absl-py": "Apache-2.0",
    "altgraph": "MIT",
    "annotated-doc": "MIT",
    "annotated-types": "MIT",
    "anyio": "MIT",
    "attrs": "MIT",
    "certifi": "MPL-2.0",
    "cffi": "MIT",
    "click": "BSD-3-Clause",
    "contourpy": "BSD-3-Clause",
    "cryptography": "Apache-2.0 OR BSD-3-Clause",
    "cv2-enumerate-cameras": "GPL-3.0-or-later",
    "cycler": "BSD-3-Clause",
    "fastapi": "MIT",
    "flatbuffers": "Apache-2.0",
    "fonttools": "MIT",
    "h11": "MIT",
    "httpcore": "BSD-3-Clause",
    "httpx": "BSD-3-Clause",
    "idna": "BSD-3-Clause",
    "iniconfig": "MIT",
    "jax": "Apache-2.0",
    "jaxlib": "Apache-2.0",
    "jsonschema": "MIT",
    "jsonschema-specifications": "MIT",
    "kiwisolver": "BSD-3-Clause",
    "matplotlib": "PSF-2.0",
    "mediapipe": "Apache-2.0",
    "ml-dtypes": "Apache-2.0",
    "numpy": "BSD-3-Clause",
    "opencv-contrib-python": "Apache-2.0",
    "openvino": "Apache-2.0",
    "openvino-telemetry": "Apache-2.0",
    "opt-einsum": "MIT",
    "packaging": "Apache-2.0 OR BSD-2-Clause",
    "pillow": "HPND",
    "pip": "MIT",
    "pluggy": "MIT",
    "protobuf": "BSD-3-Clause",
    "pycparser": "BSD-3-Clause",
    "pydantic": "MIT",
    "pydantic-core": "MIT",
    "pygments": "BSD-2-Clause",
    "pyinstaller": "GPL-2.0-or-later WITH Bootloader-exception",
    "pyinstaller-hooks-contrib": "GPL-2.0-or-later",
    "pyparsing": "MIT",
    "pefile": "MIT",
    "pytest": "MIT",
    "pywin32-ctypes": "BSD-3-Clause",
    "python-dateutil": "Apache-2.0 OR BSD-3-Clause",
    "referencing": "MIT",
    "rpds-py": "MIT",
    "scipy": "BSD-3-Clause",
    "setuptools": "MIT",
    "six": "MIT",
    "sounddevice": "MIT",
    "starlette": "BSD-3-Clause",
    "typing-extensions": "PSF-2.0",
    "typing-inspection": "MIT",
    "uvicorn": "BSD-3-Clause",
    "websockets": "BSD-3-Clause",
    "wheel": "MIT",
}


def resolved_license(normalized_name: str, installed: dict) -> str:
    """Return reviewed SPDX, refusing an unknown package rather than NOASSERTION."""
    if normalized_name == "cv2-enumerate-cameras":
        metadata_license = installed.get("license", "")
        if "GNU GENERAL PUBLIC LICENSE" not in metadata_license.upper() or "VERSION 3" not in metadata_license.upper():
            raise DependencyLockError("cv2-enumerate-cameras metadata no longer declares GPLv3")
    value = LICENSE_OVERRIDES.get(normalized_name)
    if not value:
        raise DependencyLockError(f"installed dependency requires reviewed SPDX license mapping: {normalized_name}")
    return value


def verify_dependency_closure(lock_path: str | Path, wheelhouse: str | Path, python: str) -> list[dict]:
    lock = read_hash_locked_requirements(lock_path)
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
            "license": resolved_license(normalized, distribution),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description="verify the fully locked Vision Python dependency closure")
    parser.add_argument("--requirements-lock", default="requirements.txt")
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--python", default="python")
    args = parser.parse_args()
    inventory = verify_dependency_closure(args.requirements_lock, args.wheelhouse, args.python)
    print(json.dumps({"dependencies": inventory}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
