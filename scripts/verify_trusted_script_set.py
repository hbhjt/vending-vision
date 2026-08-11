from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


EXPECTED_SCRIPTS = {
    "scripts/approve_candidate_source.py",
    "scripts/candidate_artifact_manifest.py",
    "scripts/evidence_artifact.py",
    "scripts/generate_trusted_candidate_evidence.py",
    "scripts/sign_candidate_evidence.py",
    "scripts/verify_trusted_candidate_inputs.py",
    "scripts/verify_release_tag_ruleset.py",
    "scripts/verify_trusted_script_set.py",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, descriptor: Path) -> None:
    raw = descriptor.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise AssertionError("trusted script descriptor is invalid") from exc
    if canonical_bytes(value) + b"\n" != raw:
        raise AssertionError("trusted script descriptor is not canonical")
    if set(value) != {"schemaVersion", "scripts"} or value["schemaVersion"] != "vending-vision-trusted-signer-scripts/v1":
        raise AssertionError("trusted script descriptor contract mismatch")
    scripts = value["scripts"]
    if not isinstance(scripts, list):
        raise AssertionError("trusted script descriptor scripts missing")
    by_path: dict[str, dict] = {}
    for item in scripts:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise AssertionError("trusted script descriptor entry shape mismatch")
        relative = item["path"]
        if (
            not isinstance(relative, str)
            or PurePosixPath(relative).as_posix() != relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or re.fullmatch(r"[a-f0-9]{64}", item["sha256"] or "") is None
            or relative in by_path
        ):
            raise AssertionError("trusted script descriptor entry invalid")
        by_path[relative] = item
    if set(by_path) != EXPECTED_SCRIPTS:
        raise AssertionError("trusted script descriptor allowlist mismatch")
    for relative, item in by_path.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != item["sha256"]:
            raise AssertionError(f"trusted script digest mismatch: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--descriptor", required=True, type=Path)
    args = parser.parse_args()
    try:
        verify(args.root.resolve(), args.descriptor.resolve())
    except (AssertionError, OSError) as exc:
        print(f"TRUSTED_SIGNER_SCRIPT_SET=FAIL:{exc}")
        return 1
    print("TRUSTED_SIGNER_SCRIPT_SET=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
