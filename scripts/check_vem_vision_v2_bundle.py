#!/usr/bin/env python3
"""Verify this repository's vendored VEM Vision V2 contract bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any


EXPECTED_FILES = {
    "__init__.py",
    "vision-v2.schema.json",
    "fixtures/valid.json",
    "fixtures/invalid.json",
    "python/__init__.py",
    "python/vision_v2_models.py",
    "manifest.json",
}
MANIFEST_METADATA = {
    "schemaVersion": "vem-vision-v2-contract-bundle/v1",
    "protocol": "vem.vision.v2",
    "bundleVersion": "1",
}
DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def check_bundle(bundle_root: Path) -> list[str]:
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("files")
    if not isinstance(declared, dict):
        return ["manifest files must be an object"]

    expected_payloads = EXPECTED_FILES - {"manifest.json"}
    failures: list[str] = []
    if set(manifest) != {*MANIFEST_METADATA, "files", "bundleDigest"}:
        failures.append("manifest metadata keys are not exact")
    for key, expected_value in MANIFEST_METADATA.items():
        if manifest.get(key) != expected_value:
            failures.append(f"unexpected manifest {key}")
    if not isinstance(manifest.get("bundleDigest"), str) or not DIGEST_PATTERN.fullmatch(
        manifest["bundleDigest"]
    ):
        failures.append("bundle digest is not a SHA-256 hex string")
    actual_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual_files != EXPECTED_FILES:
        failures.append("vendored bundle contains missing or unmanifested files")
    if set(declared) != expected_payloads:
        failures.append("manifest file set does not match the vendored bundle")

    for relative_path, expected_digest in declared.items():
        pure_path = PurePosixPath(relative_path)
        if (
            not isinstance(relative_path, str)
            or pure_path.is_absolute()
            or any(part in {"", ".", ".."} for part in pure_path.parts)
            or pure_path.as_posix() != relative_path
        ):
            failures.append(f"manifest path is not canonical: {relative_path!r}")
            continue
        if not isinstance(expected_digest, str) or not DIGEST_PATTERN.fullmatch(
            expected_digest
        ):
            failures.append(f"file digest is not a SHA-256 hex string: {relative_path}")
            continue
        path = bundle_root / relative_path
        if not path.is_file():
            failures.append(f"missing bundle file: {relative_path}")
            continue
        if sha256(path.read_bytes()) != expected_digest:
            failures.append(f"digest mismatch: {relative_path}")

    bundle_digest = sha256(canonical_json({**MANIFEST_METADATA, "files": declared}))
    if manifest.get("bundleDigest") != bundle_digest:
        failures.append("bundle digest mismatch")
    return failures


def main() -> int:
    bundle_root = Path(__file__).resolve().parents[1] / "contracts" / "vem_vision_v2"
    failures = check_bundle(bundle_root)
    if failures:
        print("VEM Vision V2 bundle check failed:")
        print("\n".join(failures))
        return 1
    print("VEM Vision V2 bundle is internally digest-consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
