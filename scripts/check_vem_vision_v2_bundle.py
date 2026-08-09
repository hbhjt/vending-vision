#!/usr/bin/env python3
"""Verify this repository's vendored VEM Vision V2 contract bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_FILES = {
    "vision-v2.schema.json",
    "fixtures/valid.json",
    "fixtures/invalid.json",
    "python/vision_v2_models.py",
    "manifest.json",
}


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
        path = bundle_root / relative_path
        if not path.is_file():
            failures.append(f"missing bundle file: {relative_path}")
            continue
        if sha256(path.read_bytes()) != expected_digest:
            failures.append(f"digest mismatch: {relative_path}")

    bundle_digest = sha256(canonical_json(declared))
    if manifest.get("bundleDigest") != bundle_digest:
        failures.append("bundle digest mismatch")
    if manifest.get("protocol") != "vem.vision.v2":
        failures.append("unexpected protocol")
    if manifest.get("schemaVersion") != "vem-vision-v2-contract-bundle/v1":
        failures.append("unexpected schema version")
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
