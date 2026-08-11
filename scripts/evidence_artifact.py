from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


DOCUMENTS = (
    "vision-artifact-attestation.json",
    "vision-provenance.json",
    "vision-release-descriptor.json",
    "vision-sbom.spdx.json",
)
UNSIGNED_MANIFEST = "trusted-unsigned-evidence-manifest.json"
SIGNED_MANIFEST = "trusted-signed-evidence-manifest.json"
FULL_DIGEST = re.compile(r"^[a-f0-9]{64}$")


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


def _payload_names(kind: str) -> tuple[str, ...]:
    if kind == "unsigned":
        return DOCUMENTS
    if kind == "signed":
        return (
            *DOCUMENTS,
            *(f"{name}.sig.json" for name in DOCUMENTS),
            UNSIGNED_MANIFEST,
        )
    raise AssertionError("unsupported evidence artifact kind")


def _manifest_name(kind: str) -> str:
    return UNSIGNED_MANIFEST if kind == "unsigned" else SIGNED_MANIFEST


def _regular_files(directory: Path) -> set[str]:
    if not directory.is_dir() or directory.is_symlink():
        raise AssertionError("evidence artifact directory is invalid")
    result: set[str] = set()
    for path in directory.iterdir():
        if not path.is_file() or path.is_symlink():
            raise AssertionError("evidence artifact contains non-regular member")
        result.add(path.name)
    return result


def seal(directory: Path, kind: str) -> str:
    payload_names = _payload_names(kind)
    if _regular_files(directory) != set(payload_names):
        raise AssertionError("evidence artifact payload set mismatch")
    manifest = {
        "schemaVersion": "vending-vision-trusted-evidence-artifact/v1",
        "kind": kind,
        "files": [
            {
                "path": name,
                "size": (directory / name).stat().st_size,
                "sha256": sha256_file(directory / name),
            }
            for name in sorted(payload_names)
        ],
    }
    raw = canonical_bytes(manifest)
    (directory / _manifest_name(kind)).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def verify(directory: Path, kind: str, expected_digest: str) -> None:
    if not FULL_DIGEST.fullmatch(expected_digest):
        raise AssertionError("expected evidence digest is invalid")
    payload_names = _payload_names(kind)
    manifest_name = _manifest_name(kind)
    if _regular_files(directory) != {*payload_names, manifest_name}:
        raise AssertionError("sealed evidence artifact member set mismatch")
    manifest_path = directory / manifest_name
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise AssertionError("sealed evidence manifest digest mismatch")
    try:
        manifest = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise AssertionError("sealed evidence manifest is invalid") from exc
    if canonical_bytes(manifest) != raw:
        raise AssertionError("sealed evidence manifest is not canonical")
    expected = {
        "schemaVersion": "vending-vision-trusted-evidence-artifact/v1",
        "kind": kind,
        "files": [
            {
                "path": name,
                "size": (directory / name).stat().st_size,
                "sha256": sha256_file(directory / name),
            }
            for name in sorted(payload_names)
        ],
    }
    if manifest != expected:
        raise AssertionError("sealed evidence payload binding mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("seal", "verify"))
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("unsigned", "signed"))
    parser.add_argument("--expected-digest")
    args = parser.parse_args()
    try:
        if args.operation == "seal":
            if args.expected_digest is not None:
                raise AssertionError("seal does not accept an expected digest")
            digest = seal(args.directory.resolve(), args.kind)
            print(canonical_bytes({"sha256": digest}).decode("utf-8"))
        else:
            if args.expected_digest is None:
                raise AssertionError("verify requires an expected digest")
            verify(args.directory.resolve(), args.kind, args.expected_digest)
            print("TRUSTED_EVIDENCE_ARTIFACT=PASS")
    except (AssertionError, OSError) as exc:
        print(f"TRUSTED_EVIDENCE_ARTIFACT=FAIL:{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
