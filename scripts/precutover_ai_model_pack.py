"""Bounded pre-cutover proof for an externally supplied official model pack."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.ai_model_pack_release import (
    descriptor_sha256,
    install_model_pack_zip,
)
from vision.ai_model_pack import canonical_ai_model_manifest_json


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate descriptor key")
        result[key] = value
    return result


def _canonical_regular(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label}_regular_file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise RuntimeError(f"{label}_canonical_path")
    return path


def verify_and_install_model_pack(
    *,
    archive: Path,
    descriptor_path: Path,
    expected_archive_byte_size: int,
    expected_archive_sha256: str,
    expected_descriptor_sha256: str,
    install_root: Path,
) -> dict[str, object]:
    archive = _canonical_regular(archive, "model_pack_archive")
    descriptor_path = _canonical_regular(descriptor_path, "model_pack_descriptor")
    if (
        not isinstance(expected_archive_byte_size, int)
        or expected_archive_byte_size <= 0
        or _SHA256_RE.fullmatch(expected_archive_sha256) is None
        or _SHA256_RE.fullmatch(expected_descriptor_sha256) is None
        or not install_root.is_absolute()
        or install_root.exists()
    ):
        raise RuntimeError("model_pack_external_identity")
    descriptor_bytes = descriptor_path.read_bytes()
    try:
        descriptor = json.loads(
            descriptor_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("model_pack_descriptor_json") from exc
    canonical = canonical_ai_model_manifest_json(descriptor).encode("utf-8")
    if canonical != descriptor_bytes:
        raise RuntimeError("model_pack_descriptor_noncanonical")
    if (
        descriptor_sha256(descriptor) != expected_descriptor_sha256
        or archive.stat().st_size != expected_archive_byte_size
    ):
        raise RuntimeError("model_pack_external_identity")
    installed = install_model_pack_zip(
        archive,
        install_root,
        descriptor,
        outer_sha256=expected_archive_sha256,
    )
    return {
        "archive": {
            "byteSize": expected_archive_byte_size,
            "sha256": expected_archive_sha256,
        },
        "descriptor": {
            "catvtonSourceRevision": descriptor["catvtonSourceRevision"],
            "schemaVersion": descriptor["schemaVersion"],
            "sha256": expected_descriptor_sha256,
            "totalByteSize": descriptor["totalByteSize"],
            "upstreams": descriptor["upstreams"],
        },
        "installedPack": str(installed),
        "schemaVersion": "vending-vision-precutover-model-pack-proof/v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--expected-archive-byte-size", type=int, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-descriptor-sha256", required=True)
    parser.add_argument("--install-root", type=Path, required=True)
    args = parser.parse_args()
    report = verify_and_install_model_pack(
        archive=args.archive,
        descriptor_path=args.descriptor,
        expected_archive_byte_size=args.expected_archive_byte_size,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_descriptor_sha256=args.expected_descriptor_sha256,
        install_root=args.install_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
