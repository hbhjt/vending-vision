"""Build, verify, and safely install AI model packs from a fixed descriptor."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

from vision.ai_model_pack import (
    MANIFEST_NAME,
    AiModelPackError,
    canonical_ai_model_manifest_json,
    verify_ai_model_pack,
)

FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
REGULAR_FILE_MODE = 0o100644 << 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor_sha256(descriptor: dict) -> str:
    return hashlib.sha256(canonical_ai_model_manifest_json(descriptor).encode("utf-8")).hexdigest()


def _safe_zip_name(name: str, seen: set[str]) -> PurePosixPath:
    if "\\" in name or ":" in name or unicodedata.normalize("NFC", name) != name:
        raise AiModelPackError("ai_model_zip_path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise AiModelPackError("ai_model_zip_path")
    collision = unicodedata.normalize("NFC", name).casefold()
    if collision in seen:
        raise AiModelPackError("ai_model_zip_duplicate")
    seen.add(collision)
    return path


def build_model_pack_zip(source_root: Path, output_zip: Path, descriptor: dict) -> str:
    manifest_bytes = canonical_ai_model_manifest_json(descriptor).encode("utf-8")
    entries = [(MANIFEST_NAME, manifest_bytes)]
    for item in descriptor["files"]:
        source = source_root / item["path"]
        if source.is_symlink() or not source.is_file():
            raise AiModelPackError("ai_model_pack_digest")
        if source.stat().st_size != item["byteSize"] or _sha256(source) != item["sha256"]:
            raise AiModelPackError("ai_model_pack_digest")
        entries.append((item["path"], source.read_bytes()))
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.comment = b""
        for name, data in sorted(entries, key=lambda entry: entry[0]):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = REGULAR_FILE_MODE
            info.extra = b""
            info.comment = b""
            archive.writestr(info, data)
    return _sha256(output_zip)


def verify_model_pack_zip(zip_path: Path, descriptor: dict, *, outer_sha256: str | None = None) -> str:
    digest = _sha256(zip_path)
    if outer_sha256 is not None and digest != outer_sha256:
        raise AiModelPackError("ai_model_zip_outer_digest")
    expected = {MANIFEST_NAME, *(item["path"] for item in descriptor["files"])}
    seen: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        if archive.comment:
            raise AiModelPackError("ai_model_zip_metadata")
        names = []
        for info in archive.infolist():
            path = _safe_zip_name(info.filename, seen)
            mode = (info.external_attr >> 16) & 0o170000
            if info.is_dir() or mode == 0o120000 or info.compress_type != zipfile.ZIP_STORED or info.extra or info.comment:
                raise AiModelPackError("ai_model_zip_metadata")
            names.append(path.as_posix())
        if set(names) != expected:
            raise AiModelPackError("ai_model_zip_entries")
        manifest = archive.read(MANIFEST_NAME).decode("utf-8")
        if manifest != canonical_ai_model_manifest_json(descriptor):
            raise AiModelPackError("ai_model_manifest_descriptor_mismatch")
    return digest


def build_model_pack_release_manifest(zip_path: Path, descriptor: dict, *, outer_sha256: str | None = None) -> dict:
    archive_sha = verify_model_pack_zip(zip_path, descriptor, outer_sha256=outer_sha256)
    return {
        "schemaVersion": "vem-ai-model-pack-release/v1",
        "archive": {
            "path": zip_path.name,
            "sha256": archive_sha,
            "byteSize": zip_path.stat().st_size,
        },
        "descriptor": {
            "schemaVersion": descriptor["schemaVersion"],
            "sha256": descriptor_sha256(descriptor),
            "totalByteSize": descriptor["totalByteSize"],
        },
    }


def install_model_pack_zip(zip_path: Path, install_root: Path, descriptor: dict, *, outer_sha256: str | None = None) -> Path:
    digest = verify_model_pack_zip(zip_path, descriptor, outer_sha256=outer_sha256)
    install_root.mkdir(parents=True, exist_ok=True)
    packs_root = install_root / "packs"
    packs_root.mkdir(exist_ok=True)
    target = packs_root / digest
    active_file = install_root / "active-pack.json"
    if target.exists():
        verify_ai_model_pack(target, descriptor=descriptor)
    else:
        staging = packs_root / f".staging-{digest}-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for info in archive.infolist():
                    destination = staging / info.filename
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
            verify_ai_model_pack(staging, descriptor=descriptor)
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    temp_active = active_file.with_suffix(".tmp")
    temp_active.write_text(
        json.dumps(
            {
                "schemaVersion": "vem-ai-model-pack-selection/v1",
                "archiveSha256": digest,
                "descriptorSha256": descriptor_sha256(descriptor),
                "installDigest": digest,
                "path": str(target),
            },
            sort_keys=True,
        ),
        "utf-8",
    )
    os.replace(temp_active, active_file)
    return target
