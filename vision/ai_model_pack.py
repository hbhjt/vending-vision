"""Immutable CatVTON model-pack validation; it never downloads anything.

The pack is deliberately outside the Vision archive.  A valid pack is a
canonical JSON allowlist whose entries name the upstream immutable revision,
relative path, byte count and digest.  Extra files are rejected so a mutable
Hugging Face snapshot cannot accidentally become a production dependency.
"""
from __future__ import annotations

import json
import unicodedata
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from vision.ai_runtime_descriptor import digest_runtime_descriptor

OFFICIAL_CATVTON_REPOSITORY = "zhengchong/CatVTON"
OFFICIAL_CATVTON_REVISION = "2969fcf85fe62f2036605716f0b56f0b81d01d79"
OFFICIAL_CATVTON_SOURCE_REVISION = "3b795364a4d2f3b5adb365f39cdea376d20bc53c"
MANIFEST_NAME = "ai-model-manifest.json"
MANIFEST_SCHEMA_VERSION = "vem-official-ai-model-pack-descriptor/v2"
OFFICIAL_DESCRIPTOR_PATH = Path(__file__).resolve().parents[1] / "official-ai-model-pack-descriptor.json"
_READINESS_CACHE: dict[tuple[object, ...], bool] = {}


class AiModelPackError(RuntimeError):
    pass


@dataclass(frozen=True)
class AiModelPack:
    root: Path
    upstream_repository: str
    upstream_revision: str
    files: tuple[dict, ...]
    descriptor: dict


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _quick_file_identity(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _source_worker_identity() -> str:
    value = hashlib.sha256()
    for relative in (
        "ai_attempt_worker.py",
        "ai_attempt_process.py",
        "catvton_pose_masks.py",
        "catvton_preprocess.py",
        "vendor/catvton/PROVENANCE.md",
    ):
        path = Path(__file__).resolve().parent / relative
        if path.is_file():
            value.update(relative.encode("utf-8"))
            value.update(path.read_bytes())
    return value.hexdigest()


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict:
    seen: set[str] = set()
    result = {}
    for key, value in pairs:
        if key in seen:
            raise AiModelPackError("ai_model_manifest_duplicate_key")
        seen.add(key)
        result[key] = value
    return result


def _validate_manifest_relative_path(relative: object) -> PurePosixPath:
    pure = PurePosixPath(relative) if isinstance(relative, str) else None
    if (
        not pure
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or ":" in relative
        or unicodedata.normalize("NFC", relative) != relative
        or pure.as_posix() != relative
    ):
        raise AiModelPackError("ai_model_manifest_path")
    return pure


def canonical_ai_model_manifest_json(manifest: dict) -> str:
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_official_ai_model_pack_descriptor() -> dict:
    try:
        descriptor = json.loads(
            OFFICIAL_DESCRIPTOR_PATH.read_text("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except AiModelPackError:
        raise
    except (OSError, ValueError) as exc:
        raise AiModelPackError("ai_model_descriptor_missing_or_invalid") from exc
    _validate_descriptor_shape(descriptor)
    if canonical_ai_model_manifest_json(descriptor) != OFFICIAL_DESCRIPTOR_PATH.read_text("utf-8"):
        raise AiModelPackError("ai_model_descriptor_noncanonical")
    return descriptor


def create_ai_model_manifest(
    root: str | Path,
    *,
    repository: str,
    revision: str,
    paths: list[str],
) -> dict:
    pack_root = Path(root).resolve()
    seen: set[str] = set()
    files = []
    for relative in sorted(paths):
        pure = _validate_manifest_relative_path(relative)
        if pure.as_posix() in seen:
            raise AiModelPackError("ai_model_manifest_path")
        path = (pack_root / pure).resolve()
        if pack_root not in path.parents or not path.is_file():
            raise AiModelPackError("ai_model_pack_digest")
        seen.add(pure.as_posix())
        files.append(
            {
                "path": pure.as_posix(),
                "byteSize": path.stat().st_size,
                "sha256": _digest(path),
            }
        )
    if not files:
        raise AiModelPackError("ai_model_manifest_files")
    return {
        "schemaVersion": "vem-catvton-model-pack/v1",
        "upstream": {"repository": repository, "revision": revision},
        "files": files,
    }


def _validate_descriptor_shape(descriptor: dict) -> None:
    if set(descriptor) != {"schemaVersion", "catvtonSourceRevision", "totalByteSize", "upstreams", "files"}:
        raise AiModelPackError("ai_model_descriptor_shape")
    if descriptor["schemaVersion"] != MANIFEST_SCHEMA_VERSION:
        raise AiModelPackError("ai_model_descriptor_shape")
    if not isinstance(descriptor["catvtonSourceRevision"], str) or not descriptor["catvtonSourceRevision"]:
        raise AiModelPackError("ai_model_descriptor_source_revision")
    upstreams = descriptor["upstreams"]
    files = descriptor["files"]
    if not isinstance(upstreams, list) or not upstreams:
        raise AiModelPackError("ai_model_descriptor_upstreams")
    upstream_ids: set[str] = set()
    for upstream in upstreams:
        if not isinstance(upstream, dict) or set(upstream) != {"id", "repository", "revision"}:
            raise AiModelPackError("ai_model_descriptor_upstreams")
        if not all(isinstance(upstream[key], str) and upstream[key] for key in upstream):
            raise AiModelPackError("ai_model_descriptor_upstreams")
        if upstream["id"] in upstream_ids:
            raise AiModelPackError("ai_model_descriptor_upstreams")
        upstream_ids.add(upstream["id"])
    if not isinstance(files, list) or not files:
        raise AiModelPackError("ai_model_descriptor_files")
    seen_paths: set[str] = set()
    seen_casefold: set[str] = set()
    seen_roles: set[str] = set()
    total = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "upstreamPath", "upstream", "role", "format", "byteSize", "sha256"}:
            raise AiModelPackError("ai_model_descriptor_entry")
        path = item["path"]
        upstream_path = item["upstreamPath"]
        _validate_manifest_relative_path(path)
        _validate_manifest_relative_path(upstream_path)
        collision_key = unicodedata.normalize("NFC", path).casefold()
        if path in seen_paths or collision_key in seen_casefold:
            raise AiModelPackError("ai_model_manifest_path")
        if item["upstream"] not in upstream_ids or item["role"] in seen_roles:
            raise AiModelPackError("ai_model_descriptor_entry")
        if (
            not isinstance(item["byteSize"], int)
            or item["byteSize"] <= 0
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or item["sha256"] != item["sha256"].lower()
            or any(character not in "0123456789abcdef" for character in item["sha256"])
            or not isinstance(item["format"], str)
            or not item["format"]
            or not isinstance(item["role"], str)
            or not item["role"]
        ):
            raise AiModelPackError("ai_model_manifest_integrity")
        seen_paths.add(path)
        seen_casefold.add(collision_key)
        seen_roles.add(item["role"])
        total += item["byteSize"]
    if files != sorted(files, key=lambda item: item["path"]):
        raise AiModelPackError("ai_model_descriptor_noncanonical")
    if descriptor["totalByteSize"] != total:
        raise AiModelPackError("ai_model_descriptor_total")


def verify_ai_model_pack(root: str | Path | None, *, descriptor: dict | None = None) -> AiModelPack:
    if not root:
        raise AiModelPackError("ai_model_pack_missing")
    expected_descriptor = descriptor or load_official_ai_model_pack_descriptor()
    _validate_descriptor_shape(expected_descriptor)
    expected_manifest_bytes = canonical_ai_model_manifest_json(expected_descriptor)
    pack_root = Path(root).resolve()
    manifest_path = pack_root / MANIFEST_NAME
    try:
        manifest_bytes = manifest_path.read_text("utf-8")
        manifest = json.loads(
            manifest_bytes,
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except AiModelPackError:
        raise
    except (OSError, ValueError) as exc:
        raise AiModelPackError("ai_model_manifest_missing_or_invalid") from exc
    if manifest != expected_descriptor or manifest_bytes != expected_manifest_bytes:
        raise AiModelPackError("ai_model_manifest_descriptor_mismatch")
    files = manifest["files"]
    seen: set[str] = set()
    expected = {MANIFEST_NAME}
    allowed_dirs = {PurePosixPath(".")}
    for item in files:
        relative = item["path"]
        pure = _validate_manifest_relative_path(relative)
        if relative in seen:
            raise AiModelPackError("ai_model_manifest_path")
        path = (pack_root / pure).resolve()
        if (
            pack_root not in path.parents
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != item["byteSize"]
            or _digest(path) != item["sha256"]
        ):
            raise AiModelPackError("ai_model_pack_digest")
        seen.add(relative)
        expected.add(relative)
        parent = pure.parent
        while parent != PurePosixPath("."):
            allowed_dirs.add(parent)
            parent = parent.parent
    actual_files = set()
    for path in pack_root.rglob("*"):
        relative = PurePosixPath(path.relative_to(pack_root).as_posix())
        if path.is_symlink():
            raise AiModelPackError("ai_model_pack_symlink")
        if path.is_dir():
            if relative not in allowed_dirs:
                raise AiModelPackError("ai_model_pack_extra_or_missing")
            continue
        if path.is_file():
            actual_files.add(relative.as_posix())
            continue
        raise AiModelPackError("ai_model_pack_file_type")
    if actual_files != expected:
        raise AiModelPackError("ai_model_pack_extra_or_missing")
    upstreams = {item["id"]: item for item in manifest["upstreams"]}
    primary = upstreams.get("catvton") or manifest["upstreams"][0]
    return AiModelPack(pack_root, primary["repository"], primary["revision"], tuple(files), manifest)


def official_ai_readiness(root: str | Path | None) -> bool:
    """Startup-only worker probe: no inference/model load and cached per pack."""
    if not root:
        return False
    pack_root = Path(root).resolve()
    try:
        manifest_identity = _quick_file_identity(pack_root / MANIFEST_NAME)
        descriptor_identity = _quick_file_identity(OFFICIAL_DESCRIPTOR_PATH)
        worker_identity = _source_worker_identity()
        runtime_descriptor_digest = digest_runtime_descriptor()
    except OSError:
        return False
    cache_key = (
        str(pack_root),
        manifest_identity,
        descriptor_identity,
        worker_identity,
        runtime_descriptor_digest,
    )
    cached = _READINESS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        verify_ai_model_pack(pack_root)
        from vision.ai_attempt_process import probe_ai_attempt_worker

        probe_ai_attempt_worker(pack_root)
        _READINESS_CACHE.clear()
        _READINESS_CACHE[cache_key] = True
        return True
    except (AiModelPackError, ImportError, RuntimeError, OSError):
        _READINESS_CACHE.clear()
        _READINESS_CACHE[cache_key] = False
        return False
