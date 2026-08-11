"""Immutable CatVTON model-pack validation; it never downloads anything.

The pack is deliberately outside the Vision archive.  A valid pack is a
canonical JSON allowlist whose entries name the upstream immutable revision,
relative path, byte count and digest.  Extra files are rejected so a mutable
Hugging Face snapshot cannot accidentally become a production dependency.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

OFFICIAL_CATVTON_REPOSITORY = "zhengchong/CatVTON"
OFFICIAL_CATVTON_REVISION = "9f415fa"
MANIFEST_NAME = "ai-model-manifest.json"
MANIFEST_SCHEMA_VERSION = "vem-catvton-model-pack/v1"


class AiModelPackError(RuntimeError):
    pass


@dataclass(frozen=True)
class AiModelPack:
    root: Path
    upstream_repository: str
    upstream_revision: str
    files: tuple[dict, ...]


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
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
        or pure.as_posix() != relative
    ):
        raise AiModelPackError("ai_model_manifest_path")
    return pure


def canonical_ai_model_manifest_json(manifest: dict) -> str:
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "upstream": {"repository": repository, "revision": revision},
        "files": files,
    }


def verify_ai_model_pack(root: str | Path | None) -> AiModelPack:
    if not root:
        raise AiModelPackError("ai_model_pack_missing")
    pack_root = Path(root).resolve()
    manifest_path = pack_root / MANIFEST_NAME
    try:
        manifest = json.loads(
            manifest_path.read_text("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except AiModelPackError:
        raise
    except (OSError, ValueError) as exc:
        raise AiModelPackError("ai_model_manifest_missing_or_invalid") from exc
    if set(manifest) != {"schemaVersion", "upstream", "files"} or manifest["schemaVersion"] != MANIFEST_SCHEMA_VERSION:
        raise AiModelPackError("ai_model_manifest_shape")
    upstream = manifest["upstream"]
    files = manifest["files"]
    if not isinstance(upstream, dict) or set(upstream) != {"repository", "revision"} or not all(isinstance(upstream[key], str) and upstream[key] for key in upstream):
        raise AiModelPackError("ai_model_manifest_upstream")
    if not isinstance(files, list) or not files:
        raise AiModelPackError("ai_model_manifest_files")
    seen: set[str] = set()
    expected = {MANIFEST_NAME}
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "byteSize", "sha256"}:
            raise AiModelPackError("ai_model_manifest_entry")
        relative = item["path"]
        pure = _validate_manifest_relative_path(relative)
        if relative in seen:
            raise AiModelPackError("ai_model_manifest_path")
        if (
            not isinstance(item["byteSize"], int)
            or item["byteSize"] <= 0
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or item["sha256"] != item["sha256"].lower()
            or any(character not in "0123456789abcdef" for character in item["sha256"])
        ):
            raise AiModelPackError("ai_model_manifest_integrity")
        path = (pack_root / pure).resolve()
        if pack_root not in path.parents or not path.is_file() or path.stat().st_size != item["byteSize"] or _digest(path) != item["sha256"]:
            raise AiModelPackError("ai_model_pack_digest")
        seen.add(relative)
        expected.add(relative)
    actual = {path.relative_to(pack_root).as_posix() for path in pack_root.rglob("*") if path.is_file()}
    if actual != expected:
        raise AiModelPackError("ai_model_pack_extra_or_missing")
    return AiModelPack(pack_root, upstream["repository"], upstream["revision"], tuple(files))


def official_ai_readiness(root: str | Path | None) -> bool:
    """A startup-only lightweight boundary probe: no inference/model load."""
    try:
        pack = verify_ai_model_pack(root)
        # Import only the worker module, deliberately not torch/diffusers.
        from vision import ai_attempt_worker  # noqa: F401
        return (
            pack.upstream_repository == OFFICIAL_CATVTON_REPOSITORY
            and pack.upstream_revision == OFFICIAL_CATVTON_REVISION
        )
    except (AiModelPackError, ImportError):
        return False
