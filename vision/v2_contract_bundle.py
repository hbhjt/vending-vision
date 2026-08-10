"""V2 boundary and runtime identity from the vendored generated bundle only."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


_BUNDLE_ROOT = Path(__file__).parents[1] / "contracts" / "vem_vision_v2"
_EXPECTED_FILES = {
    "__init__.py",
    "vision-v2.client.schema.json",
    "vision-v2.server.schema.json",
    "fixtures/client-valid.json",
    "fixtures/client-invalid.json",
    "fixtures/server-valid.json",
    "fixtures/server-invalid.json",
    "python/__init__.py",
    "python/vision_v2_models.py",
    "manifest.json",
}
_MANIFEST_METADATA = {
    "schemaVersion": "vem-vision-v2-contract-bundle/v1",
    "protocol": "vem.vision.v2",
    "bundleVersion": "1",
}
_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


class V2ContractBundleUnavailable(RuntimeError):
    """The optional Fast bundle is unavailable; core Vision keeps running."""


class _DuplicateManifestKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise _DuplicateManifestKey(key)
        value[key] = nested
    return value


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _canonical_bundle_path(relative_path: object, bundle_root: Path) -> Path | None:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        return None
    if _WINDOWS_DRIVE_PATTERN.match(relative_path) or relative_path.startswith("//"):
        return None
    pure_path = PurePosixPath(relative_path)
    if (
        pure_path.is_absolute()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or pure_path.as_posix() != relative_path
    ):
        return None
    root = bundle_root.resolve()
    path = (root / pure_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path


@dataclass(frozen=True)
class V2ContractIdentity:
    protocol: str
    schema_version: str
    bundle_version: str
    contract_digest: str


def load_v2_contract_identity() -> V2ContractIdentity:
    """Validate the canonical bundle before exposing any runtime identity."""
    try:
        manifest_path = _BUNDLE_ROOT / "manifest.json"
        raw_manifest = manifest_path.read_text("utf-8")
        manifest = json.loads(
            raw_manifest,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
        if not isinstance(manifest, dict) or set(manifest) != {
            * _MANIFEST_METADATA,
            "files",
            "bundleDigest",
        }:
            raise ValueError("manifest metadata keys")
        if any(
            type(manifest.get(key)) is not str or manifest[key] != expected
            for key, expected in _MANIFEST_METADATA.items()
        ):
            raise ValueError("manifest metadata")
        declared = manifest.get("files")
        if not isinstance(declared, dict) or set(declared) != _EXPECTED_FILES - {
            "manifest.json"
        }:
            raise ValueError("manifest files")
        if not isinstance(manifest.get("bundleDigest"), str) or not _DIGEST_PATTERN.fullmatch(
            manifest["bundleDigest"]
        ):
            raise ValueError("manifest bundle digest")
        if raw_manifest != _canonical_json(manifest) + "\n":
            raise ValueError("manifest canonical JSON")

        paths: dict[str, Path] = {}
        for relative_path, digest in declared.items():
            path = _canonical_bundle_path(relative_path, _BUNDLE_ROOT)
            if path is None or not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
                raise ValueError("manifest file path or digest")
            paths[relative_path] = path
        actual_files = {
            path.relative_to(_BUNDLE_ROOT).as_posix()
            for path in _BUNDLE_ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        if actual_files != _EXPECTED_FILES:
            raise ValueError("bundle file set")
        for relative_path, path in paths.items():
            if not path.is_file() or _sha256(path.read_bytes()) != declared[relative_path]:
                raise ValueError("bundle file digest")
        metadata = {**_MANIFEST_METADATA, "files": declared}
        if _sha256(_canonical_json(metadata)) != manifest["bundleDigest"]:
            raise ValueError("bundle digest")
    except (OSError, TypeError, ValueError, KeyError, _DuplicateManifestKey) as error:
        raise V2ContractBundleUnavailable("vision_v2_contract_bundle_unavailable") from error
    return V2ContractIdentity(
        protocol=manifest["protocol"],
        schema_version=manifest["schemaVersion"],
        bundle_version=manifest["bundleVersion"],
        contract_digest=manifest["bundleDigest"],
    )


def _parse_generated(direction: str, value: Any):
    """Parse an explicitly directed V2 boundary; never use a generic envelope."""
    try:
        from contracts.vem_vision_v2.python import vision_v2_models

        return getattr(vision_v2_models, f"parse_{direction}_message")(value)
    except (ImportError, OSError, ModuleNotFoundError) as error:
        raise V2ContractBundleUnavailable("vision_v2_contract_bundle_unavailable") from error
    except (ValueError, TypeError) as error:
        raise ValueError("invalid_v2_boundary_message") from error


def parse_v2_client_message(value: Any):
    return _parse_generated("client", value)


def parse_v2_server_message(value: Any):
    return _parse_generated("server", value)
