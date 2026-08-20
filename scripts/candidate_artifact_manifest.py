from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.build_identity import load_build_identity  # noqa: E402
from scripts.hard_cutover_policy import is_retired_runtime_dependency  # noqa: E402


SCHEMA = "vending-vision-candidate-artifact/v3"
EMBEDDED_MANIFEST = "candidate-manifest.json"
LAYOUT = {
    "mainOnedir": "vending-vision",
    "mainExecutable": "vending-vision/vending-vision.exe",
}
BINDING_PATHS = {
    "mainExecutable": LAYOUT["mainExecutable"],
}
_MAX_ARCHIVE_FILES = 100_000
_MAX_NESTED_ARCHIVE_DEPTH = 3
_MAX_NESTED_ARCHIVE_FILES = 10_000
_MAX_NESTED_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_ENTRY_BYTES = 128 * 1024 * 1024
_MAX_ARCHIVE_COMPRESSION_RATIO = 200
_ARCHIVE_SUFFIXES = {".pyz", ".whl", ".zip"}


def retired_packaged_entries(entries, *, include_historical_generic_modules=True):
    """Return normalized archive/resource entries owned by the retired paths."""
    return sorted(
        entry
        for entry in entries
        if _entry_is_retired(
            entry,
            include_historical_generic_modules=include_historical_generic_modules,
        )
    )


def _contains_tokens(tokens: list[str], expected: tuple[str, ...]) -> bool:
    width = len(expected)
    return any(tuple(tokens[index : index + width]) == expected for index in range(len(tokens) - width + 1))


def _entry_is_retired(entry: str, *, include_historical_generic_modules: bool) -> bool:
    if any(
        is_retired_runtime_dependency(token)
        for token in re.findall(r"[A-Za-z0-9_.-]+", entry)
    ):
        return True
    tokens = [
        token
        for token in re.split(r"[\\/._:\-]+", entry.casefold())
        if token
    ]
    sequences = (
        ("try", "on", "session"),
        ("try", "on", "frontend"),
        ("try", "on", "ai"),
        ("try", "on", "fast"),
        ("profile", "fast", "try", "on"),
        ("vem", "vision", "v1"),
        ("ai", "acceptance", "evidence"),
        ("ai", "attempt"),
        ("ai", "model"),
        ("ai", "process", "tree", "worker"),
        ("ai", "runtime"),
        ("ai", "source", "provenance"),
        ("ai", "wheelhouse"),
        ("ai", "worker"),
        ("".join(("cat", "vton")),),
        ("regional", "evaluator"),
        ("official", "ai"),
        ("requirements", "ai"),
        ("venv", "packaging", "ai"),
        ("materialize", "ai", "wheelhouse"),
        ("verify", "ai", "wheelhouse"),
        ("render", "ai", "build", "requirements"),
        ("fast", "attempt"),
        ("fast", "result"),
        ("fast", "adjustment"),
        ("vending", "vision", "ai", "worker"),
        ("safetensors",),
    )
    if any(_contains_tokens(tokens, sequence) for sequence in sequences):
        return True
    if include_historical_generic_modules:
        return _contains_tokens(tokens, ("vision", "process", "supervisor")) or _contains_tokens(
            tokens, ("vision", "source", "provenance")
        )
    return False


def audit_packaged_model_files(payload: list[tuple[str, Path]]) -> None:
    """Bind every packaged model weight to the packaged production manifest."""
    prefix = f'{LAYOUT["mainOnedir"]}/_internal/models/'
    by_relative = {relative: path for relative, path in payload}
    model_members = {
        relative: path
        for relative, path in by_relative.items()
        if relative.casefold().startswith(prefix.casefold())
    }
    if not model_members:
        raise RuntimeError("candidate_model_manifest")
    manifest_relative = prefix + "model-manifest.json"
    manifest_path = by_relative.get(manifest_relative)
    if manifest_path is None:
        raise RuntimeError("candidate_model_manifest")
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
        models = manifest["models"]
        if not isinstance(models, list) or not all(
            isinstance(item, dict)
            and set(item) == {"path", "role", "sha256"}
            and isinstance(item.get("path"), str)
            and isinstance(item.get("role"), str)
            and isinstance(item.get("sha256"), str)
            for item in models
        ):
            raise ValueError("model entries")
        declared = {
            f'{LAYOUT["mainOnedir"]}/_internal/{item["path"]}': item["sha256"]
            for item in models
        }
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise RuntimeError("candidate_model_manifest") from error
    if (
        manifest.get("schemaVersion") != "vending-vision-model-manifest/v1"
        or len(declared) != len(models)
        or not all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in declared.values())
    ):
        raise RuntimeError("candidate_model_manifest")
    model_artifact_suffixes = {
        ".bin",
        ".caffemodel",
        ".ckpt",
        ".onnx",
        ".pt",
        ".pth",
        ".prototxt",
        ".safetensors",
    }
    packaged_model_artifacts = {
        relative
        for relative in model_members
        if PurePosixPath(relative).suffix.casefold() in model_artifact_suffixes
    }
    if packaged_model_artifacts != set(declared):
        raise RuntimeError("candidate_model_set")
    if any(_sha256(by_relative[relative]) != digest for relative, digest in declared.items()):
        raise RuntimeError("candidate_model_digest")


def _safe_archive_entry(value: str) -> str:
    normalized = value.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or "\0" in normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in normalized
    ):
        raise RuntimeError("candidate_archive_unsafe_path")
    return relative.as_posix()


def _archive_payload(name: str, payload: bytes) -> bool:
    return PurePosixPath(name).suffix.casefold() in _ARCHIVE_SUFFIXES or payload.startswith(
        b"PK\x03\x04"
    )


def _scan_archive(payload: bytes, name: str, depth: int, state: dict[str, int]) -> None:
    if depth > _MAX_NESTED_ARCHIVE_DEPTH:
        raise RuntimeError("candidate_archive_depth")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError("candidate_archive_uninspectable") from error
    with archive:
        seen: set[str] = set()
        folded: set[str] = set()
        for info in archive.infolist():
            normalized = _safe_archive_entry(info.filename)
            if normalized in seen or normalized.casefold() in folded:
                raise RuntimeError("candidate_archive_duplicate")
            seen.add(normalized)
            folded.add(normalized.casefold())
            if info.flag_bits & 0x1:
                raise RuntimeError("candidate_archive_encrypted")
            file_type = (info.external_attr >> 16) & 0o170000
            if file_type == stat.S_IFLNK:
                raise RuntimeError("candidate_archive_unsafe_path")
            if info.is_dir():
                continue
            state["files"] += 1
            state["bytes"] += info.file_size
            if state["files"] > _MAX_NESTED_ARCHIVE_FILES:
                raise RuntimeError("candidate_archive_file_count")
            if info.file_size > _MAX_ARCHIVE_ENTRY_BYTES or state["bytes"] > _MAX_NESTED_ARCHIVE_BYTES:
                raise RuntimeError("candidate_archive_uncompressed_size")
            if (
                info.file_size >= 1024 * 1024
                and info.file_size > max(info.compress_size, 1) * _MAX_ARCHIVE_COMPRESSION_RATIO
            ):
                raise RuntimeError("candidate_archive_ratio")
            nested_name = f"{name}:{normalized}"
            if retired_packaged_entries([nested_name]):
                raise RuntimeError("candidate_archive_retired")
            try:
                nested_payload = archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise RuntimeError("candidate_archive_uninspectable") from error
            if _archive_payload(normalized, nested_payload):
                _scan_archive(nested_payload, nested_name, depth + 1, state)


def audit_packaged_archives(payload: list[tuple[str, Path]]) -> None:
    """Recursively audit ZIP-compatible runtime members under explicit bounds."""
    state = {"files": 0, "bytes": 0}
    for relative, path in payload:
        with path.open("rb") as stream:
            signature = stream.read(4)
        if (
            PurePosixPath(relative).suffix.casefold() not in _ARCHIVE_SUFFIXES
            and signature != b"PK\x03\x04"
        ):
            continue
        if path.stat().st_size > _MAX_NESTED_ARCHIVE_BYTES:
            raise RuntimeError("candidate_archive_uncompressed_size")
        _scan_archive(path.read_bytes(), relative, 1, state)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_relative(value: str) -> PurePosixPath:
    relative = PurePosixPath(value.replace("\\", "/"))
    if (
        not value
        or value != relative.as_posix()
        or relative.is_absolute()
        or ".." in relative.parts
        or ":" in value
    ):
        raise AssertionError("candidate archive unsafe path")
    return relative


def _payload_files(dist_root: Path) -> list[tuple[str, Path]]:
    if not dist_root.is_dir():
        raise RuntimeError("candidate_dist_missing")
    result = []
    for path in sorted(dist_root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("candidate_payload_symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError("candidate_payload_special")
        relative = path.relative_to(dist_root).as_posix()
        safe_relative = _safe_relative(relative)
        if not safe_relative.parts or safe_relative.parts[0] != LAYOUT["mainOnedir"]:
            raise RuntimeError("candidate_payload_layout")
        if retired_packaged_entries([relative]):
            raise RuntimeError("candidate_payload_retired")
        result.append((relative, path))
    if not result or len(result) > _MAX_ARCHIVE_FILES:
        raise RuntimeError("candidate_payload_count")
    result = sorted(result, key=lambda item: item[0])
    audit_packaged_archives(result)
    audit_packaged_model_files(result)
    return result


def _assert_source_identity(dist_root: Path, source_commit: str, repository_root: Path) -> None:
    try:
        actual_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        actual_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("candidate_source_repository") from error
    if Path(actual_root).resolve() != repository_root.resolve() or actual_head != source_commit:
        raise RuntimeError("candidate_source_commit")
    if tracked_status:
        raise RuntimeError("candidate_source_dirty")
    marker_root = dist_root / LAYOUT["mainOnedir"] / "_internal" / "vision"
    try:
        identity = load_build_identity(marker_root, require_source_commit=True)
    except RuntimeError as error:
        raise RuntimeError("candidate_build_commit") from error
    if identity.source_commit != source_commit:
        raise RuntimeError("candidate_build_commit")
    packaged_model_manifest = (
        dist_root
        / LAYOUT["mainOnedir"]
        / "_internal"
        / "models"
        / "model-manifest.json"
    )
    source_model_manifest = repository_root / "models" / "model-manifest.json"
    if (
        not packaged_model_manifest.is_file()
        or not source_model_manifest.is_file()
        or _sha256(packaged_model_manifest) != _sha256(source_model_manifest)
    ):
        raise RuntimeError("candidate_model_manifest")


def _build_manifest(
    dist_root: Path, source_commit: str, repository_root: Path
) -> tuple[dict, list[tuple[str, Path]]]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise RuntimeError("candidate_source_commit")
    _assert_source_identity(dist_root, source_commit, repository_root)
    payload = _payload_files(dist_root)
    files = [
        {"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)}
        for relative, path in payload
    ]
    by_path = {item["path"]: item for item in files}
    missing = [relative for relative in BINDING_PATHS.values() if relative not in by_path]
    if missing:
        raise RuntimeError(f"candidate_artifact_input_missing:{missing}")
    bindings = {
        name: {"path": relative, "sha256": by_path[relative]["sha256"]}
        for name, relative in BINDING_PATHS.items()
    }
    return {
        "schemaVersion": SCHEMA,
        "sourceCommit": source_commit,
        "layout": LAYOUT,
        "bindings": bindings,
        "files": files,
    }, payload


def _zip_info(relative: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_candidate_archive(
    dist_root: Path,
    artifact: Path,
    manifest_output: Path,
    *,
    source_commit: str,
    repository_root: Path | None = None,
) -> dict[str, str]:
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[1]
    manifest, payload = _build_manifest(
        dist_root.resolve(), source_commit, repository_root.resolve()
    )
    manifest_bytes = canonical_json(manifest).encode("utf-8")
    _write_atomic(manifest_output, manifest_bytes)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{artifact.name}-", dir=artifact.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            archive.writestr(_zip_info(EMBEDDED_MANIFEST), manifest_bytes)
            for relative, path in payload:
                with archive.open(_zip_info(relative), "w", force_zip64=True) as output, path.open("rb") as source:
                    shutil.copyfileobj(source, output, 1024 * 1024)
        os.replace(temporary, artifact)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "embeddedManifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "subjectSha256": _sha256(artifact),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-root", default="dist")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--repository-root", default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    result = write_candidate_archive(
        Path(args.dist_root).resolve(),
        Path(args.artifact).resolve(),
        Path(args.manifest_output).resolve(),
        source_commit=args.source_commit,
        repository_root=Path(args.repository_root).resolve(),
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
