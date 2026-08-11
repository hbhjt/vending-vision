from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile


SCHEMA = "vending-vision-candidate-artifact/v3"
EMBEDDED_MANIFEST = "candidate-manifest.json"
LAYOUT = {
    "mainOnedir": "vending-vision",
    "mainExecutable": "vending-vision/vending-vision.exe",
    "workerOnedir": "vending-vision-ai-worker",
    "workerExecutable": "vending-vision-ai-worker/vending-vision-ai-worker.exe",
    "workerInternal": "vending-vision-ai-worker/_internal",
}
BINDING_PATHS = {
    "mainExecutable": LAYOUT["mainExecutable"],
    "workerExecutable": LAYOUT["workerExecutable"],
    "runtimeDescriptor": f'{LAYOUT["workerInternal"]}/ai-runtime-descriptor.json',
    "aiLock": f'{LAYOUT["workerInternal"]}/requirements-ai.lock.json',
    "sourceDescriptor": f'{LAYOUT["workerInternal"]}/official-ai-source-descriptor.json',
    "modelPackDescriptor": f'{LAYOUT["workerInternal"]}/official-ai-model-pack-descriptor.json',
}
_MAX_ARCHIVE_FILES = 100_000
_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


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
        _safe_relative(relative)
        result.append((relative, path))
    if not result or len(result) > _MAX_ARCHIVE_FILES:
        raise RuntimeError("candidate_payload_count")
    return sorted(result, key=lambda item: item[0])


def _build_manifest(dist_root: Path, source_commit: str) -> tuple[dict, list[tuple[str, Path]]]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise RuntimeError("candidate_source_commit")
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
) -> dict[str, str]:
    manifest, payload = _build_manifest(dist_root.resolve(), source_commit)
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


def _load_manifest(raw: bytes, expected_digest: str, expected_source_commit: str) -> dict:
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise AssertionError("embedded manifest digest mismatch")
    try:
        manifest = json.loads(raw)
    except ValueError as exc:
        raise AssertionError("embedded manifest invalid") from exc
    if canonical_json(manifest).encode("utf-8") != raw:
        raise AssertionError("embedded manifest is not canonical")
    if set(manifest) != {"schemaVersion", "sourceCommit", "layout", "bindings", "files"}:
        raise AssertionError("embedded manifest shape mismatch")
    if manifest["schemaVersion"] != SCHEMA or manifest["layout"] != LAYOUT:
        raise AssertionError("embedded manifest contract mismatch")
    if manifest["sourceCommit"] != expected_source_commit:
        raise AssertionError("embedded manifest source commit mismatch")
    return manifest


def verify_candidate_archive(
    artifact: Path,
    destination: Path,
    *,
    expected_subject_sha256: str,
    expected_manifest_sha256: str,
    expected_source_commit: str,
) -> dict:
    if re.fullmatch(r"[0-9a-f]{64}", expected_subject_sha256) is None or re.fullmatch(
        r"[0-9a-f]{64}", expected_manifest_sha256
    ) is None:
        raise AssertionError("external candidate trust digest missing")
    if not artifact.is_file() or _sha256(artifact) != expected_subject_sha256:
        raise AssertionError("trusted subject digest mismatch")
    if not zipfile.is_zipfile(artifact):
        raise AssertionError("candidate artifact is not a ZIP")
    if destination.exists():
        raise AssertionError("candidate extraction destination exists")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        with zipfile.ZipFile(artifact) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > _MAX_ARCHIVE_FILES + 1:
                raise AssertionError("candidate archive file count")
            by_name: dict[str, zipfile.ZipInfo] = {}
            seen_casefold: set[str] = set()
            total = 0
            for info in infos:
                relative = _safe_relative(info.filename)
                key = relative.as_posix().casefold()
                mode = info.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if (
                    info.is_dir()
                    or info.compress_type != zipfile.ZIP_STORED
                    or file_type not in {0, stat.S_IFREG}
                ):
                    raise AssertionError("candidate archive symlink special or compressed entry")
                if key in seen_casefold:
                    raise AssertionError("candidate archive path collision")
                seen_casefold.add(key)
                by_name[relative.as_posix()] = info
                total += info.file_size
                if total > _MAX_ARCHIVE_BYTES:
                    raise AssertionError("candidate archive extracted size")
            manifest_info = by_name.get(EMBEDDED_MANIFEST)
            if manifest_info is None or manifest_info.file_size > _MAX_MANIFEST_BYTES:
                raise AssertionError("embedded manifest missing or oversized")
            manifest_raw = archive.read(manifest_info)
            manifest = _load_manifest(
                manifest_raw, expected_manifest_sha256, expected_source_commit
            )
            files = manifest["files"]
            if not isinstance(files, list) or not files:
                raise AssertionError("embedded manifest files missing")
            expected_files: dict[str, dict] = {}
            previous = ""
            for item in files:
                if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
                    raise AssertionError("embedded manifest file shape")
                relative = _safe_relative(item["path"]).as_posix()
                if (
                    relative <= previous
                    or type(item["size"]) is not int
                    or item["size"] < 0
                    or re.fullmatch(r"[0-9a-f]{64}", item["sha256"] or "") is None
                ):
                    raise AssertionError("embedded manifest file value")
                previous = relative
                expected_files[relative] = item
            if set(by_name) != {EMBEDDED_MANIFEST, *expected_files}:
                raise AssertionError("candidate archive payload set mismatch")
            bindings = manifest["bindings"]
            if not isinstance(bindings, dict) or set(bindings) != set(BINDING_PATHS):
                raise AssertionError("embedded manifest bindings mismatch")
            for name, relative in BINDING_PATHS.items():
                binding = bindings[name]
                if (
                    not isinstance(binding, dict)
                    or set(binding) != {"path", "sha256"}
                    or binding["path"] != relative
                    or relative not in expected_files
                    or binding["sha256"] != expected_files[relative]["sha256"]
                ):
                    raise AssertionError("embedded manifest binding mismatch")

            for relative, item in expected_files.items():
                info = by_name[relative]
                if info.file_size != item["size"]:
                    raise AssertionError("candidate archive payload size mismatch")
                target = staging.joinpath(*PurePosixPath(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with archive.open(info) as source, target.open("xb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        if size + len(chunk) > item["size"]:
                            raise AssertionError("candidate archive payload size mismatch")
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if size != item["size"] or digest.hexdigest() != item["sha256"]:
                    raise AssertionError("candidate archive payload digest mismatch")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "root": destination,
        "mainExecutable": destination / LAYOUT["mainExecutable"],
        "workerExecutable": destination / LAYOUT["workerExecutable"],
        "manifest": manifest,
        "manifestSha256": expected_manifest_sha256,
        "subjectSha256": expected_subject_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-root", default="dist")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    result = write_candidate_archive(
        Path(args.dist_root).resolve(),
        Path(args.artifact).resolve(),
        Path(args.manifest_output).resolve(),
        source_commit=args.source_commit,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
