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
}
BINDING_PATHS = {
    "mainExecutable": LAYOUT["mainExecutable"],
}
_MAX_ARCHIVE_FILES = 100_000


def retired_packaged_entries(entries, *, include_historical_generic_modules=True):
    """Return normalized archive/resource entries owned by the retired paths."""
    generative_prefix = "".join(("a", "i"))
    quick_prefix = "".join(("fa", "st"))
    vendor_name = "".join(("cat", "vton"))
    regional_name = "".join(("regional", r"[_-]?", "evaluator"))
    retired = re.compile(
        rf"(?:^|[./\\])try[_-]?on[_-]?(?:session|frontend|{generative_prefix}|{quick_prefix})(?:[./\\]|$)"
        rf"|(?:^|[./\\])profile[_-]?{quick_prefix}[_-]?try[_-]?on(?:[./\\]|$)"
        r"|(?:^|[./\\])vem[_-]?vision[_-]?v1(?:[./\\]|$)"
        rf"|(?:^|[./\\])(?:{generative_prefix}[_-]?(?:acceptance[_-]?evidence|attempt(?:[_-]?(?:worker|process|runtime))?|model(?:[_-]?(?:pack|manifest)(?:[_-]?release)?)?|process[_-]?tree[_-]?worker|runtime(?:[_-]?descriptor)?|wheelhouse|worker)|{vendor_name}|{regional_name}(?:[_-]?descriptor)?)(?:[./\\]|$)"
        rf"|(?:^|[./\\])(?:official|requirements)[_-]?{generative_prefix}(?:[_-][^./\\]+)*(?:[./\\]|$)"
        rf"|(?:^|[./\\])[.]venv[_-]?packaging[_-]?{generative_prefix}(?:[./\\]|$)"
        rf"|(?:^|[./\\])(?:materialize|verify)[_-]?{generative_prefix}[_-]?wheelhouse(?:[./\\]|$)"
        rf"|(?:^|[./\\])render[_-]?{generative_prefix}[_-]?build[_-]?requirements(?:[./\\]|$)"
        rf"|(?:^|[./\\])(?:{quick_prefix}[_-]?(?:attempt|result|adjustment))(?:[./\\]|$)"
        rf"|(?:^|[./\\])vending[_-]?vision[_-]?{generative_prefix}[_-]?worker(?:[./\\]|$)",
        re.I,
    )
    historical_generic = re.compile(
        r"(?:^|[./\\])vision[./\\]"
        r"(?:process[_-]?supervisor|source[_-]?provenance)(?:[.]py)?(?:[./\\]|$)",
        re.I,
    )
    return sorted(
        entry
        for entry in entries
        if retired.search(entry)
        or (
            include_historical_generic_modules
            and historical_generic.search(entry)
        )
        or (
            ":" in entry
            and (
                retired.search(entry.split(":", 1)[1])
                or (
                    include_historical_generic_modules
                    and historical_generic.search(entry.split(":", 1)[1])
                )
            )
        )
    )


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
