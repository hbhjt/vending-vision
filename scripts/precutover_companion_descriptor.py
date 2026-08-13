"""Build and verify a deterministic standalone pre-cutover companion archive."""
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


SCHEMA = "vending-vision-precutover-companion/v1"
DESCRIPTOR_MEMBER = "precutover-companion-descriptor.json"
ENTRYPOINT = "vending-vision-precutover-verifier.exe"
MAX_FILES = 20_000
MAX_BYTES = 1024 * 1024 * 1024


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        not value
        or value != relative.as_posix()
        or relative.is_absolute()
        or ".." in relative.parts
        or ":" in value
    ):
        raise AssertionError("companion unsafe path")
    return relative


def _payload(root: Path) -> list[tuple[str, Path]]:
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise AssertionError("companion root is unsafe")
    files: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.relative_to(root).as_posix()):
        if path.is_symlink():
            raise AssertionError("companion payload symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise AssertionError("companion payload special file")
        relative = path.relative_to(root).as_posix()
        _safe_relative(relative)
        files.append((relative, path))
    if not files or len(files) > MAX_FILES or sum(path.stat().st_size for _, path in files) > MAX_BYTES:
        raise AssertionError("companion payload bounds")
    return files


def build_descriptor(root: Path, *, source_commit: str, toolchain: dict[str, str]) -> dict:
    if re.fullmatch(r"[a-f0-9]{40}", source_commit) is None:
        raise AssertionError("companion source commit")
    if set(toolchain) != {"pyinstaller", "python", "runnerImage", "runnerImageVersion"}:
        raise AssertionError("companion toolchain shape")
    files = [
        {"byteSize": path.stat().st_size, "path": relative, "sha256": sha256_file(path)}
        for relative, path in _payload(root)
    ]
    entry = next((item for item in files if item["path"] == ENTRYPOINT), None)
    if entry is None:
        raise AssertionError("companion entrypoint missing")
    return {
        "entrypoint": {"path": ENTRYPOINT, "sha256": entry["sha256"]},
        "files": files,
        "schemaVersion": SCHEMA,
        "sourceCommit": source_commit,
        "toolchain": toolchain,
    }


def validate_descriptor(raw: bytes) -> dict:
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise AssertionError("companion descriptor JSON") from exc
    if canonical_bytes(value) != raw:
        raise AssertionError("companion descriptor noncanonical")
    if set(value) != {"entrypoint", "files", "schemaVersion", "sourceCommit", "toolchain"}:
        raise AssertionError("companion descriptor shape")
    if value["schemaVersion"] != SCHEMA or re.fullmatch(r"[a-f0-9]{40}", value["sourceCommit"]) is None:
        raise AssertionError("companion descriptor identity")
    if set(value["entrypoint"]) != {"path", "sha256"} or value["entrypoint"]["path"] != ENTRYPOINT:
        raise AssertionError("companion descriptor entrypoint")
    if set(value["toolchain"]) != {"pyinstaller", "python", "runnerImage", "runnerImageVersion"}:
        raise AssertionError("companion descriptor toolchain")
    previous = ""
    paths = set()
    for item in value["files"]:
        if set(item) != {"byteSize", "path", "sha256"}:
            raise AssertionError("companion descriptor file shape")
        _safe_relative(item["path"])
        if (
            item["path"] <= previous
            or item["path"].casefold() in paths
            or type(item["byteSize"]) is not int
            or item["byteSize"] < 0
            or re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) is None
        ):
            raise AssertionError("companion descriptor file identity")
        previous = item["path"]
        paths.add(item["path"].casefold())
    entry = next((item for item in value["files"] if item["path"] == ENTRYPOINT), None)
    if entry is None or entry["sha256"] != value["entrypoint"]["sha256"]:
        raise AssertionError("companion descriptor entrypoint binding")
    return value


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def build_archive(root: Path, descriptor: dict, output: Path) -> None:
    descriptor_raw = canonical_bytes(descriptor)
    files = _payload(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor_fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}-", dir=output.parent)
    os.close(descriptor_fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            archive.writestr(_zip_info(DESCRIPTOR_MEMBER), descriptor_raw)
            for relative, path in files:
                with archive.open(_zip_info(relative), "w", force_zip64=True) as destination, path.open("rb") as source:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def verify_archive(archive_path: Path, destination: Path, *, expected_sha256: str, expected_descriptor_sha256: str) -> dict:
    if (
        re.fullmatch(r"[a-f0-9]{64}", expected_sha256) is None
        or re.fullmatch(r"[a-f0-9]{64}", expected_descriptor_sha256) is None
        or archive_path.is_symlink()
        or not archive_path.is_file()
        or sha256_file(archive_path) != expected_sha256
        or destination.exists()
    ):
        raise AssertionError("companion external identity")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_FILES + 1:
                raise AssertionError("companion archive file count")
            names: dict[str, zipfile.ZipInfo] = {}
            folded = set()
            total = 0
            for info in infos:
                relative = _safe_relative(info.filename).as_posix()
                mode = stat.S_IFMT(info.external_attr >> 16)
                key = relative.casefold()
                if (
                    info.is_dir()
                    or info.compress_type != zipfile.ZIP_STORED
                    or mode not in {0, stat.S_IFREG}
                    or key in folded
                ):
                    raise AssertionError("companion archive unsafe member")
                total += info.file_size
                if total > MAX_BYTES:
                    raise AssertionError("companion archive size")
                folded.add(key)
                names[relative] = info
            descriptor_info = names.get(DESCRIPTOR_MEMBER)
            if descriptor_info is None or descriptor_info.file_size > 16 * 1024 * 1024:
                raise AssertionError("companion descriptor missing")
            descriptor_raw = archive.read(descriptor_info)
            if hashlib.sha256(descriptor_raw).hexdigest() != expected_descriptor_sha256:
                raise AssertionError("companion descriptor digest")
            descriptor = validate_descriptor(descriptor_raw)
            expected = {DESCRIPTOR_MEMBER, *(item["path"] for item in descriptor["files"])}
            if set(names) != expected:
                raise AssertionError("companion archive member set")
            for item in descriptor["files"]:
                info = names[item["path"]]
                if info.file_size != item["byteSize"]:
                    raise AssertionError("companion file size")
                target = staging.joinpath(*PurePosixPath(item["path"]).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with archive.open(info) as source, target.open("xb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        size += len(chunk)
                        if size > item["byteSize"]:
                            raise AssertionError("companion file size")
                        digest.update(chunk)
                        output.write(chunk)
                if size != item["byteSize"] or digest.hexdigest() != item["sha256"]:
                    raise AssertionError("companion file digest")
        os.replace(staging, destination)
        return descriptor
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", required=True, type=Path)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--python-version", required=True)
    build.add_argument("--pyinstaller-version", required=True)
    build.add_argument("--runner-image", required=True)
    build.add_argument("--runner-image-version", required=True)
    build.add_argument("--descriptor-output", required=True, type=Path)
    build.add_argument("--archive-output", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--archive", required=True, type=Path)
    verify.add_argument("--destination", required=True, type=Path)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--expected-descriptor-sha256", required=True)
    args = parser.parse_args()
    if args.command == "verify":
        descriptor = verify_archive(
            args.archive.resolve(),
            args.destination.resolve(),
            expected_sha256=args.expected_sha256,
            expected_descriptor_sha256=args.expected_descriptor_sha256,
        )
        print(json.dumps({"entrypoint": descriptor["entrypoint"], "schemaVersion": descriptor["schemaVersion"]}, sort_keys=True, separators=(",", ":")))
        return 0
    descriptor = build_descriptor(
        args.root.resolve(),
        source_commit=args.source_commit,
        toolchain={
            "pyinstaller": args.pyinstaller_version,
            "python": args.python_version,
            "runnerImage": args.runner_image,
            "runnerImageVersion": args.runner_image_version,
        },
    )
    raw = canonical_bytes(descriptor)
    args.descriptor_output.write_bytes(raw)
    build_archive(args.root.resolve(), descriptor, args.archive_output.resolve())
    print(json.dumps({
        "archiveByteSize": args.archive_output.stat().st_size,
        "archiveSha256": sha256_file(args.archive_output),
        "descriptorSha256": hashlib.sha256(raw).hexdigest(),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
