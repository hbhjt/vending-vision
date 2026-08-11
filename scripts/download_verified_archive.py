from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class ArchiveError(RuntimeError):
    pass


_MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
_MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024 * 1024


def _safe_relative(name: str) -> PurePosixPath:
    value = PurePosixPath(name.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts or ":" in name:
        raise ArchiveError("archive_unsafe_path")
    return value


def _validate_members(members: list[tuple[PurePosixPath, bool, int]], max_extracted_bytes: int) -> None:
    seen: set[str] = set()
    files: set[str] = set()
    total = 0
    for relative, is_dir, size in members:
        if relative == PurePosixPath("."):
            if is_dir:
                continue
            raise ArchiveError("archive_unsafe_path")
        key = relative.as_posix().casefold()
        parent_keys = {parent.as_posix().casefold() for parent in relative.parents if parent != PurePosixPath(".")}
        if key in seen or parent_keys & files or (not is_dir and any(item.startswith(key + "/") for item in seen)):
            raise ArchiveError("archive_path_collision")
        seen.add(key)
        if not is_dir:
            files.add(key)
            total += size
            if total > max_extracted_bytes:
                raise ArchiveError("archive_extracted_size")


def _extract_archive(archive_path: Path, destination: Path, *, max_extracted_bytes: int) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            members = []
            for member in archive.infolist():
                relative = _safe_relative(member.filename)
                mode = member.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise ArchiveError("archive_symlink_or_special")
                members.append((relative, member.is_dir(), member.file_size))
            _validate_members(members, max_extracted_bytes)
            for member, (relative, _is_dir, _size) in zip(archive.infolist(), members):
                target = destination.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
        return
    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except tarfile.TarError as exc:
        raise ArchiveError("archive_format") from exc
    with archive:
        members = []
        for member in archive.getmembers():
            relative = _safe_relative(member.name)
            if not (member.isfile() or member.isdir()):
                raise ArchiveError("archive_symlink_or_special")
            members.append((relative, member.isdir(), member.size))
        _validate_members(members, max_extracted_bytes)
        for member, (relative, _is_dir, _size) in zip(archive.getmembers(), members):
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ArchiveError("archive_member")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)


def download_verified_archive(
    url: str,
    sha256: str,
    destination: Path,
    *,
    expected_bytes: int,
    opener=urlopen,
    max_download_bytes: int = _MAX_DOWNLOAD_BYTES,
    max_extracted_bytes: int = _MAX_EXTRACTED_BYTES,
) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or len(sha256) != 64:
        raise ArchiveError("archive_source")
    try:
        int(sha256, 16)
    except ValueError as exc:
        raise ArchiveError("archive_sha256") from exc
    if (
        type(expected_bytes) is not int
        or expected_bytes <= 0
        or type(max_download_bytes) is not int
        or max_download_bytes <= 0
        or expected_bytes > max_download_bytes
    ):
        raise ArchiveError("archive_download_size")
    if destination.exists():
        raise ArchiveError("archive_destination_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    archive_path = work / "payload.archive"
    extracted = work / "extracted"
    extracted.mkdir()
    try:
        digest = hashlib.sha256()
        request = Request(url, headers={"User-Agent": "vem-release-archive-fetcher/1"})
        with opener(request, timeout=120.0) as response:
            if response.geturl() != url:
                raise ArchiveError("archive_redirect_identity")
            with archive_path.open("xb") as output:
                downloaded = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    if downloaded + len(chunk) > expected_bytes or downloaded + len(chunk) > max_download_bytes:
                        raise ArchiveError("archive_download_size")
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
        if downloaded != expected_bytes:
            raise ArchiveError("archive_download_size")
        if digest.hexdigest() != sha256.lower():
            raise ArchiveError("archive_digest")
        _extract_archive(archive_path, extracted, max_extracted_bytes=max_extracted_bytes)
        os.replace(extracted, destination)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    download_verified_archive(
        args.url,
        args.sha256,
        Path(args.destination).resolve(),
        expected_bytes=args.expected_bytes,
    )
    print("Verified archive extracted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
