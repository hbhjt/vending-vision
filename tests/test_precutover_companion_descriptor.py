from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile

import pytest

from scripts.precutover_companion_descriptor import (
    DESCRIPTOR_MEMBER,
    build_archive,
    build_descriptor,
    canonical_bytes,
    validate_descriptor,
    verify_archive,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WindowsOrderedMember:
    """Path-shaped test member whose ordering follows Windows separators."""

    def __init__(self, path: Path, root: "WindowsOrderedRoot") -> None:
        self.path = path
        self.root = root

    def __lt__(self, other: "WindowsOrderedMember") -> bool:
        return (
            str(self.path).replace("/", "\\").casefold()
            < str(other.path).replace("/", "\\").casefold()
        )

    def is_symlink(self) -> bool:
        return self.path.is_symlink()

    def is_dir(self) -> bool:
        return self.path.is_dir()

    def is_file(self) -> bool:
        return self.path.is_file()

    def relative_to(self, root: "WindowsOrderedRoot") -> Path:
        assert root is self.root
        return self.path.relative_to(root.path)

    def stat(self):
        return self.path.stat()

    def open(self, *args, **kwargs):
        return self.path.open(*args, **kwargs)


class WindowsOrderedRoot:
    """A controlled Windows Path ordering seam for the public descriptor builder."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self) -> "WindowsOrderedRoot":
        return self

    def is_symlink(self) -> bool:
        return False

    def is_dir(self) -> bool:
        return True

    def rglob(self, pattern: str):
        assert pattern == "*"
        return [WindowsOrderedMember(path, self) for path in self.path.rglob("*")]


def fixture(root: Path) -> tuple[Path, dict, Path, str]:
    onedir = (root / "vending-vision-precutover-verifier").resolve()
    internal = onedir / "_internal"
    internal.mkdir(parents=True)
    (onedir / "vending-vision-precutover-verifier.exe").write_bytes(b"MZ-test-entry")
    (internal / "python311.dll").write_bytes(b"test-runtime")
    descriptor = build_descriptor(
        onedir,
        source_commit="a" * 40,
        toolchain={
            "pyinstaller": "6.16.0",
            "python": "3.11.9",
            "runnerImage": "Windows",
            "runnerImageVersion": "20260801.1",
        },
    )
    archive = root / "companion.zip"
    build_archive(onedir, descriptor, archive)
    descriptor_sha = hashlib.sha256(canonical_bytes(descriptor)).hexdigest()
    return onedir, descriptor, archive, descriptor_sha


def test_companion_archive_round_trip_binds_the_exact_onedir_and_entrypoint(tmp_path):
    _, descriptor, archive, descriptor_sha = fixture(tmp_path)
    destination = tmp_path / "verified"

    verified = verify_archive(
        archive,
        destination,
        expected_sha256=sha256(archive),
        expected_descriptor_sha256=descriptor_sha,
    )

    assert verified == descriptor
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    } == {item["path"] for item in descriptor["files"]}


def test_descriptor_builder_is_canonical_under_windows_path_ordering(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "a").mkdir()
    (payload / "a" / "child.bin").write_bytes(b"child")
    (payload / "a0.bin").write_bytes(b"sibling")
    (payload / "vending-vision-precutover-verifier.exe").write_bytes(b"entry")

    descriptor = build_descriptor(
        WindowsOrderedRoot(payload),
        source_commit="a" * 40,
        toolchain={
            "pyinstaller": "6.16.0",
            "python": "3.11.9",
            "runnerImage": "Windows",
            "runnerImageVersion": "test",
        },
    )

    assert [item["path"] for item in descriptor["files"]] == [
        "a/child.bin",
        "a0.bin",
        "vending-vision-precutover-verifier.exe",
    ]
    assert validate_descriptor(canonical_bytes(descriptor)) == descriptor


def rewrite_archive(
    source: Path,
    destination: Path,
    mutation: str,
) -> None:
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination, "w") as changed:
        infos = original.infolist()
        for index, info in enumerate(infos):
            if mutation == "missing" and info.filename.endswith("python311.dll"):
                continue
            name = info.filename
            data = original.read(info.filename)
            mode = stat.S_IFREG | 0o644
            compression = zipfile.ZIP_STORED
            if mutation == "path" and info.filename.endswith("python311.dll"):
                name = "../python311.dll"
            if mutation == "symlink" and info.filename.endswith("python311.dll"):
                mode = stat.S_IFLNK | 0o777
            if mutation == "special" and info.filename.endswith("python311.dll"):
                mode = stat.S_IFIFO | 0o600
            if mutation == "compression" and info.filename.endswith("python311.dll"):
                compression = zipfile.ZIP_DEFLATED
            if mutation == "descriptor" and info.filename == DESCRIPTOR_MEMBER:
                value = json.loads(data)
                value["sourceCommit"] = "b" * 40
                data = canonical_bytes(value)
            replacement = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            replacement.create_system = 3
            replacement.external_attr = mode << 16
            replacement.compress_type = compression
            changed.writestr(replacement, data)
        if mutation == "extra":
            changed.writestr("extra.exe", b"extra")
        if mutation == "case":
            changed.writestr("VENDING-VISION-PRECUTOVER-VERIFIER.EXE", b"collision")


@pytest.mark.parametrize(
    "mutation",
    ["descriptor", "extra", "missing", "path", "case", "symlink", "special", "compression"],
)
def test_companion_archive_rejects_tamper_and_unsafe_member_shapes(tmp_path, mutation):
    _, _, archive, descriptor_sha = fixture(tmp_path)
    changed = tmp_path / f"{mutation}.zip"
    rewrite_archive(archive, changed, mutation)

    with pytest.raises(AssertionError):
        verify_archive(
            changed,
            tmp_path / f"verified-{mutation}",
            expected_sha256=sha256(changed),
            expected_descriptor_sha256=descriptor_sha,
        )

    assert not (tmp_path / f"verified-{mutation}").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "symlink",
        pytest.param(
            "special",
            marks=pytest.mark.skipif(
                not hasattr(os, "mkfifo"), reason="POSIX FIFO fixture"
            ),
        ),
    ],
)
def test_descriptor_builder_rejects_non_regular_payload_members(tmp_path, kind):
    root = (tmp_path / "payload").resolve()
    root.mkdir()
    (root / "vending-vision-precutover-verifier.exe").write_bytes(b"entry")
    unsafe = root / "unsafe"
    if kind == "symlink":
        unsafe.symlink_to(root / "vending-vision-precutover-verifier.exe")
    else:
        unsafe.mkdir()
        # A directory is allowed, so replace it with a FIFO to exercise a special file.
        unsafe.rmdir()
        os.mkfifo(unsafe)

    with pytest.raises(AssertionError, match="symlink|special"):
        build_descriptor(
            root,
            source_commit="a" * 40,
            toolchain={
                "pyinstaller": "6.16.0",
                "python": "3.11.9",
                "runnerImage": "Windows",
                "runnerImageVersion": "test",
            },
        )
