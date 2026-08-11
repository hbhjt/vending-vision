import hashlib
import io
import stat
import tarfile
import zipfile

import pytest

from scripts.download_verified_archive import ArchiveError, download_verified_archive


class _Response(io.BytesIO):
    def __init__(self, payload, url):
        super().__init__(payload)
        self._url = url

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _zip(member):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(member, b"wheel")
    return output.getvalue()


def test_verified_archive_download_extracts_only_digest_bound_safe_members(tmp_path):
    url = "https://example.invalid/core-wheelhouse.zip"
    payload = _zip("demo.whl")

    def opener(_request, timeout):
        assert timeout == 120.0
        return _Response(payload, url)

    destination = tmp_path / "wheelhouse"
    download_verified_archive(url, hashlib.sha256(payload).hexdigest(), destination, opener=opener)

    assert (destination / "demo.whl").read_bytes() == b"wheel"


def test_verified_archive_rejects_traversal_without_publishing_destination(tmp_path):
    url = "https://example.invalid/core-wheelhouse.zip"
    payload = _zip("../escape.whl")

    def opener(_request, timeout):
        return _Response(payload, url)

    destination = tmp_path / "wheelhouse"
    with pytest.raises(ArchiveError, match="archive_unsafe_path"):
        download_verified_archive(url, hashlib.sha256(payload).hexdigest(), destination, opener=opener)

    assert not destination.exists()
    assert not (tmp_path / "escape.whl").exists()


@pytest.mark.parametrize("case", ["symlink", "collision", "oversize", "special"])
def test_verified_archive_rejects_symlink_special_collision_and_size(tmp_path, case):
    if case == "special":
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            member = tarfile.TarInfo("fifo")
            member.type = tarfile.FIFOTYPE
            archive.addfile(member)
        payload = output.getvalue()
    else:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            if case == "symlink":
                member = zipfile.ZipInfo("link")
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(member, "target")
            elif case == "collision":
                archive.writestr("Demo.whl", b"one")
                archive.writestr("demo.whl", b"two")
            else:
                archive.writestr("large.whl", b"12345")
        payload = output.getvalue()
    url = "https://example.invalid/core-wheelhouse.archive"

    def opener(_request, timeout):
        return _Response(payload, url)

    with pytest.raises(ArchiveError, match="archive_"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            tmp_path / "wheelhouse",
            opener=opener,
            max_extracted_bytes=4 if case == "oversize" else 1024,
        )
    assert not (tmp_path / "wheelhouse").exists()
