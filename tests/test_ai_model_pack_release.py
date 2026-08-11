import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.ai_model_pack_release import (
    build_model_pack_release_manifest,
    build_model_pack_zip,
    descriptor_sha256,
    install_model_pack_zip,
    verify_model_pack_zip,
)
from vision.ai_model_pack import (
    MANIFEST_NAME,
    AiModelPackError,
    canonical_ai_model_manifest_json,
    verify_ai_model_pack,
)


def mini_descriptor(root: Path) -> dict:
    files = []
    for relative, content in {
        "a/config.json": b"{}",
        "weights/model.bin": b"mini-weights",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append(
            {
                "path": relative,
                "upstreamPath": relative,
                "upstream": "mini",
                "role": relative.replace("/", "_").replace(".", "_"),
                "format": relative.rsplit(".", 1)[-1],
                "byteSize": path.stat().st_size,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "schemaVersion": "vem-official-ai-model-pack-descriptor/v2",
        "catvtonSourceRevision": "mini-source",
        "totalByteSize": sum(item["byteSize"] for item in files),
        "upstreams": [{"id": "mini", "repository": "example/mini", "revision": "abc"}],
        "files": sorted(files, key=lambda item: item["path"]),
    }


def test_model_pack_zip_build_is_deterministic_and_verifiable(tmp_path):
    source = tmp_path / "source"
    descriptor = mini_descriptor(source)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_sha = build_model_pack_zip(source, first, descriptor)
    second_sha = build_model_pack_zip(source, second, descriptor)

    assert first.read_bytes() == second.read_bytes()
    assert first_sha == second_sha
    assert verify_model_pack_zip(first, descriptor, outer_sha256=first_sha) == first_sha
    release = build_model_pack_release_manifest(first, descriptor, outer_sha256=first_sha)
    assert release["archive"]["sha256"] == first_sha
    assert release["descriptor"]["sha256"] == descriptor_sha256(descriptor)
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == sorted([MANIFEST_NAME, "a/config.json", "weights/model.bin"])
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.extra == b""
            assert info.comment == b""


@pytest.mark.parametrize("bad_name", ["../escape.bin", "weights\\model.bin", "weights/model.bin:ads", "Weights/model.bin"])
def test_model_pack_zip_checker_rejects_unsafe_or_duplicate_entries(tmp_path, bad_name):
    source = tmp_path / "source"
    descriptor = mini_descriptor(source)
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(MANIFEST_NAME, canonical_ai_model_manifest_json(descriptor))
        archive.writestr("a/config.json", b"{}")
        archive.writestr("weights/model.bin", b"mini-weights")
        archive.writestr(bad_name, b"bad")

    with pytest.raises(AiModelPackError):
        verify_model_pack_zip(archive_path, descriptor)


def test_model_pack_installer_is_idempotent_and_failed_install_keeps_active(tmp_path):
    source = tmp_path / "source"
    descriptor = mini_descriptor(source)
    archive_path = tmp_path / "pack.zip"
    digest = build_model_pack_zip(source, archive_path, descriptor)
    install_root = tmp_path / "install"

    active = install_model_pack_zip(archive_path, install_root, descriptor, outer_sha256=digest)
    second = install_model_pack_zip(archive_path, install_root, descriptor, outer_sha256=digest)

    assert active == second
    assert verify_ai_model_pack(active, descriptor=descriptor).root == active.resolve()
    active_record = json.loads((install_root / "active-pack.json").read_text("utf-8"))
    assert active_record["schemaVersion"] == "vem-ai-model-pack-selection/v1"
    assert active_record["archiveSha256"] == digest
    assert active_record["installDigest"] == digest
    assert active_record["descriptorSha256"] == descriptor_sha256(descriptor)

    bad_archive = tmp_path / "bad.zip"
    bad_archive.write_bytes(archive_path.read_bytes() + b"tamper")
    with pytest.raises(AiModelPackError, match="ai_model_zip_outer_digest"):
        install_model_pack_zip(bad_archive, install_root, descriptor, outer_sha256=digest)
    assert json.loads((install_root / "active-pack.json").read_text("utf-8")) == active_record
