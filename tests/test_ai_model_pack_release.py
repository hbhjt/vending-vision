import hashlib
import asyncio
import json
import struct
import subprocess
import sys
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
from scripts.precutover_ai_model_pack import verify_and_install_model_pack
from scripts.precutover_ai_worker_probe import probe_worker
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


def test_model_pack_zip_build_and_verify_supports_required_zip64_metadata(tmp_path, monkeypatch):
    source = tmp_path / "source"
    descriptor = mini_descriptor(source)
    payload = b"large-for-test" * 128
    model = source / "weights/model.bin"
    model.write_bytes(payload)
    descriptor["files"][1]["byteSize"] = len(payload)
    descriptor["files"][1]["sha256"] = hashlib.sha256(payload).hexdigest()
    descriptor["totalByteSize"] = sum(item["byteSize"] for item in descriptor["files"])
    archive_path = tmp_path / "zip64-pack.zip"
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1024)

    digest = build_model_pack_zip(source, archive_path, descriptor)

    with zipfile.ZipFile(archive_path) as archive:
        model_info = archive.getinfo("weights/model.bin")
        assert model_info.extra.startswith(b"\x01\x00")
    assert verify_model_pack_zip(archive_path, descriptor, outer_sha256=digest) == digest


def test_model_pack_zip_build_and_verify_supports_required_zip64_offset_metadata(tmp_path, monkeypatch):
    source = tmp_path / "source"
    descriptor = mini_descriptor(source)
    payload = b"large-for-test" * 128
    original_model = source / "weights/model.bin"
    original_model.unlink()
    model = source / "a-large/model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(payload)
    descriptor["files"][1]["path"] = "a-large/model.bin"
    descriptor["files"][1]["upstreamPath"] = "a-large/model.bin"
    descriptor["files"][1]["byteSize"] = len(payload)
    descriptor["files"][1]["sha256"] = hashlib.sha256(payload).hexdigest()
    descriptor["files"] = sorted(descriptor["files"], key=lambda item: item["path"])
    descriptor["totalByteSize"] = sum(item["byteSize"] for item in descriptor["files"])
    archive_path = tmp_path / "zip64-offset-pack.zip"
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1024)

    digest = build_model_pack_zip(source, archive_path, descriptor)

    with zipfile.ZipFile(archive_path) as archive:
        large_info = archive.getinfo("a-large/model.bin")
        later_small_info = archive.getinfo("a/config.json")
        assert len(large_info.extra) == 20
        assert later_small_info.file_size < zipfile.ZIP64_LIMIT
        assert later_small_info.header_offset > zipfile.ZIP64_LIMIT
        assert len(later_small_info.extra) == 12
    assert verify_model_pack_zip(archive_path, descriptor, outer_sha256=digest) == digest


def test_model_pack_zip_checker_rejects_zip64_entry_with_unknown_extra(tmp_path, monkeypatch):
    source = tmp_path / "source"
    descriptor = mini_descriptor(source)
    payload = b"large-for-test" * 128
    model = source / "weights/model.bin"
    model.write_bytes(payload)
    descriptor["files"][1]["byteSize"] = len(payload)
    descriptor["files"][1]["sha256"] = hashlib.sha256(payload).hexdigest()
    descriptor["totalByteSize"] = sum(item["byteSize"] for item in descriptor["files"])
    archive_path = tmp_path / "zip64-pack.zip"
    tampered = tmp_path / "zip64-extra.zip"
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1024)
    build_model_pack_zip(source, archive_path, descriptor)

    with zipfile.ZipFile(archive_path) as src, zipfile.ZipFile(tampered, "w", allowZip64=True) as dst:
        for info in src.infolist():
            replacement = zipfile.ZipInfo(info.filename, info.date_time)
            replacement.compress_type = info.compress_type
            replacement.external_attr = info.external_attr
            if info.filename == "weights/model.bin":
                replacement.extra = b"\xfe\xca\x00\x00"
            dst.writestr(replacement, src.read(info.filename))

    with pytest.raises(AiModelPackError, match="ai_model_zip_metadata"):
        verify_model_pack_zip(tampered, descriptor)


@pytest.mark.parametrize("copies", [1, 2])
def test_model_pack_zip_checker_rejects_unnecessary_or_duplicate_zip64_extra(tmp_path, copies):
    source = tmp_path / "source"
    descriptor = mini_descriptor(source)
    archive_path = tmp_path / "noncanonical.zip"
    zip64_extra = struct.pack("<HHQQ", 0x0001, 16, len(b"mini-weights"), len(b"mini-weights"))

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(MANIFEST_NAME, canonical_ai_model_manifest_json(descriptor))
        archive.writestr("a/config.json", b"{}")
        info = zipfile.ZipInfo("weights/model.bin")
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o100644 << 16
        info.extra = zip64_extra * copies
        archive.writestr(info, b"mini-weights")

    with pytest.raises(AiModelPackError, match="ai_model_zip_metadata"):
        verify_model_pack_zip(archive_path, descriptor)


def test_precutover_proof_requires_actual_archive_and_installs_verified_bytes(tmp_path):
    source = tmp_path / "source"
    descriptor = mini_descriptor(source)
    archive = tmp_path / "model-pack.zip"
    archive_sha = build_model_pack_zip(source, archive, descriptor)
    descriptor_text = canonical_ai_model_manifest_json(descriptor)
    descriptor_path = tmp_path / "official-ai-model-pack-descriptor.json"
    descriptor_path.write_text(descriptor_text, "utf-8")

    report = verify_and_install_model_pack(
        archive=archive,
        descriptor_path=descriptor_path,
        expected_archive_byte_size=archive.stat().st_size,
        expected_archive_sha256=archive_sha,
        expected_descriptor_sha256=hashlib.sha256(
            descriptor_text.encode("utf-8"),
        ).hexdigest(),
        install_root=tmp_path / "private-install",
    )

    assert report["archive"] == {
        "byteSize": archive.stat().st_size,
        "sha256": archive_sha,
    }
    assert report["descriptor"]["catvtonSourceRevision"] == "mini-source"
    assert Path(report["installedPack"]).is_dir()


def test_precutover_verifier_imports_with_the_pinned_stdlib_python():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0,{str(Path(__file__).resolve().parents[1])!r});"
                "import scripts.precutover_ai_model_pack"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_precutover_worker_probe_uses_production_tree_supervisor(tmp_path):
    worker = tmp_path / "worker"
    worker.write_text(
        "#!/usr/bin/python3\nimport json,sys\nprint(json.dumps({'probe':'official-catvton-worker-runtime'}))\n",
        "utf-8",
    )
    worker.chmod(0o700)

    result = asyncio.run(
        probe_worker(worker=worker, mode="runtime", model_pack=None, timeout=2.0)
    )

    assert result["returncode"] == 0
    assert json.loads(result["stdout"])["probe"] == "official-catvton-worker-runtime"


def test_model_pack_zip_checker_streams_each_entry_and_rejects_same_size_tamper(tmp_path):
    source = tmp_path / "source"
    descriptor = mini_descriptor(source)
    archive_path = tmp_path / "pack.zip"
    build_model_pack_zip(source, archive_path, descriptor)

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive_path) as src, zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "weights/model.bin":
                data = b"X" + data[1:]
            replacement = zipfile.ZipInfo(info.filename, info.date_time)
            replacement.compress_type = info.compress_type
            replacement.external_attr = info.external_attr
            dst.writestr(replacement, data)

    with pytest.raises(AiModelPackError, match="ai_model_zip_entry_digest"):
        verify_model_pack_zip(tampered, descriptor)


def test_model_pack_zip_checker_rejects_compressed_zip_bomb_before_extracting(tmp_path):
    source = tmp_path / "source"
    descriptor = mini_descriptor(source)
    archive_path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, canonical_ai_model_manifest_json(descriptor))
        archive.writestr("a/config.json", b"{}")
        archive.writestr("weights/model.bin", b"mini-weights")

    with pytest.raises(AiModelPackError, match="ai_model_zip_metadata"):
        verify_model_pack_zip(archive_path, descriptor)


def test_model_pack_zip_checker_rejects_symlink_entry_before_extracting(tmp_path):
    source = tmp_path / "source"
    descriptor = mini_descriptor(source)
    archive_path = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(MANIFEST_NAME, canonical_ai_model_manifest_json(descriptor))
        archive.writestr("a/config.json", b"{}")
        info = zipfile.ZipInfo("weights/model.bin")
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o120777 << 16
        archive.writestr(info, b"mini-weights")

    with pytest.raises(AiModelPackError, match="ai_model_zip_metadata"):
        verify_model_pack_zip(archive_path, descriptor)


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


def test_model_pack_installer_bad_payload_cannot_replace_active_or_leave_evidence(tmp_path):
    source = tmp_path / "source"
    descriptor = mini_descriptor(source)
    archive_path = tmp_path / "pack.zip"
    digest = build_model_pack_zip(source, archive_path, descriptor)
    install_root = tmp_path / "install"
    install_model_pack_zip(archive_path, install_root, descriptor, outer_sha256=digest)
    active_record = json.loads((install_root / "active-pack.json").read_text("utf-8"))

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive_path) as src, zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "weights/model.bin":
                data = b"X" + data[1:]
            replacement = zipfile.ZipInfo(info.filename, info.date_time)
            replacement.compress_type = info.compress_type
            replacement.external_attr = info.external_attr
            dst.writestr(replacement, data)

    with pytest.raises(AiModelPackError, match="ai_model_zip_entry_digest"):
        install_model_pack_zip(tampered, install_root, descriptor)

    assert json.loads((install_root / "active-pack.json").read_text("utf-8")) == active_record
    assert not any(path.name.startswith(".staging-") for path in (install_root / "packs").iterdir())
