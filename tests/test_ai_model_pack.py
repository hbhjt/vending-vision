import hashlib
import json
import os
import asyncio
import time
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest

from vision.ai_model_pack import (
    AiModelPackError,
    canonical_ai_model_manifest_json,
    create_ai_model_manifest,
    load_official_ai_model_pack_descriptor,
    official_ai_readiness,
    official_ai_readiness_snapshot,
    refresh_official_ai_readiness,
    reset_official_ai_readiness_cache_for_tests,
    shutdown_official_ai_readiness_refresh,
    verify_ai_model_pack,
)
from vision.source_provenance import (
    OFFICIAL_SOURCE_DESCRIPTOR_PATH,
    build_official_source_descriptor,
    canonical_source_descriptor_json,
    verify_official_source_provenance_at_root,
    verify_official_source_provenance,
)
import vision.ai_model_pack as ai_model_pack_module


EXPECTED_OFFICIAL_LFS_OIDS = {
    "CatVTON/SCHP/exp-schp-201908261155-lip.pth": (
        267_449_349,
        "24fa3254ceeb74c8435458994a64b522fb439a3635b7b86ff470457e0413da00",
    ),
    "CatVTON/SCHP/exp-schp-201908301523-atr.pth": (
        267_445_237,
        "e9d7c91ce3b4e7133df56b599fc817b533e3439c5e8d282a59126d2fda339a2a",
    ),
    "CatVTON/mix-48k-1024/attention/model.safetensors": (
        198_303_368,
        "a1fc093f1b6744623079e6f4e7313411f524e388c4b7467df1e0e7f577cba23a",
    ),
    "inpainting/unet/diffusion_pytorch_model.bin": (
        3_438_412_325,
        "af4e22eecaca7a1c5dd849f51924effdccc1f765141e898c323b775cc80a43b3",
    ),
    "vae/diffusion_pytorch_model.safetensors": (
        334_643_276,
        "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815",
    ),
}


def write_pack(tmp_path, *, extra=False, corrupt=False, revision="9f415fa"):
    model = tmp_path / "CatVTON" / "attention.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"official-weight")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = {"schemaVersion": "vem-catvton-model-pack/v1", "upstream": {"repository": "zhengchong/CatVTON", "revision": revision}, "files": [{"path": "CatVTON/attention.safetensors", "byteSize": model.stat().st_size, "sha256": digest}]}
    (tmp_path / "ai-model-manifest.json").write_text(canonical_ai_model_manifest_json(manifest), "utf-8")
    if corrupt:
        model.write_bytes(b"tampered")
    if extra:
        (tmp_path / "snapshot.bin").write_bytes(b"not-allowed")


def test_exact_model_allowlist_is_deterministic_and_fake_never_claims_official_ready(tmp_path):
    write_pack(tmp_path)
    with pytest.raises(AiModelPackError, match="ai_model_manifest_descriptor_mismatch"):
        verify_ai_model_pack(tmp_path)
    assert official_ai_readiness(tmp_path) is False


def test_official_descriptor_is_canonical_multi_upstream_and_exact_eight_files():
    descriptor = load_official_ai_model_pack_descriptor()

    assert descriptor["schemaVersion"] == "vem-official-ai-model-pack-descriptor/v2"
    assert descriptor["catvtonSourceRevision"] == "3b795364a4d2f3b5adb365f39cdea376d20bc53c"
    assert {upstream["id"] for upstream in descriptor["upstreams"]} == {"catvton", "inpainting", "vae"}
    assert len(descriptor["files"]) == 8
    assert sum(file["byteSize"] for file in descriptor["files"]) == 4_506_255_163
    assert [file["path"] for file in descriptor["files"]] == sorted(file["path"] for file in descriptor["files"])
    assert canonical_ai_model_manifest_json(descriptor) == (
        __import__("pathlib").Path(__file__).parents[1] / "official-ai-model-pack-descriptor.json"
    ).read_text("utf-8").rstrip("\n")
    by_path = {file["path"]: file for file in descriptor["files"]}
    for path, (size, oid) in EXPECTED_OFFICIAL_LFS_OIDS.items():
        assert by_path[path]["byteSize"] == size
        assert by_path[path]["sha256"] == oid


def test_mini_descriptor_verifies_generic_pack_but_never_official_ready(tmp_path):
    model = tmp_path / "mini" / "a.bin"
    model.parent.mkdir()
    model.write_bytes(b"mini-model")
    descriptor = {
        "schemaVersion": "vem-official-ai-model-pack-descriptor/v2",
        "catvtonSourceRevision": "test-source",
        "totalByteSize": model.stat().st_size,
        "upstreams": [{"id": "mini", "repository": "example/mini", "revision": "abc"}],
        "files": [
            {
                "path": "mini/a.bin",
                "upstreamPath": "a.bin",
                "upstream": "mini",
                "role": "mini_weight",
                "format": "bin",
                "byteSize": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "ai-model-manifest.json").write_text(canonical_ai_model_manifest_json(descriptor), "utf-8")

    pack = verify_ai_model_pack(tmp_path, descriptor=descriptor)

    assert pack.files[0]["role"] == "mini_weight"
    assert official_ai_readiness(tmp_path) is False


def test_official_readiness_runs_worker_probe_once_and_caches_by_pack_identity(tmp_path, monkeypatch):
    model = tmp_path / "mini" / "a.bin"
    model.parent.mkdir()
    model.write_bytes(b"mini-model")
    descriptor = {
        "schemaVersion": "vem-official-ai-model-pack-descriptor/v2",
        "catvtonSourceRevision": "test-source",
        "totalByteSize": model.stat().st_size,
        "upstreams": [{"id": "mini", "repository": "example/mini", "revision": "abc"}],
        "files": [
            {
                "path": "mini/a.bin",
                "upstreamPath": "a.bin",
                "upstream": "mini",
                "role": "mini_weight",
                "format": "bin",
                "byteSize": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "ai-model-manifest.json").write_text(canonical_ai_model_manifest_json(descriptor), "utf-8")
    calls = []

    monkeypatch.setattr(ai_model_pack_module, "load_official_ai_model_pack_descriptor", lambda: descriptor)
    monkeypatch.setattr("vision.ai_attempt_process.probe_ai_attempt_worker", lambda pack: calls.append(pack))
    reset_official_ai_readiness_cache_for_tests()

    assert official_ai_readiness(tmp_path) is False
    asyncio.run(refresh_official_ai_readiness(tmp_path))
    assert official_ai_readiness(tmp_path) is True
    assert official_ai_readiness(tmp_path) is True
    assert calls == [tmp_path.resolve()]


def test_official_readiness_cache_misses_when_weight_identity_changes(tmp_path, monkeypatch):
    model = tmp_path / "mini" / "a.bin"
    model.parent.mkdir()
    model.write_bytes(b"mini-model")
    descriptor = {
        "schemaVersion": "vem-official-ai-model-pack-descriptor/v2",
        "catvtonSourceRevision": "test-source",
        "totalByteSize": model.stat().st_size,
        "upstreams": [{"id": "mini", "repository": "example/mini", "revision": "abc"}],
        "files": [
            {
                "path": "mini/a.bin",
                "upstreamPath": "a.bin",
                "upstream": "mini",
                "role": "mini_weight",
                "format": "bin",
                "byteSize": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "ai-model-manifest.json").write_text(canonical_ai_model_manifest_json(descriptor), "utf-8")
    original_stat = model.stat()
    calls = []

    monkeypatch.setattr(ai_model_pack_module, "load_official_ai_model_pack_descriptor", lambda: descriptor)
    monkeypatch.setattr("vision.ai_attempt_process.probe_ai_attempt_worker", lambda pack: calls.append(pack))
    reset_official_ai_readiness_cache_for_tests()

    asyncio.run(refresh_official_ai_readiness(tmp_path))
    assert official_ai_readiness(tmp_path) is True
    model.write_bytes(b"tampered!!")
    os.utime(model, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    asyncio.run(refresh_official_ai_readiness(tmp_path))
    assert official_ai_readiness(tmp_path) is False
    assert calls == [tmp_path.resolve()]


@pytest.mark.parametrize(
    ("failure", "expected_diagnostic"),
    [
        ("missing", "model_pack_missing"),
        ("incomplete", "model_pack_invalid"),
        ("digest", "model_pack_invalid"),
        ("corrupt", "model_pack_invalid"),
        ("probe", "worker_unavailable"),
    ],
)
def test_readiness_failures_expose_only_stable_diagnostics(
    tmp_path, monkeypatch, failure, expected_diagnostic
):
    pack = tmp_path / "pack"
    model = pack / "mini" / "a.bin"
    expected_bytes = b"mini-model"
    descriptor = {
        "schemaVersion": "vem-official-ai-model-pack-descriptor/v2",
        "catvtonSourceRevision": "test-source",
        "totalByteSize": len(expected_bytes),
        "upstreams": [{"id": "mini", "repository": "example/mini", "revision": "abc"}],
        "files": [{
            "path": "mini/a.bin",
            "upstreamPath": "a.bin",
            "upstream": "mini",
            "role": "mini_weight",
            "format": "bin",
            "byteSize": len(expected_bytes),
            "sha256": hashlib.sha256(expected_bytes).hexdigest(),
        }],
    }
    if failure != "missing":
        model.parent.mkdir(parents=True)
        if failure != "incomplete":
            model.write_bytes(b"bad-digest" if failure == "digest" else expected_bytes)
        (pack / "ai-model-manifest.json").write_text(
            "{corrupt"
            if failure == "corrupt"
            else canonical_ai_model_manifest_json(descriptor),
            "utf-8",
        )

    monkeypatch.setattr(
        ai_model_pack_module,
        "load_official_ai_model_pack_descriptor",
        lambda: descriptor,
    )
    if failure == "probe":
        monkeypatch.setattr(
            "vision.ai_attempt_process.probe_ai_attempt_worker",
            lambda _pack: (_ for _ in ()).throw(RuntimeError("C:\\private\\worker.exe")),
        )
    else:
        monkeypatch.setattr("vision.ai_attempt_process.probe_ai_attempt_worker", lambda _pack: None)
    reset_official_ai_readiness_cache_for_tests()

    snapshot = asyncio.run(refresh_official_ai_readiness(pack))

    assert snapshot.ready is False
    assert snapshot.diagnostic == expected_diagnostic
    assert str(tmp_path) not in snapshot.diagnostic


def test_official_readiness_refresh_runs_heavy_probe_off_loop_and_cache_read_is_stat_only(tmp_path, monkeypatch):
    model = tmp_path / "mini" / "a.bin"
    model.parent.mkdir()
    model.write_bytes(b"mini-model")
    descriptor = {
        "schemaVersion": "vem-official-ai-model-pack-descriptor/v2",
        "catvtonSourceRevision": "test-source",
        "totalByteSize": model.stat().st_size,
        "upstreams": [{"id": "mini", "repository": "example/mini", "revision": "abc"}],
        "files": [
            {
                "path": "mini/a.bin",
                "upstreamPath": "a.bin",
                "upstream": "mini",
                "role": "mini_weight",
                "format": "bin",
                "byteSize": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "ai-model-manifest.json").write_text(canonical_ai_model_manifest_json(descriptor), "utf-8")
    calls = []

    def slow_probe(pack):
        calls.append(pack)
        time.sleep(0.1)

    monkeypatch.setattr(ai_model_pack_module, "load_official_ai_model_pack_descriptor", lambda: descriptor)
    monkeypatch.setattr("vision.ai_attempt_process.probe_ai_attempt_worker", slow_probe)
    reset_official_ai_readiness_cache_for_tests()

    async def exercise():
        gaps: list[float] = []

        async def ticker():
            last = time.perf_counter()
            for _ in range(6):
                await asyncio.sleep(0.02)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        await asyncio.gather(refresh_official_ai_readiness(tmp_path), ticker())
        assert max(gaps) < 0.08
        assert official_ai_readiness(tmp_path) is True

    asyncio.run(exercise())
    assert calls == [tmp_path.resolve()]


def test_ready_pack_hot_identity_change_fails_immediately_and_refreshes_once_off_loop(tmp_path, monkeypatch):
    model = tmp_path / "mini" / "a.bin"
    model.parent.mkdir()
    model.write_bytes(b"mini-model")
    descriptor = {
        "schemaVersion": "vem-official-ai-model-pack-descriptor/v2",
        "catvtonSourceRevision": "test-source",
        "totalByteSize": model.stat().st_size,
        "upstreams": [{"id": "mini", "repository": "example/mini", "revision": "abc"}],
        "files": [{
            "path": "mini/a.bin", "upstreamPath": "a.bin", "upstream": "mini",
            "role": "mini_weight", "format": "bin", "byteSize": model.stat().st_size,
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        }],
    }
    (tmp_path / "ai-model-manifest.json").write_text(canonical_ai_model_manifest_json(descriptor), "utf-8")
    change_time = {"value": 1}
    monkeypatch.setattr(
        ai_model_pack_module,
        "_WINDOWS_CHANGE_TIME",
        lambda _path: change_time["value"],
    )
    monkeypatch.setattr(ai_model_pack_module, "load_official_ai_model_pack_descriptor", lambda: descriptor)
    monkeypatch.setattr("vision.ai_attempt_process.probe_ai_attempt_worker", lambda _pack: None)
    reset_official_ai_readiness_cache_for_tests()
    asyncio.run(refresh_official_ai_readiness(tmp_path))
    assert official_ai_readiness(tmp_path) is True

    original_stat = model.stat()
    model.write_bytes(b"evil-model")
    os.utime(model, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    change_time["value"] = 2
    original_compute = ai_model_pack_module._compute_official_ai_readiness_snapshot
    refresh_calls = []

    def slow_compute(root):
        refresh_calls.append(root)
        time.sleep(0.1)
        return original_compute(root)

    monkeypatch.setattr(ai_model_pack_module, "_compute_official_ai_readiness_snapshot", slow_compute)

    async def exercise():
        assert official_ai_readiness(tmp_path) is False
        assert official_ai_readiness(tmp_path) is False
        gaps = []

        async def ticker():
            last = time.perf_counter()
            for _ in range(6):
                await asyncio.sleep(0.02)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        await asyncio.gather(ticker(), shutdown_official_ai_readiness_refresh())
        assert max(gaps) < 0.08
        assert official_ai_readiness(tmp_path) is False

    asyncio.run(exercise())
    assert refresh_calls == [str(tmp_path.resolve())]


def test_windows_quick_identity_uses_change_time_when_size_and_times_are_restored(
    monkeypatch,
):
    """NTFS ChangeTime fences an in-place overwrite hidden by SetFileTime."""
    change_time = {"value": 100}

    class WindowsFile:
        def resolve(self):
            return "C:/pack/weight.bin"

        def stat(self):
            return SimpleNamespace(
                st_size=4096,
                st_mtime_ns=1,
                st_ctime_ns=1,
                st_ino=7,
                st_dev=9,
            )

    monkeypatch.setattr(
        ai_model_pack_module,
        "_WINDOWS_CHANGE_TIME",
        lambda _path: change_time["value"],
        raising=False,
    )
    before = ai_model_pack_module._quick_file_identity(WindowsFile())
    change_time["value"] = 200
    after = ai_model_pack_module._quick_file_identity(WindowsFile())

    assert before != after


def test_windows_change_time_api_preserves_wide_handles_and_requires_close():
    calls = []

    class Kernel32:
        def CreateFileW(self, *arguments):
            calls.append(("open", arguments))
            return 1 << 40

        def GetFileInformationByHandleEx(self, handle, kind, result, size):
            calls.append(("facts", handle, kind, size))
            result._obj.ChangeTime = 123456789
            return 1

        def CloseHandle(self, handle):
            calls.append(("close", handle))
            return 1

    api = ai_model_pack_module._WindowsFileIdentityApi(Kernel32())

    assert api.change_time(Path("C:/pack/weight.bin")) == 123456789
    assert calls[1][1] == 1 << 40
    assert calls[2] == ("close", 1 << 40)


def test_ready_pack_selection_change_atomically_replaces_snapshot(tmp_path, monkeypatch):
    roots = [tmp_path / "one", tmp_path / "two"]
    descriptor = None
    for root in roots:
        model = root / "mini" / "a.bin"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"mini-model")
        descriptor = {
            "schemaVersion": "vem-official-ai-model-pack-descriptor/v2",
            "catvtonSourceRevision": "test-source",
            "totalByteSize": model.stat().st_size,
            "upstreams": [{"id": "mini", "repository": "example/mini", "revision": "abc"}],
            "files": [{
                "path": "mini/a.bin", "upstreamPath": "a.bin", "upstream": "mini",
                "role": "mini_weight", "format": "bin", "byteSize": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }],
        }
        (root / "ai-model-manifest.json").write_text(canonical_ai_model_manifest_json(descriptor), "utf-8")
    monkeypatch.setattr(ai_model_pack_module, "load_official_ai_model_pack_descriptor", lambda: descriptor)
    monkeypatch.setattr("vision.ai_attempt_process.probe_ai_attempt_worker", lambda _pack: None)
    reset_official_ai_readiness_cache_for_tests()

    async def exercise():
        await refresh_official_ai_readiness(roots[0])
        assert official_ai_readiness(roots[0]) is True
        assert official_ai_readiness(roots[1]) is False
        assert official_ai_readiness_snapshot(roots[1]).root == str(
            roots[1].resolve()
        )
        assert official_ai_readiness_snapshot(roots[1]).ready is False
        await shutdown_official_ai_readiness_refresh()
        assert official_ai_readiness(roots[1]) is True

    asyncio.run(exercise())


def test_unset_root_generation_fences_out_an_older_background_refresh(
    tmp_path, monkeypatch
):
    model = tmp_path / "mini" / "a.bin"
    model.parent.mkdir()
    model.write_bytes(b"mini-model")
    descriptor = {
        "schemaVersion": "vem-official-ai-model-pack-descriptor/v2",
        "catvtonSourceRevision": "test-source",
        "totalByteSize": model.stat().st_size,
        "upstreams": [
            {"id": "mini", "repository": "example/mini", "revision": "abc"}
        ],
        "files": [
            {
                "path": "mini/a.bin",
                "upstreamPath": "a.bin",
                "upstream": "mini",
                "role": "mini_weight",
                "format": "bin",
                "byteSize": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "ai-model-manifest.json").write_text(
        canonical_ai_model_manifest_json(descriptor), "utf-8"
    )
    monkeypatch.setattr(
        ai_model_pack_module,
        "load_official_ai_model_pack_descriptor",
        lambda: descriptor,
    )
    monkeypatch.setattr(
        "vision.ai_attempt_process.probe_ai_attempt_worker", lambda _pack: None
    )
    reset_official_ai_readiness_cache_for_tests()
    asyncio.run(refresh_official_ai_readiness(tmp_path))

    original_stat = model.stat()
    model.write_bytes(b"evil-model")
    os.utime(model, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    original_compute = ai_model_pack_module._compute_official_ai_readiness_snapshot
    started = threading.Event()
    release = threading.Event()

    def slow_compute(root):
        started.set()
        assert release.wait(timeout=2)
        return original_compute(root)

    monkeypatch.setattr(
        ai_model_pack_module, "_compute_official_ai_readiness_snapshot", slow_compute
    )

    async def exercise():
        assert official_ai_readiness_snapshot(tmp_path).ready is False
        assert await asyncio.to_thread(started.wait, 2)
        unset = official_ai_readiness_snapshot(None)
        assert (unset.ready, unset.diagnostic) == (False, "model_pack_missing")
        release.set()
        await shutdown_official_ai_readiness_refresh()
        final = official_ai_readiness_snapshot(None)
        assert (final.root, final.ready, final.diagnostic) == (
            None,
            False,
            "model_pack_missing",
        )

    asyncio.run(exercise())


def test_official_source_descriptor_binds_tracked_worker_and_vendor_sources():
    descriptor = json.loads(OFFICIAL_SOURCE_DESCRIPTOR_PATH.read_text("utf-8"))

    assert descriptor["schemaVersion"] == "vem-official-ai-source-descriptor/v1"
    assert descriptor["catvtonSourceRevision"] == "3b795364a4d2f3b5adb365f39cdea376d20bc53c"
    paths = {source["path"] for source in descriptor["sources"]}
    assert "vision/ai_attempt_worker.py" in paths
    assert "vision/ai_attempt_process.py" in paths
    assert "vision/process_supervisor.py" in paths
    assert "vision/ai_runtime_descriptor.py" in paths
    assert "vision/catvton_preprocess.py" in paths
    assert "vision/catvton_pose_masks.py" in paths
    assert "vision/vendor/catvton/model/pipeline.py" in paths
    assert canonical_source_descriptor_json(build_official_source_descriptor()) == OFFICIAL_SOURCE_DESCRIPTOR_PATH.read_text("utf-8").rstrip("\n")
    assert verify_official_source_provenance() is True


def test_official_provenance_owns_reference_lineage_and_exclusions_without_runtime_dependency():
    provenance = (
        Path(__file__).parents[1] / "vision/vendor/catvton/PROVENANCE.md"
    ).read_text("utf-8")
    fixture_provenance = (
        Path(__file__).parents[1] / "fixtures/recorded-video/README.md"
    ).read_text("utf-8")

    for fact in (
        "c0a76e499a620a253b7ac0a6a07f8ee0754c2c10",
        "3b795364a4d2f3b5adb365f39cdea376d20bc53c",
        "487ac2261ae102a80f8a2142d2a369af7776869cc3e91d9b6729a122bd49af03",
        "4fae4fa44ee8ab75c94869680deea944de5e5c03a4b56689e8c23422c3cfc18d",
        "person-woman-front.png",
        "coral-tee",
        "cream-sweater",
        "midnight-jacket",
        "ocean-polo",
        "652ab2a22dd83ec45e81e283af5310ec.jpg",
        "c196741201df156a8a2ff68fabd2d034.jpg",
    ):
        assert fact in provenance
    assert (
        "not a build, test, packaging, runtime, download, fallback, or deployment"
        in provenance
    )
    assert "c0a76e499a620a253b7ac0a6a07f8ee0754c2c10" in fixture_provenance
    assert (
        "659f08c709c8d526552713741f5e2cfe3fa819a34a63a34a8372a3404890952c"
        in fixture_provenance
    )


def test_official_source_provenance_verifies_real_frozen_like_layout_and_detects_tamper(tmp_path):
    descriptor = json.loads(OFFICIAL_SOURCE_DESCRIPTOR_PATH.read_text("utf-8"))
    for source in descriptor["sources"]:
        src = Path(__file__).parents[1] / source["path"]
        dst = tmp_path / source["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    (tmp_path / "official-ai-source-descriptor.json").write_text(
        OFFICIAL_SOURCE_DESCRIPTOR_PATH.read_text("utf-8"),
        "utf-8",
    )

    assert verify_official_source_provenance_at_root(tmp_path) is True

    tampered = tmp_path / "vision" / "process_supervisor.py"
    tampered.write_text(tampered.read_text("utf-8") + "\n# tamper\n", "utf-8")
    assert verify_official_source_provenance_at_root(tmp_path) is False
    tampered.write_bytes((Path(__file__).parents[1] / "vision" / "process_supervisor.py").read_bytes())
    (tmp_path / "vision" / "ai_attempt_worker.py").unlink()
    assert verify_official_source_provenance_at_root(tmp_path) is False


@pytest.mark.parametrize("options", [{"extra": True}, {"corrupt": True}])
def test_tampered_or_extra_model_pack_is_not_ready(tmp_path, options):
    write_pack(tmp_path, **options)
    with pytest.raises(AiModelPackError):
        verify_ai_model_pack(tmp_path)
    assert official_ai_readiness(tmp_path) is False


@pytest.mark.parametrize(
    "mutate, expected_error",
    [
        (lambda root: (root / "ai-model-manifest.json").unlink(), "ai_model_manifest_missing_or_invalid"),
        (lambda root: (root / "CatVTON" / "attention.safetensors").unlink(), "ai_model_manifest_descriptor_mismatch"),
        (lambda root: (root / "ai-model-manifest.json").write_text('{"schemaVersion":"vem-catvton-model-pack/v1","schemaVersion":"dup","upstream":{"repository":"zhengchong/CatVTON","revision":"9f415fa"},"files":[]}', "utf-8"), "ai_model_manifest_duplicate_key"),
    ],
)
def test_missing_files_and_duplicate_manifest_keys_are_not_ready(tmp_path, mutate, expected_error):
    write_pack(tmp_path)
    mutate(tmp_path)

    with pytest.raises(AiModelPackError, match=expected_error):
        verify_ai_model_pack(tmp_path)
    assert official_ai_readiness(tmp_path) is False


def test_wrong_official_revision_verifies_but_is_not_runtime_ready(tmp_path):
    write_pack(tmp_path, revision="different-revision")

    with pytest.raises(AiModelPackError, match="ai_model_manifest_descriptor_mismatch"):
        verify_ai_model_pack(tmp_path)
    assert official_ai_readiness(tmp_path) is False


def test_manifest_generator_is_deterministic_and_rejects_escaping_or_duplicate_paths(tmp_path):
    (tmp_path / "CatVTON").mkdir()
    first = tmp_path / "CatVTON" / "attention.safetensors"
    second = tmp_path / "CatVTON" / "base-model.pin"
    first.write_bytes(b"official-weight")
    second.write_text("base-model=runwayml/stable-diffusion-inpainting\nrevision=pinned\n", "utf-8")

    manifest_a = create_ai_model_manifest(
        tmp_path,
        repository="zhengchong/CatVTON",
        revision="9f415fa",
        paths=["CatVTON/base-model.pin", "CatVTON/attention.safetensors"],
    )
    manifest_b = create_ai_model_manifest(
        tmp_path,
        repository="zhengchong/CatVTON",
        revision="9f415fa",
        paths=["CatVTON/base-model.pin", "CatVTON/attention.safetensors"],
    )

    assert canonical_ai_model_manifest_json(manifest_a) == canonical_ai_model_manifest_json(manifest_b)
    assert [item["path"] for item in manifest_a["files"]] == [
        "CatVTON/attention.safetensors",
        "CatVTON/base-model.pin",
    ]

    with pytest.raises(AiModelPackError, match="ai_model_manifest_path"):
        create_ai_model_manifest(tmp_path, repository="zhengchong/CatVTON", revision="9f415fa", paths=["../escape"])
    with pytest.raises(AiModelPackError, match="ai_model_manifest_path"):
        create_ai_model_manifest(tmp_path, repository="zhengchong/CatVTON", revision="9f415fa", paths=["CatVTON/base-model.pin", "CatVTON/base-model.pin"])
