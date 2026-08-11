import hashlib
import json
import os

import pytest

from vision.ai_model_pack import (
    AiModelPackError,
    canonical_ai_model_manifest_json,
    create_ai_model_manifest,
    load_official_ai_model_pack_descriptor,
    official_ai_readiness,
    verify_ai_model_pack,
)
from vision.source_provenance import OFFICIAL_SOURCE_DESCRIPTOR_PATH, verify_official_source_provenance
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
    ai_model_pack_module._READINESS_CACHE.clear()

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
    ai_model_pack_module._READINESS_CACHE.clear()

    assert official_ai_readiness(tmp_path) is True
    model.write_bytes(b"tampered!!")
    os.utime(model, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert official_ai_readiness(tmp_path) is False
    assert calls == [tmp_path.resolve()]


def test_official_source_descriptor_binds_tracked_worker_and_vendor_sources():
    descriptor = json.loads(OFFICIAL_SOURCE_DESCRIPTOR_PATH.read_text("utf-8"))

    assert descriptor["schemaVersion"] == "vem-official-ai-source-descriptor/v1"
    assert descriptor["catvtonSourceRevision"] == "3b795364a4d2f3b5adb365f39cdea376d20bc53c"
    paths = {source["path"] for source in descriptor["sources"]}
    assert "vision/ai_attempt_worker.py" in paths
    assert "vision/catvton_preprocess.py" in paths
    assert "vision/catvton_pose_masks.py" in paths
    assert "vision/vendor/catvton/model/pipeline.py" in paths
    assert verify_official_source_provenance() is True


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
