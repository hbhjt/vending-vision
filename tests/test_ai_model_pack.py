import hashlib
import json

import pytest

from vision.ai_model_pack import (
    AiModelPackError,
    canonical_ai_model_manifest_json,
    create_ai_model_manifest,
    official_ai_readiness,
    verify_ai_model_pack,
)


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
    assert verify_ai_model_pack(tmp_path).upstream_repository == "zhengchong/CatVTON"
    assert official_ai_readiness(tmp_path) is True


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
        (lambda root: (root / "CatVTON" / "attention.safetensors").unlink(), "ai_model_pack_digest"),
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

    assert verify_ai_model_pack(tmp_path).upstream_revision == "different-revision"
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
