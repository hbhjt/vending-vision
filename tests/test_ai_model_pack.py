import hashlib
import json

import pytest

from vision.ai_model_pack import AiModelPackError, official_ai_readiness, verify_ai_model_pack


def write_pack(tmp_path, *, extra=False, corrupt=False):
    model = tmp_path / "CatVTON" / "attention.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"official-weight")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = {"schemaVersion": "vem-catvton-model-pack/v1", "upstream": {"repository": "zhengchong/CatVTON", "revision": "9f415fa"}, "files": [{"path": "CatVTON/attention.safetensors", "byteSize": model.stat().st_size, "sha256": digest}]}
    (tmp_path / "ai-model-manifest.json").write_text(json.dumps(manifest), "utf-8")
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
