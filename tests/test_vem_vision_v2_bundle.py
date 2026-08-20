import json
import hashlib
import shutil
from pathlib import Path

import pytest

from contracts.vem_vision_v2.python.vision_v2_models import (
    parse_client_message,
    parse_server_message,
)
from vision.v2_contract_bundle import (
    V2ContractBundleUnavailable,
    load_v2_contract_identity,
    parse_v2_client_message,
    parse_v2_server_message,
)
from vision import v2_contract_bundle


ROOT = Path(__file__).parents[1] / "contracts" / "vem_vision_v2" / "fixtures"
BUNDLE_ROOT = ROOT.parent


@pytest.mark.parametrize(
    ("direction", "parser"),
    [("client", parse_client_message), ("server", parse_server_message)],
)
def test_generated_direction_parser_accepts_only_its_explicit_valid_corpus(direction, parser):
    fixtures = json.loads((ROOT / f"{direction}-valid.json").read_text("utf-8"))
    assert [parser(fixture).type for fixture in fixtures]
    opposite = parse_server_message if direction == "client" else parse_client_message
    for fixture in fixtures:
        with pytest.raises(ValueError):
            opposite(fixture)


@pytest.mark.parametrize(
    ("direction", "parser"),
    [("client", parse_client_message), ("server", parse_server_message)],
)
def test_every_negative_mutates_a_valid_message_and_is_rejected_in_its_direction(direction, parser):
    fixtures = json.loads((ROOT / f"{direction}-invalid.json").read_text("utf-8"))
    for fixture in fixtures:
        assert parser(fixture["base"])
        with pytest.raises(ValueError, match="invalid vem.vision.v2 message"):
            parser(fixture["message"])


def test_runtime_boundary_has_no_generic_message_parser():
    hello = json.loads((ROOT / "client-valid.json").read_text("utf-8"))[0]
    ready = json.loads((ROOT / "server-valid.json").read_text("utf-8"))[0]
    assert parse_v2_client_message(hello).type == "vision.hello"
    assert parse_v2_server_message(ready).type == "vision.ready"


def _try_on_start_fixture():
    valid = json.loads((ROOT / "client-valid.json").read_text("utf-8"))
    return next(item for item in valid if item["type"] == "vision.try_on.attempt.start")


def _try_on_start_payload_schema(schema):
    start = next(
        option
        for option in schema["oneOf"]
        if option["properties"]["type"].get("const")
        == "vision.try_on.attempt.start"
    )
    return start["properties"]["payload"]


def test_try_on_start_contract_has_exact_single_path_shape():
    schema = json.loads(
        (BUNDLE_ROOT / "vision-v2.client.schema.json").read_text("utf-8")
    )
    payload_schema = _try_on_start_payload_schema(schema)

    assert payload_schema["type"] == "object"
    assert payload_schema["additionalProperties"] is False
    assert set(payload_schema["properties"]) == {"attemptId", "variantId", "garment"}
    assert payload_schema["required"] == ["attemptId", "variantId", "garment"]
    assert payload_schema["properties"]["garment"]["additionalProperties"] is False
    assert set(payload_schema["properties"]["garment"]["properties"]) == {
        "assetId",
        "reference",
        "digest",
        "contentType",
        "byteSize",
        "template",
    }

    assert "mode" not in _try_on_start_fixture()["payload"]


@pytest.mark.parametrize("retired_mode", ["fast", "ai", "automatic"])
@pytest.mark.parametrize("parser", [parse_client_message, parse_v2_client_message])
def test_public_try_on_start_parsers_reject_every_mode_value(parser, retired_mode):
    start = _try_on_start_fixture()
    start["payload"]["mode"] = retired_mode

    with pytest.raises(ValueError):
        parser(start)


def test_bundle_identity_rejects_mode_even_after_schema_fixture_and_digest_rewrite(
    tmp_path, monkeypatch
):
    bundle = tmp_path / "vem_vision_v2"
    shutil.copytree(BUNDLE_ROOT, bundle)
    schema_path = bundle / "vision-v2.client.schema.json"
    schema = json.loads(schema_path.read_text("utf-8"))
    payload_schema = _try_on_start_payload_schema(schema)
    payload_schema["properties"]["mode"] = {
        "enum": ["fast", "ai", "automatic"],
        "type": "string",
    }
    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    valid_path = bundle / "fixtures" / "client-valid.json"
    valid = json.loads(valid_path.read_text("utf-8"))
    next(
        item for item in valid if item["type"] == "vision.try_on.attempt.start"
    )["payload"]["mode"] = "automatic"
    valid_path.write_text(
        json.dumps(valid, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    for relative_path in (
        "vision-v2.client.schema.json",
        "fixtures/client-valid.json",
    ):
        manifest["files"][relative_path] = hashlib.sha256(
            (bundle / relative_path).read_bytes()
        ).hexdigest()
    metadata = {
        "schemaVersion": manifest["schemaVersion"],
        "protocol": manifest["protocol"],
        "bundleVersion": manifest["bundleVersion"],
        "files": manifest["files"],
    }
    manifest["bundleDigest"] = hashlib.sha256(
        json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(v2_contract_bundle, "_BUNDLE_ROOT", bundle)

    with pytest.raises(V2ContractBundleUnavailable):
        load_v2_contract_identity()


def test_bundle_identity_rejects_hidden_start_mode_shadow_branch(tmp_path, monkeypatch):
    bundle = tmp_path / "vem_vision_v2"
    shutil.copytree(BUNDLE_ROOT, bundle)
    schema_path = bundle / "vision-v2.client.schema.json"
    schema = json.loads(schema_path.read_text("utf-8"))
    explicit_start = next(
        option
        for option in schema["oneOf"]
        if option["properties"]["type"].get("const")
        == "vision.try_on.attempt.start"
    )
    shadow_start = json.loads(json.dumps(explicit_start))
    shadow_start["properties"]["type"] = {"type": "string"}
    shadow_payload = shadow_start["properties"]["payload"]
    shadow_payload["properties"]["mode"] = {
        "enum": ["fast", "ai", "automatic"],
        "type": "string",
    }
    shadow_payload["required"].append("mode")
    schema["oneOf"].append(shadow_start)
    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["files"]["vision-v2.client.schema.json"] = hashlib.sha256(
        schema_path.read_bytes()
    ).hexdigest()
    metadata = {
        "schemaVersion": manifest["schemaVersion"],
        "protocol": manifest["protocol"],
        "bundleVersion": manifest["bundleVersion"],
        "files": manifest["files"],
    }
    manifest["bundleDigest"] = hashlib.sha256(
        json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(v2_contract_bundle, "_BUNDLE_ROOT", bundle)

    with pytest.raises(V2ContractBundleUnavailable):
        load_v2_contract_identity()
