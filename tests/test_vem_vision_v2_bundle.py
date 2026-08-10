import json
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from contracts.vem_vision_v2.python.vision_v2_models import parse_message
from scripts.check_vem_vision_v2_bundle import check_bundle


BUNDLE_ROOT = Path(__file__).parents[1] / "contracts" / "vem_vision_v2"


def test_generated_v2_boundary_accepts_shared_fast_fixtures():
    fixtures = json.loads((BUNDLE_ROOT / "fixtures" / "valid.json").read_text("utf-8"))
    assert [parse_message(fixture).type for fixture in fixtures] == [
        "vision.hello",
        "vision.ready",
        "vision.try_on.attempt.start",
        "vision.try_on.attempt.start",
        "vision.try_on.attempt.start",
        "vision.try_on.attempt.accepted",
        "vision.try_on.attempt.progress",
        "vision.try_on.attempt.completed",
        "vision.try_on.attempt.failed",
        "vision.try_on.attempt.capture",
        "vision.try_on.attempt.cancel",
        "vision.try_on.attempt.acquiring",
        "vision.try_on.attempt.generating",
        "vision.try_on.attempt.canceled",
        "vision.ready",
    ]


def test_generated_v2_boundary_rejects_shared_negative_fixtures():
    fixtures = json.loads((BUNDLE_ROOT / "fixtures" / "invalid.json").read_text("utf-8"))
    for fixture in fixtures:
        with pytest.raises((ValidationError, ValueError)):
            parse_message(fixture["message"])


def test_standalone_json_schema_rejects_control_after_token():
    schema = json.loads((BUNDLE_ROOT / "vision-v2.schema.json").read_text("utf-8"))
    start = next(
        branch
        for branch in schema["oneOf"]
        if branch["properties"]["type"]["const"] == "vision.try_on.attempt.start"
    )
    reference_schema = start["properties"]["payload"]["properties"]["garment"][
        "properties"
    ]["reference"]
    assert reference_schema["pattern"].endswith(r"(?![\s\S])")
    validator = Draft202012Validator(schema)
    fixtures = json.loads((BUNDLE_ROOT / "fixtures" / "invalid.json").read_text("utf-8"))
    for fixture in fixtures:
        if fixture["name"].startswith("rejects-token-url-trailing-"):
            assert list(validator.iter_errors(fixture["message"])), fixture["name"]


def test_json_integer_fields_normalize_integral_floats_but_reject_non_integers():
    start = json.loads((BUNDLE_ROOT / "fixtures" / "valid.json").read_text("utf-8"))[2]

    for value in (1, 1.0):
        accepted = json.loads(json.dumps(start))
        accepted["payload"]["garment"]["byteSize"] = value
        parsed = parse_message(accepted)
        assert parsed.payload.garment.byteSize == 1

    for value in (True, "1", 1.5):
        invalid = json.loads(json.dumps(start))
        invalid["payload"]["garment"]["byteSize"] = value
        with pytest.raises((ValidationError, ValueError)):
            parse_message(invalid)


def test_generated_python_matches_loopback_port_boundaries():
    start = json.loads((BUNDLE_ROOT / "fixtures" / "valid.json").read_text("utf-8"))[2]

    accepted = json.loads(json.dumps(start))
    accepted["payload"]["garment"][
        "reference"
    ] = "http://127.0.0.1:65535/garment?token=opaque"
    assert parse_message(accepted).payload.garment.reference.endswith("token=opaque")

    for port in (0, 65536, 99999):
        invalid = json.loads(json.dumps(start))
        invalid["payload"]["garment"][
            "reference"
        ] = f"http://127.0.0.1:{port}/garment?token=opaque"
        with pytest.raises(ValueError):
            parse_message(invalid)


def test_schema_added_required_field_generates_a_typed_payload_without_a_python_map(
    tmp_path: Path,
):
    bundle = tmp_path / "vem_vision_v2"
    shutil.copytree(BUNDLE_ROOT, bundle)
    schema_path = bundle / "vision-v2.schema.json"
    schema = json.loads(schema_path.read_text("utf-8"))
    hello = next(
        branch
        for branch in schema["oneOf"]
        if branch["properties"]["type"]["const"] == "vision.hello"
    )
    hello_payload = hello["properties"]["payload"]
    hello_payload["properties"]["deploymentId"] = {
        "minLength": 1,
        "type": "string",
    }
    hello_payload["required"].append("deploymentId")
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    module_spec = importlib.util.spec_from_file_location(
        "mutated_vision_v2_models", bundle / "python" / "vision_v2_models.py"
    )
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    hello_fixture = json.loads(
        (bundle / "fixtures" / "valid.json").read_text("utf-8")
    )[0]
    with pytest.raises(ValueError):
        module.parse_message(hello_fixture)
    hello_fixture["payload"]["deploymentId"] = "site-a"
    parsed = module.parse_message(hello_fixture)
    assert type(parsed).__name__ == "VisionHelloEnvelope"
    assert type(parsed.payload).__name__ == "VisionHelloEnvelopePayload"
    assert module.VisionHelloEnvelope is type(parsed)
    assert module.VisionHelloEnvelopePayload is type(parsed.payload)
    assert parsed.payload.deploymentId == "site-a"


def test_vendored_bundle_has_an_independently_checkable_manifest():
    result = subprocess.run(
        [sys.executable, "scripts/check_vem_vision_v2_bundle.py"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_vendored_bundle_check_rejects_a_drifted_generated_file(tmp_path: Path):
    drifted_bundle = tmp_path / "vem_vision_v2"
    shutil.copytree(BUNDLE_ROOT, drifted_bundle)
    (drifted_bundle / "fixtures" / "valid.json").write_text("[]\n", encoding="utf-8")

    assert "digest mismatch: fixtures/valid.json" in check_bundle(drifted_bundle)


def test_vendored_bundle_check_rejects_manifest_metadata_and_path_bypasses(
    tmp_path: Path,
):
    drifted_bundle = tmp_path / "vem_vision_v2"
    shutil.copytree(BUNDLE_ROOT, drifted_bundle)
    manifest_path = drifted_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["protocol"] = "vem.vision.unsupported"
    manifest["files"]["../escape.json"] = "a" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    failures = check_bundle(drifted_bundle)
    assert "unexpected manifest protocol" in failures
    assert "manifest file set does not match the vendored bundle" in failures
    assert "manifest path is not canonical: '../escape.json'" in failures


def test_vendored_bundle_check_rejects_version_digest_and_missing_file_tampering(
    tmp_path: Path,
):
    cases = [
        ("schemaVersion", "unexpected manifest schemaVersion"),
        ("bundleVersion", "unexpected manifest bundleVersion"),
        ("fileDigest", "digest mismatch: fixtures/valid.json"),
        ("missingFile", "missing bundle file: fixtures/valid.json"),
    ]
    for mutation, expected_failure in cases:
        bundle = tmp_path / mutation
        shutil.copytree(BUNDLE_ROOT, bundle)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        if mutation == "schemaVersion":
            manifest["schemaVersion"] = "vem-vision-v2-contract-bundle/v2"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        elif mutation == "bundleVersion":
            manifest["bundleVersion"] = "2"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        elif mutation == "fileDigest":
            manifest["files"]["fixtures/valid.json"] = "f" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        else:
            (bundle / "fixtures" / "valid.json").unlink()

        assert expected_failure in check_bundle(bundle), mutation


@pytest.mark.parametrize("declared_path", ["../escape.json", "/escape.json", "fixtures//valid.json"])
def test_vendored_bundle_check_rejects_all_noncanonical_manifest_paths(
    tmp_path: Path, declared_path: str
):
    bundle = tmp_path / declared_path.replace("/", "_").replace(".", "_")
    shutil.copytree(BUNDLE_ROOT, bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["files"][declared_path] = "a" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert f"manifest path is not canonical: {declared_path!r}" in check_bundle(bundle)


def test_vendored_bundle_check_rejects_an_unmanifested_physical_file(tmp_path: Path):
    bundle = tmp_path / "vem_vision_v2"
    shutil.copytree(BUNDLE_ROOT, bundle)
    (bundle / "unexpected.py").write_text("# tampered\n", encoding="utf-8")

    assert "vendored bundle contains missing or unmanifested files" in check_bundle(bundle)


def test_vendored_bundle_check_rejects_noncanonical_and_duplicate_manifest_json(
    tmp_path: Path,
):
    pretty_bundle = tmp_path / "pretty"
    shutil.copytree(BUNDLE_ROOT, pretty_bundle)
    pretty_manifest = pretty_bundle / "manifest.json"
    pretty_manifest.write_text(
        json.dumps(json.loads(pretty_manifest.read_text("utf-8")), indent=2) + "\n",
        encoding="utf-8",
    )
    assert "manifest must use exact canonical JSON" in check_bundle(pretty_bundle)

    duplicate_bundle = tmp_path / "duplicate"
    shutil.copytree(BUNDLE_ROOT, duplicate_bundle)
    duplicate_manifest = duplicate_bundle / "manifest.json"
    raw = duplicate_manifest.read_text("utf-8")
    duplicate_manifest.write_text(
        raw.replace('"protocol":"vem.vision.v2",', '"protocol":"vem.vision.v2","protocol":"vem.vision.v2",'),
        encoding="utf-8",
    )
    assert "manifest contains duplicate key: protocol" in check_bundle(duplicate_bundle)


@pytest.mark.parametrize(
    "declared_path", [r"fixtures\\valid.json", "C:/escape.json", "//server/share.json"]
)
def test_vendored_bundle_check_rejects_windows_and_unc_manifest_paths(
    tmp_path: Path, declared_path: str
):
    bundle = tmp_path / "windows-path"
    shutil.copytree(BUNDLE_ROOT, bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["files"][declared_path] = "a" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert f"manifest path is not canonical: {declared_path!r}" in check_bundle(bundle)
