from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from scripts.admit_trusted_precutover_inputs import AdmissionError, admit_inputs
from scripts.workflow_yaml import load_workflow_yaml
from scripts.trusted_precutover_proof import HANDOFF_FILES


ROOT = Path(__file__).parents[1]
CALLER = ROOT / ".github/workflows/trusted-precutover-caller.yml"
PROOF_SHA = "341f6c9ca083d584fbab072345e96ce3ed062edc"
COMPANION_BUILDER_SHA = "852ca005c5ce0fcdf7799f38d2335ae94c49be3c"
CANDIDATE_INPUTS = {
    f"{name}_{field}"
    for name in (
        "candidate_archive",
        "candidate_manifest",
        "candidate_attestation",
        "candidate_evidence",
    )
    for field in ("url", "sha256", "bytes")
}
MODEL_FINAL_INPUTS = {"model_pack_sha256", "model_pack_bytes"}
MODEL_PART_INPUTS = {
    f"model_pack_part_{index:02d}_{field}"
    for index in range(1, 4)
    for field in ("url", "sha256", "bytes")
}
INPUTS = CANDIDATE_INPUTS | MODEL_FINAL_INPUTS | MODEL_PART_INPUTS | {"model_pack_url"}
PROOF_KEYS = (
    "candidate_archive_url",
    "candidate_archive_sha256",
    "candidate_archive_bytes",
    "candidate_manifest_url",
    "candidate_manifest_sha256",
    "candidate_manifest_bytes",
    "candidate_attestation_url",
    "candidate_attestation_sha256",
    "candidate_attestation_bytes",
    "candidate_evidence_url",
    "candidate_evidence_sha256",
    "candidate_evidence_bytes",
    "model_pack_url",
    "model_pack_sha256",
    "model_pack_bytes",
    "model_pack_part_01_url",
    "model_pack_part_01_sha256",
    "model_pack_part_01_bytes",
    "model_pack_part_02_url",
    "model_pack_part_02_sha256",
    "model_pack_part_02_bytes",
    "model_pack_part_03_url",
    "model_pack_part_03_sha256",
    "model_pack_part_03_bytes",
)
COMPANION_KEYS = (
    "artifact_name",
    "archive_file",
    "archive_sha256",
    "descriptor_file",
    "descriptor_sha256",
    "attestation_bundle_file",
    "attestation_bundle_sha256",
)


def _dispatch_inputs_fixture(*, multipart: bool) -> dict[str, object]:
    """Match GitHub's omitted-key JSON shape for empty optional dispatch inputs."""
    value: dict[str, object] = {}
    for key in PROOF_KEYS:
        if multipart and key == "model_pack_url":
            continue
        if not multipart and key.startswith("model_pack_part_"):
            continue
        if key.endswith("_bytes"):
            value[key] = 1
        elif key.endswith("_sha256"):
            value[key] = "a" * 64
        else:
            value[key] = f"https://example.invalid/{key}"
    return value


def _companion_fixture() -> dict[str, str]:
    return {
        key: ("b" * 64 if key.endswith("_sha256") else f"fixed-{key}")
        for key in COMPANION_KEYS
    }


def _compact(value: dict[str, object]) -> str:
    return json.dumps(value, separators=(",", ":"))


def _explicit_proof_inputs_expression() -> str:
    fields = []
    arguments = []
    for index, key in enumerate(PROOF_KEYS):
        fields.append(f'"{key}":{{{index}}}')
        optional = key == "model_pack_url" or key.startswith("model_pack_part_")
        if key.endswith("_bytes"):
            source = f"inputs.{key} || '0'" if optional else f"inputs.{key}"
            arguments.append(f"toJSON(fromJSON({source}))")
        else:
            source = f"inputs.{key} || ''" if optional else f"inputs.{key}"
            arguments.append(f"toJSON({source})")
    return "${{ format('{{" + ",".join(fields) + "}}', " + ", ".join(arguments) + ") }}"


def _evaluate_caller_proof_inputs(
    dispatch_inputs: dict[str, str], expression: str
) -> str:
    """Evaluate the caller's format/toJSON envelope for dispatch string values."""
    proof: dict[str, object] = {}
    for key in PROOF_KEYS:
        optional = key == "model_pack_url" or key.startswith("model_pack_part_")
        fallback = "0" if key.endswith("_bytes") else ""
        value = dispatch_inputs[key] or (fallback if optional else dispatch_inputs[key])
        parses_bytes = f"toJSON(fromJSON(inputs.{key}" in expression
        proof[key] = json.loads(value) if key.endswith("_bytes") and parses_bytes else value
    return _compact(proof)


def _string_dispatch_inputs(*, multipart: bool) -> dict[str, str]:
    return {
        key: (
            ""
            if multipart and key == "model_pack_url"
            else ""
            if not multipart and key.startswith("model_pack_part_")
            else "1"
            if key.endswith("_bytes")
            else "a" * 64
            if key.endswith("_sha256")
            else f"https://example.invalid/{key}"
        )
        for key in PROOF_KEYS
    }


@pytest.mark.parametrize("multipart", (False, True), ids=("whole", "multipart"))
def test_caller_converts_all_dispatch_byte_strings_to_json_numbers_before_admission(
    multipart: bool,
):
    dispatch_inputs = _string_dispatch_inputs(multipart=multipart)
    expression = load_workflow_yaml(CALLER.read_text("utf-8"))["jobs"][
        "trusted_proof"
    ]["with"]["proof_inputs"]

    proof, _ = admit_inputs(
        _evaluate_caller_proof_inputs(dispatch_inputs, expression),
        _compact(_companion_fixture()),
    )

    admitted = json.loads(proof)
    assert list(admitted) == list(PROOF_KEYS)
    assert len(admitted) == 24
    assert all(type(admitted[key]) is int for key in PROOF_KEYS if key.endswith("_bytes"))


@pytest.mark.parametrize(
    ("value", "expression_failure"),
    (
        ("", True),
        ("1.5", False),
        ("true", False),
        (str(1 << 53), False),
        ("0", False),
        ("not-a-number", True),
        ("${UNTRUSTED_BYTES}", True),
    ),
    ids=(
        "empty",
        "float",
        "bool",
        "unsafe-integer",
        "zero",
        "non-json",
        "placeholder",
    ),
)
def test_caller_fails_closed_for_invalid_required_dispatch_byte_strings(
    value: str, expression_failure: bool
):
    dispatch_inputs = _string_dispatch_inputs(multipart=False)
    dispatch_inputs["candidate_archive_bytes"] = value
    expression = load_workflow_yaml(CALLER.read_text("utf-8"))["jobs"][
        "trusted_proof"
    ]["with"]["proof_inputs"]

    if expression_failure:
        with pytest.raises(json.JSONDecodeError):
            _evaluate_caller_proof_inputs(dispatch_inputs, expression)
    else:
        with pytest.raises(
            AdmissionError, match="proof_inputs_type:candidate_archive_bytes"
        ):
            admit_inputs(
                _evaluate_caller_proof_inputs(dispatch_inputs, expression),
                _compact(_companion_fixture()),
            )


def test_caller_fails_closed_for_invalid_optional_dispatch_byte_string():
    dispatch_inputs = _string_dispatch_inputs(multipart=True)
    dispatch_inputs["model_pack_part_01_bytes"] = "not-a-number"
    expression = load_workflow_yaml(CALLER.read_text("utf-8"))["jobs"][
        "trusted_proof"
    ]["with"]["proof_inputs"]

    with pytest.raises(json.JSONDecodeError):
        _evaluate_caller_proof_inputs(dispatch_inputs, expression)


def test_caller_normalizes_an_omitted_whole_model_url_before_exact_proof_admission():
    dispatch_inputs = _dispatch_inputs_fixture(multipart=True)
    raw_to_json_inputs = _compact(dispatch_inputs)

    assert len(json.loads(raw_to_json_inputs)) == 23
    with pytest.raises(AdmissionError, match="proof_inputs_key_set"):
        admit_inputs(raw_to_json_inputs, _compact(_companion_fixture()))

    workflow = load_workflow_yaml(CALLER.read_text("utf-8"))
    assert (
        workflow["jobs"]["trusted_proof"]["with"]["proof_inputs"]
        == _explicit_proof_inputs_expression()
    )

    canonical = {
        key: dispatch_inputs.get(key, 0 if key.endswith("_bytes") else "")
        for key in PROOF_KEYS
    }
    proof, _ = admit_inputs(_compact(canonical), _compact(_companion_fixture()))
    assert list(json.loads(proof)) == list(PROOF_KEYS)


@pytest.mark.parametrize(
    ("multipart", "omitted_count"), ((True, 23), (False, 15))
)
def test_caller_explicit_envelope_admits_both_exact_model_delivery_shapes(
    multipart: bool, omitted_count: int
):
    dispatch_inputs = _dispatch_inputs_fixture(multipart=multipart)
    assert len(dispatch_inputs) == omitted_count
    canonical = {
        key: dispatch_inputs.get(key, 0 if key.endswith("_bytes") else "")
        for key in PROOF_KEYS
    }

    proof, _ = admit_inputs(_compact(canonical), _compact(_companion_fixture()))

    admitted = json.loads(proof)
    assert set(admitted) == set(PROOF_KEYS)
    assert admitted["model_pack_url"] == (
        "" if multipart else "https://example.invalid/model_pack_url"
    )
    assert all(
        admitted[key] not in ("", 0)
        for key in PROOF_KEYS
        if multipart and key.startswith("model_pack_part_")
    )
    assert all(
        admitted[key] in ("", 0)
        for key in PROOF_KEYS
        if not multipart and key.startswith("model_pack_part_")
    )


def test_caller_canonicalization_does_not_relax_proof_admission_mutations():
    canonical = {
        key: 0 if key.endswith("_bytes") and key.startswith("model_pack_part_") else
        "" if (not key.endswith("_bytes") and key.startswith("model_pack_part_")) else
        1 if key.endswith("_bytes") else
        "a" * 64 if key.endswith("_sha256") else
        f"https://example.invalid/{key}"
        for key in PROOF_KEYS
    }
    raw = _compact(canonical)

    unknown = dict(canonical)
    unknown["unexpected"] = "value"
    missing = dict(canonical)
    missing.pop("candidate_archive_url")
    placeholder = dict(canonical)
    placeholder["candidate_archive_url"] = "https://example.invalid/${MODEL_RELEASE}"
    mutations = {
        "unknown": _compact(unknown),
        "missing": _compact(missing),
        "duplicate": raw[:-1]
        + ',"candidate_archive_url":"https://evil.invalid/archive.zip"}',
        "placeholder": _compact(placeholder),
    }

    for name, mutated in mutations.items():
        assert mutated != raw, name
        with pytest.raises(AdmissionError):
            admit_inputs(mutated, _compact(_companion_fixture()))


def test_manual_trusted_precutover_caller_has_only_closed_data_inputs():
    workflow = load_workflow_yaml(CALLER.read_text("utf-8"))
    dispatch = workflow["on"]["workflow_dispatch"]
    assert set(dispatch["inputs"]) == INPUTS
    assert len(INPUTS) == 24
    for name, descriptor in dispatch["inputs"].items():
        assert set(descriptor) == {"description", "required", "type"}
        assert descriptor["type"] == ("number" if name.endswith("_bytes") else "string")
        assert descriptor["required"] == (
            "false" if name == "model_pack_url" or name in MODEL_PART_INPUTS else "true"
        )


def _assert_flattened_caller_contract(source: str) -> None:
    workflow = load_workflow_yaml(source)
    assert workflow["permissions"] == {
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }
    assert "secrets:" not in source
    assert set(workflow["jobs"]) == {"companion_builder", "trusted_proof"}

    builder = workflow["jobs"]["companion_builder"]
    assert builder["uses"] == (
        "hbhjt/vending-vision/.github/workflows/"
        f"trusted-precutover-companion-builder.yml@{COMPANION_BUILDER_SHA}"
    )
    assert builder["permissions"] == {
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }
    assert set(builder["with"]) == {
        "core_wheelhouse_url",
        "core_wheelhouse_sha256",
        "core_wheelhouse_bytes",
    }
    assert builder["with"] == {
        "core_wheelhouse_url": "${{ vars.CORE_WHEELHOUSE_URL }}",
        "core_wheelhouse_sha256": "${{ vars.CORE_WHEELHOUSE_SHA256 }}",
        "core_wheelhouse_bytes": "${{ fromJSON(vars.CORE_WHEELHOUSE_BYTES) }}",
    }

    job = workflow["jobs"]["trusted_proof"]
    assert job["uses"] == (
        "hbhjt/vending-vision/.github/workflows/"
        f"trusted-precutover-companion-proof.yml@{PROOF_SHA}"
    )
    assert job["needs"] == "companion_builder"
    assert job["permissions"] == {
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }
    assert set(job["with"]) == {"proof_inputs", "companion_builder_outputs"}
    assert job["with"]["proof_inputs"] == _explicit_proof_inputs_expression()
    assert job["with"]["companion_builder_outputs"] == (
        "${{ toJSON(needs.companion_builder.outputs) }}"
    )
    assert "runs-on:" not in source
    assert "steps:" not in source
    assert "environment:" not in source
    assert "VISION_SUPPLIER_PRIVATE_KEY_PEM" not in source


def test_caller_only_sha_pins_the_reusable_proof_and_forwards_all_inputs():
    _assert_flattened_caller_contract(CALLER.read_text("utf-8"))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "companion_builder_outputs: ${{ toJSON(needs.companion_builder.outputs) }}",
            "companion_builder_outputs: ${{ inputs.companion_builder_outputs }}",
        ),
        (_explicit_proof_inputs_expression(), "${{ toJSON(inputs) }}"),
        ("toJSON(inputs.model_pack_url || '')", "toJSON(inputs.model_pack_url)"),
        (
            "toJSON(fromJSON(inputs.model_pack_part_01_bytes || '0'))",
            "toJSON(fromJSON(inputs.model_pack_part_01_bytes))",
        ),
        (
            "toJSON(fromJSON(inputs.candidate_archive_bytes))",
            "toJSON(inputs.candidate_archive_bytes)",
        ),
        ("needs: companion_builder", "needs: missing_builder"),
        (COMPANION_BUILDER_SHA, "a" * 40),
        ("  trusted_proof:\n", "  unexpected:\n"),
        ("      attestations: write\n", "      checks: write\n"),
        ("permissions:\n  contents: read", "secrets: inherit\npermissions:\n  contents: read"),
    ],
)
def test_caller_rejects_flattened_boundary_regressions(old: str, new: str):
    source = CALLER.read_text("utf-8")
    mutated = source.replace(old, new, 1)
    assert mutated != source
    with pytest.raises(AssertionError):
        _assert_flattened_caller_contract(mutated)


def test_caller_pinned_history_contains_the_environment_authority_and_secret_isolation():
    source = CALLER.read_text("utf-8")
    match = re.search(
        r"uses: hbhjt/vending-vision/\.github/workflows/"
        r"trusted-precutover-companion-proof\.yml@([a-f0-9]{40})",
        source,
    )
    assert match is not None
    pinned_sha = match.group(1)
    completed = subprocess.run(
        [
            "git",
            "show",
            f"{pinned_sha}:.github/workflows/trusted-precutover-companion-proof.yml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    pinned = completed.stdout
    workflow = load_workflow_yaml(pinned)
    assert workflow["jobs"]["execute"]["environment"] == "experimental-candidate"
    assert workflow["jobs"]["sign"]["environment"] == "experimental-candidate"
    assert "environment" not in workflow["jobs"]["verify"]
    assert workflow["jobs"]["execute"]["permissions"] == {
        "attestations": "read",
        "contents": "read",
    }
    assert workflow["jobs"]["sign"]["permissions"] == {
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }
    assert "41afbd9bd07b67df9f93de1dea1a9f9b0cea0228" in pinned
    assert "VISION_SUPPLIER_PRIVATE_KEY_PEM" not in pinned


def test_caller_grants_only_reusable_proof_permissions_and_produces_its_exact_three_member_artifact():
    workflow = load_workflow_yaml(CALLER.read_text("utf-8"))
    assert workflow["permissions"] == {
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }
    proof = (
        ROOT / ".github/workflows/trusted-precutover-companion-proof.yml"
    ).read_text("utf-8")
    assert "path: proof-handoff/*" in proof
    assert "verify-evidence --directory proof-handoff" in proof
    assert HANDOFF_FILES == {
        "precutover-ai-proof.json",
        "precutover-ai-proof.sigstore.json",
        "trusted-precutover-proof-evidence.json",
    }
