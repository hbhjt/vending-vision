from __future__ import annotations

import json

import pytest

from scripts.admit_trusted_precutover_inputs import AdmissionError, admit_inputs


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


def _valid_proof(*, multipart: bool = False) -> dict[str, object]:
    value: dict[str, object] = {}
    for key in PROOF_KEYS:
        if key.endswith("_bytes"):
            value[key] = 1
        elif key.endswith("_sha256"):
            value[key] = "a" * 64
        elif key.endswith("_url"):
            value[key] = f"https://example.invalid/{key}"
    if multipart:
        value["model_pack_url"] = ""
    else:
        for key in PROOF_KEYS:
            if key.startswith("model_pack_part_"):
                value[key] = 0 if key.endswith("_bytes") else ""
    return value


def _valid_companion() -> dict[str, str]:
    return {
        key: ("a" * 64 if key.endswith("_sha256") else f"fixed-{key}")
        for key in COMPANION_KEYS
    }


def _raw(value: dict[str, object]) -> str:
    return json.dumps(value, separators=(",", ":"))


def test_admission_returns_fixed_order_compact_canonical_envelopes():
    proof, companion = admit_inputs(
        json.dumps(_valid_proof(), indent=2), json.dumps(_valid_companion(), indent=2)
    )
    assert proof == _raw(_valid_proof())
    assert companion == _raw(_valid_companion())
    assert list(json.loads(proof)) == list(PROOF_KEYS)
    assert list(json.loads(companion)) == list(COMPANION_KEYS)


@pytest.mark.parametrize("envelope", ["proof", "companion"])
def test_admission_rejects_duplicate_keys(envelope: str):
    proof = _raw(_valid_proof())
    companion = _raw(_valid_companion())
    if envelope == "proof":
        proof = proof[:-1] + ',"candidate_archive_url":"https://evil.invalid/x"}'
    else:
        companion = companion[:-1] + ',"artifact_name":"decoy"}'
    with pytest.raises(AdmissionError, match="duplicate_key"):
        admit_inputs(proof, companion)


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong-type", "unsafe-int"])
def test_admission_rejects_non_exact_or_non_typed_proof_inputs(mutation: str):
    proof = _valid_proof()
    if mutation == "missing":
        proof.pop("candidate_archive_url")
    elif mutation == "extra":
        proof["unexpected"] = "value"
    elif mutation == "wrong-type":
        proof["candidate_archive_bytes"] = "1"
    else:
        proof["candidate_archive_bytes"] = 2**53
    with pytest.raises(AdmissionError):
        admit_inputs(_raw(proof), _raw(_valid_companion()))


@pytest.mark.parametrize("multipart", [False, True])
def test_admission_accepts_exactly_one_model_delivery_shape(multipart: bool):
    proof, _ = admit_inputs(
        _raw(_valid_proof(multipart=multipart)), _raw(_valid_companion())
    )
    assert json.loads(proof)["model_pack_url"] == (
        "" if multipart else "https://example.invalid/model_pack_url"
    )


def test_admission_rejects_mixed_or_incomplete_model_delivery_shapes():
    mixed = _valid_proof(multipart=True)
    mixed["model_pack_url"] = "https://example.invalid/model.zip"
    incomplete = _valid_proof(multipart=True)
    incomplete["model_pack_part_03_url"] = ""
    for proof in (mixed, incomplete):
        with pytest.raises(AdmissionError, match="model_mode"):
            admit_inputs(_raw(proof), _raw(_valid_companion()))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("candidate_archive_url", "http://example.invalid/archive.zip"),
        ("candidate_archive_url", "https://user@example.invalid/archive.zip"),
        ("candidate_archive_url", "https://example.invalid/archive.zip#fragment"),
        ("candidate_archive_sha256", "A" * 64),
        ("candidate_archive_bytes", 0),
        ("candidate_archive_url", "https://example.invalid/\u0001"),
    ],
)
def test_admission_rejects_noncanonical_identities(key: str, value: object):
    proof = _valid_proof()
    proof[key] = value
    with pytest.raises(AdmissionError):
        admit_inputs(_raw(proof), _raw(_valid_companion()))


@pytest.mark.parametrize("placeholder", ["$MODEL_RELEASE", "${MODEL_RELEASE}", "{MODEL_RELEASE}"])
def test_admission_rejects_placeholders_in_every_string_field(placeholder: str):
    for key in PROOF_KEYS:
        if key.endswith("_bytes"):
            continue
        proof = _valid_proof(multipart=key.startswith("model_pack_part_"))
        proof[key] = (
            f"https://example.invalid/release/{placeholder}/artifact"
            if key.endswith("_url")
            else placeholder
        )
        with pytest.raises(AdmissionError, match="placeholder"):
            admit_inputs(_raw(proof), _raw(_valid_companion()))

    for key in COMPANION_KEYS:
        companion = _valid_companion()
        companion[key] = f"fixed-{placeholder}"
        with pytest.raises(AdmissionError, match="placeholder"):
            admit_inputs(_raw(_valid_proof()), _raw(companion))
