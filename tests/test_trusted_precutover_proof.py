from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

import pytest

from scripts.candidate_artifact_manifest import BINDING_PATHS, LAYOUT
from scripts.trusted_precutover_proof import (
    ModelPackPart,
    ProofError,
    assemble_model_pack,
    bind_execution_proof,
    inspect_inputs,
    seal_evidence,
    verify_evidence,
    verify_execution_handoff,
    verify_proof,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_inputs(root: Path) -> tuple[dict, dict]:
    root.mkdir()
    candidate_root = root / "candidate"
    model_root = root / "model"
    candidate_root.mkdir()
    model_root.mkdir()
    source_revision = "3b7956b6f8f76a76c8b3f4fbdab4bb77df94bb52"
    payload = {
        BINDING_PATHS["mainExecutable"]: b"MZ-main",
        BINDING_PATHS["workerExecutable"]: b"MZ-worker",
        BINDING_PATHS["runtimeDescriptor"]: _canonical(
            {"schemaVersion": "vem-ai-runtime-descriptor/v1"}
        ),
        BINDING_PATHS["aiLock"]: _canonical({"schemaVersion": "vem-ai-wheelhouse-lock/v1"}),
        BINDING_PATHS["sourceDescriptor"]: _canonical(
            {
                "catvtonSourceRevision": source_revision,
                "schemaVersion": "vem-official-ai-source-descriptor/v1",
                "sources": [],
            }
        ),
        BINDING_PATHS["modelPackDescriptor"]: _canonical(
            {
                "catvtonSourceRevision": source_revision,
                "files": [],
                "schemaVersion": "vem-ai-model-pack/v1",
                "totalByteSize": 0,
            }
        ),
    }
    files = [
        {"path": path, "sha256": _sha(raw), "size": len(raw)}
        for path, raw in sorted(payload.items())
    ]
    bindings = {
        name: {"path": path, "sha256": _sha(payload[path])}
        for name, path in BINDING_PATHS.items()
    }
    source_commit = "a" * 40
    manifest = {
        "bindings": bindings,
        "files": files,
        "layout": LAYOUT,
        "schemaVersion": "vending-vision-candidate-artifact/v3",
        "sourceCommit": source_commit,
    }
    manifest_raw = _canonical(manifest)
    (candidate_root / "candidate-manifest.json").write_bytes(manifest_raw)
    candidate = candidate_root / "candidate.zip"
    with zipfile.ZipFile(candidate, "w") as archive:
        for name, raw in {"candidate-manifest.json": manifest_raw, **payload}.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, raw)
    attestation_raw = b'{"fixture":"github-attestation"}'
    (candidate_root / "github-build-provenance.sigstore.json").write_bytes(attestation_raw)
    subject_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    evidence = {
        "schemaVersion": "vending-vision-trusted-builder-evidence/v1",
        "builderRepository": "hbhjt/vending-vision",
        "builderWorkflow": ".github/workflows/trusted-ai-candidate-builder.yml",
        "builderWorkflowSha": "c90a965d117fea49f318b18e0fcd50aa047bc41",
        "sourceCommit": source_commit,
        "subjectSha256": subject_sha,
        "embeddedManifestSha256": _sha(manifest_raw),
        "attestationBundleSha256": _sha(attestation_raw),
    }
    (candidate_root / "trusted-builder-evidence.json").write_text(
        json.dumps(evidence, separators=(",", ":")), "utf-8"
    )
    model_raw = b"official model archive fixture"
    (model_root / "official-model-pack.zip").write_bytes(model_raw)

    expected = {
        "candidate": {
            "attestationSha256": _sha(attestation_raw),
            "trustedBuilderEvidenceSha256": _sha(
                (candidate_root / "trusted-builder-evidence.json").read_bytes()
            ),
            "manifestSha256": _sha(manifest_raw),
            "sourceCommit": source_commit,
            "subjectSha256": subject_sha,
        },
        "modelPack": {
            "byteSize": len(model_raw),
            "descriptorSha256": bindings["modelPackDescriptor"]["sha256"],
            "sha256": _sha(model_raw),
            "sourceRevision": source_revision,
        },
        "resources": {
            "aiLockSha256": bindings["aiLock"]["sha256"],
            "runtimeDescriptorSha256": bindings["runtimeDescriptor"]["sha256"],
            "sourceDescriptorSha256": bindings["sourceDescriptor"]["sha256"],
            "workerExecutableSha256": bindings["workerExecutable"]["sha256"],
        },
        "schemaVersion": "vending-vision-trusted-precutover-inputs/v1",
    }
    return expected, manifest


def proof_for(identity: dict) -> dict:
    probe = {
        "catvtonSourceRevision": identity["modelPack"]["sourceRevision"],
        "packaging": "25.0",
        "probe": "official-catvton-worker-runtime",
    }
    model_probe = {**probe, "probe": "official-catvton-worker"}
    return {
        "candidate": {
            "attestationBundleSha256": identity["candidate"]["attestationSha256"],
            "embeddedManifestSha256": identity["candidate"]["manifestSha256"],
            "sourceCommit": identity["candidate"]["sourceCommit"],
            "subjectSha256": identity["candidate"]["subjectSha256"],
            "trustedBuilderEvidenceSha256": identity["candidate"][
                "trustedBuilderEvidenceSha256"
            ],
            "workerExecutableSha256": identity["resources"]["workerExecutableSha256"],
            "workerMode": "frozen-windows",
        },
        "modelPack": {
            "archive": {
                "byteSize": identity["modelPack"]["byteSize"],
                "sha256": identity["modelPack"]["sha256"],
            },
            "descriptorSha256": identity["modelPack"]["descriptorSha256"],
            "sourceRevision": identity["modelPack"]["sourceRevision"],
        },
        "probes": {"model": model_probe, "runtime": probe},
        "resources": {
            "aiLockSha256": identity["resources"]["aiLockSha256"],
            "runtimeDescriptorSha256": identity["resources"]["runtimeDescriptorSha256"],
            "sourceDescriptorSha256": identity["resources"]["sourceDescriptorSha256"],
        },
        "companion": {
            "archiveSha256": "c" * 64,
            "descriptorSha256": "d" * 64,
            "sourceCommit": "d8c93b50cac005a371d06badcc398638fd8acabb",
        },
        "schemaVersion": "vending-vision-precutover-proof/v2",
    }


def test_model_pack_assembler_reconstructs_the_exact_three_verified_parts(tmp_path):
    parts_root = tmp_path / "model-parts"
    parts_root.mkdir()
    parts = (b"official-", b"model-", b"archive")
    expected_parts = []
    for index, raw in enumerate(parts, start=1):
        name = f"official-model-pack.part{index:02d}"
        (parts_root / name).write_bytes(raw)
        expected_parts.append(
            ModelPackPart(name=name, sha256=_sha(raw), byte_size=len(raw))
        )
    archive = b"".join(parts)
    destination = tmp_path / "official-model-pack.zip"

    assemble_model_pack(
        parts_root,
        destination,
        parts=tuple(expected_parts),
        expected_sha256=_sha(archive),
        expected_bytes=len(archive),
    )

    assert destination.read_bytes() == archive


@pytest.mark.parametrize("mutation", ["extra", "reordered", "rewritten-part"])
def test_model_pack_assembler_rejects_extra_or_position_drift(tmp_path, mutation):
    parts_root = tmp_path / "model-parts"
    parts_root.mkdir()
    raw_parts = (b"official-", b"model-", b"archive")
    parts = tuple(
        ModelPackPart(
            name=f"official-model-pack.part{index:02d}",
            sha256=_sha(raw),
            byte_size=len(raw),
        )
        for index, raw in enumerate(raw_parts, start=1)
    )
    for part, raw in zip(parts, raw_parts, strict=True):
        (parts_root / part.name).write_bytes(raw)
    archive = b"".join(raw_parts)
    if mutation == "extra":
        (parts_root / "untrusted-extra.bin").write_bytes(b"extra")
    elif mutation == "reordered":
        parts = (parts[1], parts[0], parts[2])
    else:
        (parts_root / parts[1].name).write_bytes(b"rewritten")

    with pytest.raises(ProofError):
        assemble_model_pack(
            parts_root,
            tmp_path / "official-model-pack.zip",
            parts=parts,
            expected_sha256=_sha(archive),
            expected_bytes=len(archive),
        )


def test_inspector_derives_identity_and_accepts_only_bound_canonical_frozen_proof(tmp_path):
    expected, _ = build_inputs(tmp_path / "inputs")
    identity = inspect_inputs(tmp_path / "inputs")
    assert identity == expected

    proof = proof_for(identity)
    proof_path = tmp_path / "proof.json"
    proof_path.write_bytes(_canonical(proof) + b"\n")
    verify_proof(proof_path, identity)


def test_execute_binds_verified_companion_and_builder_evidence_into_signed_proof(tmp_path):
    identity, _ = build_inputs(tmp_path / "inputs")
    final_proof = proof_for(identity)
    companion_report = {**final_proof}
    companion_report["candidate"] = {
        key: value
        for key, value in final_proof["candidate"].items()
        if key != "trustedBuilderEvidenceSha256"
    }
    companion_report.pop("companion")
    companion_report["schemaVersion"] = "vending-vision-precutover-proof/v1"
    report_path = tmp_path / "companion-report.json"
    report_path.write_bytes(_canonical(companion_report) + b"\n")
    proof_path = tmp_path / "precutover-ai-proof.json"

    actual = bind_execution_proof(
        report_path,
        identity,
        companion_archive_sha256="c" * 64,
        companion_descriptor_sha256="d" * 64,
        companion_source_commit="d8c93b50cac005a371d06badcc398638fd8acabb",
        output=proof_path,
    )

    assert actual == final_proof
    assert verify_proof(proof_path, identity) == final_proof


@pytest.mark.parametrize(
    "mutation",
    ["fake-emitter", "wrong-subject", "wrong-probe-version", "missing", "extra"],
)
def test_proof_validator_rejects_emitters_wrong_bindings_and_nonexact_shapes(
    tmp_path, mutation
):
    expected, _ = build_inputs(tmp_path / "inputs")
    proof = proof_for(expected)
    if mutation == "fake-emitter":
        proof["candidate"]["workerMode"] = "source-test-only"
    elif mutation == "wrong-subject":
        proof["candidate"]["subjectSha256"] = "0" * 64
    elif mutation == "wrong-probe-version":
        proof["probes"]["model"]["packaging"] = "99.0"
    elif mutation == "missing":
        del proof["resources"]["aiLockSha256"]
    else:
        proof["emitterClaim"] = "self-asserted"
    proof_path = tmp_path / "proof.json"
    proof_path.write_bytes(_canonical(proof) + b"\n")

    with pytest.raises(ProofError):
        verify_proof(proof_path, expected)


def test_proof_evidence_round_trip_rejects_missing_and_extra_artifact_members(tmp_path):
    identity, _ = build_inputs(tmp_path / "inputs")
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    proof = handoff / "precutover-ai-proof.json"
    proof.write_bytes(_canonical(proof_for(identity)) + b"\n")
    bundle = handoff / "precutover-ai-proof.sigstore.json"
    bundle.write_bytes(b'{"bundle":"fixture"}')
    evidence = handoff / "trusted-precutover-proof-evidence.json"
    workflow_sha = "b" * 40
    companion_archive = "c" * 64
    companion_descriptor = "d" * 64
    seal_evidence(
        handoff,
        identity,
        workflow_sha=workflow_sha,
        companion_archive_sha256=companion_archive,
        companion_descriptor_sha256=companion_descriptor,
        output=evidence,
    )
    arguments = {
        "workflow_sha": workflow_sha,
        "proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
        "attestation_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "source_commit": identity["candidate"]["sourceCommit"],
        "companion_archive_sha256": companion_archive,
        "companion_descriptor_sha256": companion_descriptor,
    }
    verify_evidence(handoff, **arguments)

    extra = handoff / "self-asserted.json"
    extra.write_text("{}", "utf-8")
    with pytest.raises(ProofError, match="exact_set"):
        verify_evidence(handoff, **arguments)
    extra.unlink()
    evidence.unlink()
    with pytest.raises(ProofError, match="exact_set"):
        verify_evidence(handoff, **arguments)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("archiveSha256", "1" * 64),
        ("descriptorSha256", "2" * 64),
        ("sourceCommit", "3" * 40),
        ("trustedBuilderEvidenceSha256", "4" * 64),
    ],
)
def test_signed_proof_rejects_rewritten_unsigned_identity_evidence(
    tmp_path, field, replacement
):
    identity, _ = build_inputs(tmp_path / "inputs")
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    proof = handoff / "precutover-ai-proof.json"
    proof.write_bytes(_canonical(proof_for(identity)) + b"\n")
    bundle = handoff / "precutover-ai-proof.sigstore.json"
    bundle.write_bytes(b'{"bundle":"fixture"}')
    evidence_path = handoff / "trusted-precutover-proof-evidence.json"
    arguments = {
        "workflow_sha": "b" * 40,
        "proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
        "attestation_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "source_commit": identity["candidate"]["sourceCommit"],
        "companion_archive_sha256": "c" * 64,
        "companion_descriptor_sha256": "d" * 64,
    }
    seal_evidence(
        handoff,
        identity,
        workflow_sha=arguments["workflow_sha"],
        companion_archive_sha256=arguments["companion_archive_sha256"],
        companion_descriptor_sha256=arguments["companion_descriptor_sha256"],
        output=evidence_path,
    )
    evidence = json.loads(evidence_path.read_bytes())
    if field == "trustedBuilderEvidenceSha256":
        evidence["inputIdentity"]["candidate"][field] = replacement
    else:
        evidence["companion"][field] = replacement
        if field == "archiveSha256":
            arguments["companion_archive_sha256"] = replacement
        elif field == "descriptorSha256":
            arguments["companion_descriptor_sha256"] = replacement
        else:
            # This value was never accepted as a verifier argument, so it was
            # previously self-asserted by the evidence file alone.
            pass
    evidence_path.write_bytes(_canonical(evidence))

    with pytest.raises(ProofError):
        verify_evidence(handoff, **arguments)


@pytest.mark.parametrize("location", ["candidate", "model"])
def test_input_inspector_rejects_extra_members_outside_exact4_and_model(location, tmp_path):
    build_inputs(tmp_path / "inputs")
    (tmp_path / "inputs" / location / "extra.bin").write_bytes(b"untrusted")

    with pytest.raises(ProofError, match="exact_set"):
        inspect_inputs(tmp_path / "inputs")


def test_trusted_proof_helper_cli_imports_with_stdlib_only(tmp_path):
    environment = tmp_path / "clean-python"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True
    )
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    script = Path(__file__).parents[1] / "scripts" / "trusted_precutover_proof.py"

    completed = subprocess.run(
        [str(python), str(script), "--help"], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_execution_handoff_is_exactly_one_canonical_bound_proof(tmp_path):
    identity, _ = build_inputs(tmp_path / "inputs")
    handoff = tmp_path / "execution-handoff"
    handoff.mkdir()
    proof = handoff / "precutover-ai-proof.json"
    proof.write_bytes(_canonical(proof_for(identity)) + b"\n")
    verify_execution_handoff(handoff, identity)

    extra = handoff / "self-asserted.json"
    extra.write_text("{}", "utf-8")
    with pytest.raises(ProofError, match="exact_set"):
        verify_execution_handoff(handoff, identity)

    extra.unlink()
    symlink = tmp_path / "execution-handoff-link"
    symlink.symlink_to(handoff, target_is_directory=True)
    with pytest.raises(ProofError, match="root"):
        verify_execution_handoff(symlink, identity)
