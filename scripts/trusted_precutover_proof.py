"""Validate and seal trusted Windows pre-cutover companion proof data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile
from dataclasses import dataclass

if __package__:
    from scripts.candidate_artifact_manifest import (
        BINDING_PATHS,
        EMBEDDED_MANIFEST,
        _load_manifest,
    )
else:
    from candidate_artifact_manifest import (  # type: ignore[no-redef]
        BINDING_PATHS,
        EMBEDDED_MANIFEST,
        _load_manifest,
    )


INPUT_SCHEMA = "vending-vision-trusted-precutover-inputs/v1"
COMPANION_REPORT_SCHEMA = "vending-vision-precutover-proof/v1"
PROOF_SCHEMA = "vending-vision-precutover-proof/v2"
EVIDENCE_SCHEMA = "vending-vision-trusted-precutover-proof-evidence/v1"
TRUSTED_REPOSITORY = "hbhjt/vending-vision"
TRUSTED_CANDIDATE_WORKFLOW = ".github/workflows/trusted-ai-candidate-builder.yml"
TRUSTED_CANDIDATE_WORKFLOW_SHA = "ccf38c95b3c33994d4c20039a506be062ebd2f99"
TRUSTED_PROOF_WORKFLOW = ".github/workflows/trusted-precutover-companion-proof.yml"
TRUSTED_COMPANION_SOURCE = "3abc93b61b5eaa7d4302117f593641dccf1d969c"
CANDIDATE_INPUT_FILES = {
    "candidate.zip",
    "candidate-manifest.json",
    "github-build-provenance.sigstore.json",
    "trusted-builder-evidence.json",
}
MODEL_INPUT_FILES = {"official-model-pack.zip"}
HANDOFF_FILES = {
    "precutover-ai-proof.json",
    "precutover-ai-proof.sigstore.json",
    "trusted-precutover-proof-evidence.json",
}
EXECUTION_HANDOFF_FILES = {"precutover-ai-proof.json"}
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_RESOURCE_BYTES = 64 * 1024 * 1024
MAX_MODEL_PACK_BYTES = 8 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class ModelPackPart:
    """One position-bound immutable model archive fragment."""

    name: str
    sha256: str
    byte_size: int


class ProofError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble_model_pack(
    parts_root: Path,
    destination: Path,
    *,
    parts: tuple[ModelPackPart, ...],
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    """Verify exactly three ordered fragments and publish their archive bytes.

    The complete archive remains the sole input to the frozen model verifier.
    Fragments are only a transport contract and never become an alternate model
    identity or a variable-length collection.
    """

    expected_names = tuple(
        f"official-model-pack.part{index:02d}" for index in range(1, 4)
    )
    if (
        not isinstance(parts, tuple)
        or len(parts) != 3
        or not all(isinstance(part, ModelPackPart) for part in parts)
        or tuple(part.name for part in parts) != expected_names
        or any(
            SHA256_RE.fullmatch(part.sha256) is None
            or type(part.byte_size) is not int
            or part.byte_size <= 0
            for part in parts
        )
    ):
        raise ProofError("model_parts_contract")
    if (
        SHA256_RE.fullmatch(expected_sha256) is None
        or type(expected_bytes) is not int
        or expected_bytes <= 0
        or expected_bytes > MAX_MODEL_PACK_BYTES
        or sum(part.byte_size for part in parts) != expected_bytes
    ):
        raise ProofError("model_archive_contract")
    _regular_exact_set(parts_root, set(expected_names), "model_parts")
    if (
        destination.exists()
        or destination.is_symlink()
        or not destination.parent.is_dir()
        or destination.parent.is_symlink()
        or destination.parent.resolve() != destination.parent
    ):
        raise ProofError("model_archive_destination")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        archive_digest = hashlib.sha256()
        archive_size = 0
        with os.fdopen(descriptor, "wb") as output:
            for part in parts:
                path = parts_root / part.name
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != part.byte_size
                ):
                    raise ProofError("model_part_file")
                part_digest = hashlib.sha256()
                part_size = 0
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        part_size += len(chunk)
                        if part_size > part.byte_size:
                            raise ProofError("model_part_size")
                        part_digest.update(chunk)
                        archive_size += len(chunk)
                        if archive_size > expected_bytes:
                            raise ProofError("model_archive_size")
                        archive_digest.update(chunk)
                        output.write(chunk)
                if part_size != part.byte_size or part_digest.hexdigest() != part.sha256:
                    raise ProofError("model_part_identity")
            output.flush()
            os.fsync(output.fileno())
        if archive_size != expected_bytes or archive_digest.hexdigest() != expected_sha256:
            raise ProofError("model_archive_identity")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ProofError("proof_duplicate_json_key")
        value[key] = item
    return value


def _load_json(path: Path, label: str, *, canonical: bool) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ProofError(f"{label}_file")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_float=lambda _value: (_ for _ in ()).throw(ProofError("proof_float")),
            parse_constant=lambda _value: (_ for _ in ()).throw(ProofError("proof_constant")),
        )
    except (UnicodeError, ValueError) as exc:
        raise ProofError(f"{label}_json") from exc
    if not isinstance(value, dict):
        raise ProofError(f"{label}_shape")
    expected_raw = canonical_bytes(value)
    if canonical and raw not in {expected_raw, expected_raw + b"\n"}:
        raise ProofError(f"{label}_noncanonical")
    return value, raw


def _require_exact(value: dict, keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ProofError(f"{label}_shape")


def _regular_exact_set(root: Path, expected: set[str], label: str) -> None:
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ProofError(f"{label}_root")
    actual = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ProofError(f"{label}_member")
        actual.add(path.name)
    if actual != expected:
        raise ProofError(f"{label}_exact_set")


def _input_roots(root: Path) -> tuple[Path, Path]:
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ProofError("proof_inputs_root")
    children = {path.name: path for path in root.iterdir()}
    if set(children) != {"candidate", "model"} or any(
        path.is_symlink() or not path.is_dir() for path in children.values()
    ):
        raise ProofError("proof_inputs_layout")
    candidate_root = children["candidate"].resolve()
    model_root = children["model"].resolve()
    _regular_exact_set(candidate_root, CANDIDATE_INPUT_FILES, "candidate_inputs")
    _regular_exact_set(model_root, MODEL_INPUT_FILES, "model_inputs")
    return candidate_root, model_root


def _candidate_resource(archive: zipfile.ZipFile, path: str, expected_sha256: str) -> bytes:
    if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise ProofError("candidate_resource_path")
    matches = [info for info in archive.infolist() if info.filename.casefold() == path.casefold()]
    if len(matches) != 1 or matches[0].filename != path or matches[0].file_size > MAX_RESOURCE_BYTES:
        raise ProofError("candidate_resource_member")
    raw = archive.read(matches[0])
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ProofError("candidate_resource_digest")
    return raw


def _canonical_resource(raw: bytes, label: str) -> dict:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError) as exc:
        raise ProofError(f"{label}_json") from exc
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise ProofError(f"{label}_noncanonical")
    return value


def inspect_inputs(root: Path) -> dict:
    root = root.resolve()
    candidate_root, model_root = _input_roots(root)
    candidate = candidate_root / "candidate.zip"
    manifest_path = candidate_root / "candidate-manifest.json"
    attestation = candidate_root / "github-build-provenance.sigstore.json"
    evidence_path = candidate_root / "trusted-builder-evidence.json"
    model_pack = model_root / "official-model-pack.zip"
    manifest_unchecked, manifest_raw = _load_json(
        manifest_path, "candidate_manifest", canonical=True
    )
    source_commit = manifest_unchecked.get("sourceCommit")
    if not isinstance(source_commit, str) or COMMIT_RE.fullmatch(source_commit) is None:
        raise ProofError("candidate_source_commit")
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    try:
        manifest = _load_manifest(manifest_raw, manifest_sha, source_commit)
    except AssertionError as exc:
        raise ProofError("candidate_manifest_contract") from exc
    if candidate.is_symlink() or not candidate.is_file() or not zipfile.is_zipfile(candidate):
        raise ProofError("candidate_archive")
    subject_sha = sha256_file(candidate)
    attestation_sha = sha256_file(attestation)
    evidence, evidence_raw = _load_json(evidence_path, "candidate_evidence", canonical=False)
    expected_evidence = {
        "schemaVersion": "vending-vision-trusted-builder-evidence/v1",
        "builderRepository": TRUSTED_REPOSITORY,
        "builderWorkflow": TRUSTED_CANDIDATE_WORKFLOW,
        "builderWorkflowSha": TRUSTED_CANDIDATE_WORKFLOW_SHA,
        "sourceCommit": source_commit,
        "subjectSha256": subject_sha,
        "embeddedManifestSha256": manifest_sha,
        "attestationBundleSha256": attestation_sha,
    }
    if evidence != expected_evidence:
        raise ProofError("candidate_evidence_binding")
    bindings = manifest["bindings"]
    with zipfile.ZipFile(candidate) as archive:
        embedded = _candidate_resource(archive, EMBEDDED_MANIFEST, manifest_sha)
        if embedded != manifest_raw:
            raise ProofError("candidate_embedded_manifest")
        resources = {
            name: _candidate_resource(archive, binding["path"], binding["sha256"])
            for name, binding in bindings.items()
            if name in {
                "modelPackDescriptor",
                "runtimeDescriptor",
                "aiLock",
                "sourceDescriptor",
            }
        }
    model_descriptor = _canonical_resource(
        resources["modelPackDescriptor"], "model_descriptor"
    )
    source_descriptor = _canonical_resource(resources["sourceDescriptor"], "source_descriptor")
    _canonical_resource(resources["runtimeDescriptor"], "runtime_descriptor")
    _canonical_resource(resources["aiLock"], "ai_lock")
    source_revision = source_descriptor.get("catvtonSourceRevision")
    if (
        not isinstance(source_revision, str)
        or COMMIT_RE.fullmatch(source_revision) is None
        or model_descriptor.get("catvtonSourceRevision") != source_revision
    ):
        raise ProofError("model_source_identity")
    if model_pack.is_symlink() or not model_pack.is_file() or model_pack.stat().st_size <= 0:
        raise ProofError("model_pack_file")
    return {
        "candidate": {
            "attestationSha256": attestation_sha,
            "trustedBuilderEvidenceSha256": hashlib.sha256(evidence_raw).hexdigest(),
            "manifestSha256": manifest_sha,
            "sourceCommit": source_commit,
            "subjectSha256": subject_sha,
        },
        "modelPack": {
            "byteSize": model_pack.stat().st_size,
            "descriptorSha256": bindings["modelPackDescriptor"]["sha256"],
            "sha256": sha256_file(model_pack),
            "sourceRevision": source_revision,
        },
        "resources": {
            "aiLockSha256": bindings["aiLock"]["sha256"],
            "runtimeDescriptorSha256": bindings["runtimeDescriptor"]["sha256"],
            "sourceDescriptorSha256": bindings["sourceDescriptor"]["sha256"],
            "workerExecutableSha256": bindings["workerExecutable"]["sha256"],
        },
        "schemaVersion": INPUT_SCHEMA,
    }


def validate_identity(identity: dict) -> None:
    _require_exact(identity, {"candidate", "modelPack", "resources", "schemaVersion"}, "identity")
    if identity["schemaVersion"] != INPUT_SCHEMA:
        raise ProofError("identity_schema")
    _require_exact(
        identity["candidate"],
        {
            "attestationSha256",
            "manifestSha256",
            "sourceCommit",
            "subjectSha256",
            "trustedBuilderEvidenceSha256",
        },
        "identity_candidate",
    )
    _require_exact(
        identity["modelPack"],
        {"byteSize", "descriptorSha256", "sha256", "sourceRevision"},
        "identity_model",
    )
    _require_exact(
        identity["resources"],
        {
            "aiLockSha256",
            "runtimeDescriptorSha256",
            "sourceDescriptorSha256",
            "workerExecutableSha256",
        },
        "identity_resources",
    )
    digests = [
        value
        for section in (identity["candidate"], identity["modelPack"], identity["resources"])
        for key, value in section.items()
        if key.endswith("Sha256")
    ]
    if any(not isinstance(value, str) or SHA256_RE.fullmatch(value) is None for value in digests):
        raise ProofError("identity_digest")
    if (
        COMMIT_RE.fullmatch(identity["candidate"]["sourceCommit"] or "") is None
        or COMMIT_RE.fullmatch(identity["modelPack"]["sourceRevision"] or "") is None
        or type(identity["modelPack"]["byteSize"]) is not int
        or identity["modelPack"]["byteSize"] <= 0
    ):
        raise ProofError("identity_value")


def _validate_probes(identity: dict, proof: dict) -> dict:
    probes = proof.get("probes")
    if not isinstance(probes, dict) or set(probes) != {"model", "runtime"}:
        raise ProofError("proof_probes_shape")
    runtime = probes["runtime"]
    model = probes["model"]
    if not isinstance(runtime, dict) or not isinstance(model, dict):
        raise ProofError("proof_probe_shape")
    if set(runtime) != set(model) or set(runtime) < {"probe", "catvtonSourceRevision"}:
        raise ProofError("proof_probe_keys")
    expected_revision = identity["modelPack"]["sourceRevision"]
    if (
        runtime.get("probe") != "official-catvton-worker-runtime"
        or model.get("probe") != "official-catvton-worker"
        or runtime.get("catvtonSourceRevision") != expected_revision
        or model.get("catvtonSourceRevision") != expected_revision
    ):
        raise ProofError("proof_probe_identity")
    for key in set(runtime) - {"probe"}:
        if runtime[key] != model[key] or not isinstance(runtime[key], str) or not runtime[key]:
            raise ProofError("proof_probe_value")
    return probes


def _expected_proof(identity: dict, proof: dict) -> dict:
    probes = _validate_probes(identity, proof)
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
        "companion": proof.get("companion"),
        "modelPack": {
            "archive": {
                "byteSize": identity["modelPack"]["byteSize"],
                "sha256": identity["modelPack"]["sha256"],
            },
            "descriptorSha256": identity["modelPack"]["descriptorSha256"],
            "sourceRevision": identity["modelPack"]["sourceRevision"],
        },
        "probes": probes,
        "resources": {
            "aiLockSha256": identity["resources"]["aiLockSha256"],
            "runtimeDescriptorSha256": identity["resources"]["runtimeDescriptorSha256"],
            "sourceDescriptorSha256": identity["resources"]["sourceDescriptorSha256"],
        },
        "schemaVersion": PROOF_SCHEMA,
    }


def verify_proof(path: Path, identity: dict) -> dict:
    validate_identity(identity)
    proof, raw = _load_json(path.resolve(), "companion_proof", canonical=True)
    if raw != canonical_bytes(proof) + b"\n":
        raise ProofError("companion_proof_newline")
    companion = proof.get("companion")
    if not isinstance(companion, dict):
        raise ProofError("proof_companion_shape")
    _require_exact(
        companion,
        {"archiveSha256", "descriptorSha256", "sourceCommit"},
        "proof_companion",
    )
    if (
        any(
            not isinstance(companion[key], str)
            or SHA256_RE.fullmatch(companion[key]) is None
            for key in ("archiveSha256", "descriptorSha256")
        )
        or not isinstance(companion["sourceCommit"], str)
        or COMMIT_RE.fullmatch(companion["sourceCommit"]) is None
    ):
        raise ProofError("proof_companion_identity")
    if proof != _expected_proof(identity, proof):
        raise ProofError("companion_proof_binding")
    return proof


def bind_execution_proof(
    source: Path,
    identity: dict,
    *,
    companion_archive_sha256: str,
    companion_descriptor_sha256: str,
    companion_source_commit: str,
    output: Path,
) -> dict:
    report, raw = _load_json(source.resolve(), "companion_report", canonical=True)
    if raw != canonical_bytes(report) + b"\n":
        raise ProofError("companion_report_newline")
    if report.get("schemaVersion") != COMPANION_REPORT_SCHEMA:
        raise ProofError("companion_report_schema")
    expected = _expected_proof(identity, {**report, "companion": {}})
    expected.pop("companion")
    expected["candidate"].pop("trustedBuilderEvidenceSha256")
    expected["schemaVersion"] = COMPANION_REPORT_SCHEMA
    if report != expected:
        raise ProofError("companion_report_binding")
    proof = {
        **report,
        "candidate": {
            **report["candidate"],
            "trustedBuilderEvidenceSha256": identity["candidate"][
                "trustedBuilderEvidenceSha256"
            ],
        },
        "companion": {
            "archiveSha256": companion_archive_sha256,
            "descriptorSha256": companion_descriptor_sha256,
            "sourceCommit": companion_source_commit,
        },
        "schemaVersion": PROOF_SCHEMA,
    }
    _write_exclusive_line(output, proof)
    return verify_proof(output, identity)


def verify_execution_handoff(directory: Path, identity: dict) -> dict:
    _regular_exact_set(directory, EXECUTION_HANDOFF_FILES, "proof_execution_handoff")
    return verify_proof(directory / "precutover-ai-proof.json", identity)


def _write_exclusive(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise ProofError("proof_output_exists")
    raw = canonical_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_exclusive_line(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise ProofError("proof_output_exists")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seal_evidence(
    directory: Path,
    identity: dict,
    *,
    workflow_sha: str,
    companion_archive_sha256: str,
    companion_descriptor_sha256: str,
    output: Path,
) -> dict:
    validate_identity(identity)
    if (
        directory.resolve() != output.parent.resolve()
        or output.name != "trusted-precutover-proof-evidence.json"
    ):
        raise ProofError("evidence_output_parent")
    current = {path.name for path in directory.iterdir()}
    if current != HANDOFF_FILES - {output.name}:
        raise ProofError("proof_handoff_preseal_set")
    proof = directory / "precutover-ai-proof.json"
    bundle = directory / "precutover-ai-proof.sigstore.json"
    bound_proof = verify_proof(proof, identity)
    if bound_proof["companion"] != {
        "archiveSha256": companion_archive_sha256,
        "descriptorSha256": companion_descriptor_sha256,
        "sourceCommit": TRUSTED_COMPANION_SOURCE,
    }:
        raise ProofError("companion_identity")
    if COMMIT_RE.fullmatch(workflow_sha) is None or bundle.is_symlink() or not bundle.is_file():
        raise ProofError("proof_attestation")
    evidence = {
        "attestation": {"sha256": sha256_file(bundle)},
        "companion": bound_proof["companion"],
        "inputIdentity": identity,
        "proof": {"byteSize": proof.stat().st_size, "sha256": sha256_file(proof)},
        "schemaVersion": EVIDENCE_SCHEMA,
        "workflow": {
            "repository": TRUSTED_REPOSITORY,
            "sha": workflow_sha,
            "workflow": TRUSTED_PROOF_WORKFLOW,
        },
    }
    _write_exclusive(output, evidence)
    return evidence


def verify_evidence(
    directory: Path,
    *,
    workflow_sha: str,
    proof_sha256: str,
    attestation_sha256: str,
    source_commit: str,
    companion_archive_sha256: str,
    companion_descriptor_sha256: str,
) -> dict:
    directory = directory.resolve()
    _regular_exact_set(directory, HANDOFF_FILES, "proof_handoff")
    evidence, raw = _load_json(
        directory / "trusted-precutover-proof-evidence.json",
        "proof_evidence",
        canonical=True,
    )
    if raw != canonical_bytes(evidence):
        raise ProofError("proof_evidence_canonical")
    _require_exact(
        evidence,
        {"attestation", "companion", "inputIdentity", "proof", "schemaVersion", "workflow"},
        "proof_evidence",
    )
    identity = evidence["inputIdentity"]
    bound_proof = verify_proof(directory / "precutover-ai-proof.json", identity)
    expected = {
        "attestation": {"sha256": attestation_sha256},
        "companion": bound_proof["companion"],
        "inputIdentity": identity,
        "proof": {
            "byteSize": (directory / "precutover-ai-proof.json").stat().st_size,
            "sha256": proof_sha256,
        },
        "schemaVersion": EVIDENCE_SCHEMA,
        "workflow": {
            "repository": TRUSTED_REPOSITORY,
            "sha": workflow_sha,
            "workflow": TRUSTED_PROOF_WORKFLOW,
        },
    }
    if evidence != expected:
        raise ProofError("proof_evidence_binding")
    if (
        sha256_file(directory / "precutover-ai-proof.json") != proof_sha256
        or sha256_file(directory / "precutover-ai-proof.sigstore.json") != attestation_sha256
        or identity["candidate"]["sourceCommit"] != source_commit
    ):
        raise ProofError("proof_evidence_file_binding")
    return evidence


def _load_identity(path: Path) -> dict:
    identity, raw = _load_json(path.resolve(), "proof_identity", canonical=True)
    if raw != canonical_bytes(identity):
        raise ProofError("proof_identity_canonical")
    validate_identity(identity)
    return identity


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-inputs")
    inspect.add_argument("--input-root", required=True, type=Path)
    inspect.add_argument("--identity-output", required=True, type=Path)
    assemble = commands.add_parser("assemble-model-pack")
    assemble.add_argument("--parts-root", required=True, type=Path)
    assemble.add_argument("--part-name", required=True, action="append")
    assemble.add_argument("--part-sha256", required=True, action="append")
    assemble.add_argument("--part-bytes", required=True, action="append", type=int)
    assemble.add_argument("--expected-sha256", required=True)
    assemble.add_argument("--expected-bytes", required=True, type=int)
    assemble.add_argument("--destination", required=True, type=Path)
    verify = commands.add_parser("verify-proof")
    verify.add_argument("--proof", required=True, type=Path)
    verify.add_argument("--identity", required=True, type=Path)
    execution = commands.add_parser("verify-execution-handoff")
    execution.add_argument("--directory", required=True, type=Path)
    execution.add_argument("--identity", required=True, type=Path)
    bind = commands.add_parser("bind-execution-proof")
    bind.add_argument("--source", required=True, type=Path)
    bind.add_argument("--identity", required=True, type=Path)
    bind.add_argument("--companion-archive-sha256", required=True)
    bind.add_argument("--companion-descriptor-sha256", required=True)
    bind.add_argument("--companion-source-commit", required=True)
    bind.add_argument("--output", required=True, type=Path)
    seal = commands.add_parser("seal-evidence")
    seal.add_argument("--directory", required=True, type=Path)
    seal.add_argument("--identity", required=True, type=Path)
    seal.add_argument("--workflow-sha", required=True)
    seal.add_argument("--companion-archive-sha256", required=True)
    seal.add_argument("--companion-descriptor-sha256", required=True)
    seal.add_argument("--output", required=True, type=Path)
    evidence = commands.add_parser("verify-evidence")
    evidence.add_argument("--directory", required=True, type=Path)
    evidence.add_argument("--workflow-sha", required=True)
    evidence.add_argument("--proof-sha256", required=True)
    evidence.add_argument("--attestation-sha256", required=True)
    evidence.add_argument("--source-commit", required=True)
    evidence.add_argument("--companion-archive-sha256", required=True)
    evidence.add_argument("--companion-descriptor-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.command == "inspect-inputs":
            identity = inspect_inputs(args.input_root.resolve())
            _write_exclusive(args.identity_output.resolve(), identity)
        elif args.command == "assemble-model-pack":
            if not (
                len(args.part_name)
                == len(args.part_sha256)
                == len(args.part_bytes)
                == 3
            ):
                raise ProofError("model_parts_contract")
            assemble_model_pack(
                args.parts_root.resolve(),
                args.destination.resolve(),
                parts=tuple(
                    ModelPackPart(name=name, sha256=sha256, byte_size=byte_size)
                    for name, sha256, byte_size in zip(
                        args.part_name, args.part_sha256, args.part_bytes, strict=True
                    )
                ),
                expected_sha256=args.expected_sha256,
                expected_bytes=args.expected_bytes,
            )
        elif args.command == "verify-proof":
            verify_proof(args.proof.resolve(), _load_identity(args.identity))
        elif args.command == "verify-execution-handoff":
            verify_execution_handoff(args.directory, _load_identity(args.identity))
        elif args.command == "bind-execution-proof":
            bind_execution_proof(
                args.source.resolve(),
                _load_identity(args.identity),
                companion_archive_sha256=args.companion_archive_sha256,
                companion_descriptor_sha256=args.companion_descriptor_sha256,
                companion_source_commit=args.companion_source_commit,
                output=args.output.resolve(),
            )
        elif args.command == "seal-evidence":
            seal_evidence(
                args.directory.resolve(),
                _load_identity(args.identity),
                workflow_sha=args.workflow_sha,
                companion_archive_sha256=args.companion_archive_sha256,
                companion_descriptor_sha256=args.companion_descriptor_sha256,
                output=args.output.resolve(),
            )
        else:
            verify_evidence(
                args.directory.resolve(),
                workflow_sha=args.workflow_sha,
                proof_sha256=args.proof_sha256,
                attestation_sha256=args.attestation_sha256,
                source_commit=args.source_commit,
                companion_archive_sha256=args.companion_archive_sha256,
                companion_descriptor_sha256=args.companion_descriptor_sha256,
            )
    except (ProofError, OSError, zipfile.BadZipFile) as exc:
        print(f"TRUSTED_PRECUTOVER_PROOF=FAIL:{exc}")
        return 1
    print("TRUSTED_PRECUTOVER_PROOF=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
