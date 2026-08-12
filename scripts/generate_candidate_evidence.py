from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# A release operator invokes this file as ``python scripts/...``.  In that
# mode Python puts only ``scripts/`` on sys.path, so establish the repository
# package root before importing generated contract code.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.v2_contract_bundle import load_v2_contract_identity


def verify_dependency_closure(*args, **kwargs):
    """Load packaging-only dependencies after CLI parsing has succeeded."""
    from scripts.dependency_lock import verify_dependency_closure as verify

    return verify(*args, **kwargs)


SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?$")
TRUSTED_BUILDER_COMMIT = "c90a965d117fea49f318b18e0fcd50aa047bc41"
TRUSTED_BUILDER_WORKFLOW = ".github/workflows/trusted-ai-candidate-builder.yml"


def digest_bytes(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_file(path):
    return digest_bytes(path.read_bytes())


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path, value):
    path.write_bytes(canonical_bytes(value) + b"\n")


def evidence_ref(path, **extra):
    digest = digest_file(path)
    return {
        "identity": f"factory-evidence://sha256/{digest.removeprefix('sha256:')}",
        "digest": digest,
        **extra,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--github-attestation", required=True)
    parser.add_argument("--trusted-builder-evidence", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--signer-identity", required=True)
    parser.add_argument("--requirements-lock", default="requirements.txt")
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--ai-requirements-lock", required=True)
    parser.add_argument("--ai-runtime-descriptor", required=True)
    parser.add_argument("--source-descriptor", required=True)
    parser.add_argument("--model-pack-descriptor", required=True)
    parser.add_argument("--worker-executable", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not SEMVER.fullmatch(args.version) or "-rc." not in args.version:
        raise SystemExit("candidate version must be prerelease SemVer containing -rc.")
    if re.fullmatch(r"[a-f0-9]{40}", args.commit) is None:
        raise SystemExit("candidate commit must be a full lowercase SHA")
    if not re.fullmatch(r"spki-sha256:[a-f0-9]{64}", args.signer_identity):
        raise SystemExit("signer identity must be spki-sha256:<64 lowercase hex>")

    bundle = Path(args.bundle).resolve()
    candidate_manifest = Path(args.candidate_manifest).resolve()
    github_attestation = Path(args.github_attestation).resolve()
    trusted_builder_evidence = Path(args.trusted_builder_evidence).resolve()
    ai_requirements_lock = Path(args.ai_requirements_lock).resolve()
    ai_runtime_descriptor = Path(args.ai_runtime_descriptor).resolve()
    source_descriptor = Path(args.source_descriptor).resolve()
    model_pack_descriptor = Path(args.model_pack_descriptor).resolve()
    worker_executable = Path(args.worker_executable).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_manifest = json.loads((ROOT / "models/model-manifest.json").read_text(encoding="utf-8"))
    contract_identity = load_v2_contract_identity()
    requirements = verify_dependency_closure(args.requirements_lock, args.wheelhouse, args.python)
    required_ai_files = (
        bundle, candidate_manifest, github_attestation, trusted_builder_evidence,
        ai_requirements_lock, ai_runtime_descriptor,
        source_descriptor, model_pack_descriptor, worker_executable,
    )
    if not all(path.is_file() for path in required_ai_files):
        raise SystemExit("candidate AI evidence input missing")
    ai_lock_raw = ai_requirements_lock.read_text("utf-8")
    ai_lock = json.loads(ai_lock_raw)
    if canonical_bytes(ai_lock) != ai_lock_raw.rstrip("\n").encode("utf-8"):
        raise SystemExit("AI requirements lock must be canonical")
    ai_wheels = ai_lock.get("wheels")
    if not isinstance(ai_wheels, list) or not ai_wheels:
        raise SystemExit("AI requirements lock has no wheels")
    runtime = json.loads(ai_runtime_descriptor.read_text("utf-8"))
    if runtime.get("requirementsAiLockSha256") != digest_file(ai_requirements_lock).split(":", 1)[1]:
        raise SystemExit("AI runtime descriptor does not bind AI lock")
    candidate = json.loads(candidate_manifest.read_text("utf-8"))
    if candidate.get("schemaVersion") != "vending-vision-candidate-artifact/v3":
        raise SystemExit("candidate embedded manifest schema mismatch")
    bindings = candidate.get("bindings") or {}
    bound_inputs = {
        "workerExecutable": worker_executable,
        "runtimeDescriptor": ai_runtime_descriptor,
        "aiLock": ai_requirements_lock,
        "sourceDescriptor": source_descriptor,
        "modelPackDescriptor": model_pack_descriptor,
    }
    for name, path in bound_inputs.items():
        binding = bindings.get(name)
        if not isinstance(binding, dict) or binding.get("sha256") != digest_file(path).split(":", 1)[1]:
            raise SystemExit(f"candidate manifest does not bind {name}")
    try:
        builder_evidence = json.loads(trusted_builder_evidence.read_text("utf-8"))
    except ValueError as exc:
        raise SystemExit("trusted builder evidence is invalid") from exc
    if set(builder_evidence) != {
        "schemaVersion", "builderRepository", "builderWorkflow", "builderWorkflowSha",
        "sourceCommit", "subjectSha256", "embeddedManifestSha256",
        "attestationBundleSha256",
    }:
        raise SystemExit("trusted builder evidence shape mismatch")
    expected_builder_evidence = {
        "schemaVersion": "vending-vision-trusted-builder-evidence/v1",
        "builderRepository": args.repository,
        "builderWorkflow": TRUSTED_BUILDER_WORKFLOW,
        "builderWorkflowSha": TRUSTED_BUILDER_COMMIT,
        "sourceCommit": args.commit,
        "subjectSha256": digest_file(bundle).split(":", 1)[1],
        "embeddedManifestSha256": digest_file(candidate_manifest).split(":", 1)[1],
        "attestationBundleSha256": digest_file(github_attestation).split(":", 1)[1],
    }
    if builder_evidence != expected_builder_evidence:
        raise SystemExit("trusted builder evidence binding mismatch")

    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"vending-vision-{args.version}",
        "documentNamespace": f"https://github.com/{args.repository}/releases/{args.version}/{digest_file(bundle).split(':')[1]}",
        "creationInfo": {"creators": ["Tool: vending-vision-candidate-builder"], "created": "1970-01-01T00:00:00Z"},
        "packages": [
            {
                "name": requirement["name"],
                "SPDXID": f"SPDXRef-Package-{index}",
                "versionInfo": requirement["version"],
                "downloadLocation": f"pkg:pypi/{requirement['name']}@{requirement['version']}",
                "filesAnalyzed": False,
                "checksums": [{"algorithm": "SHA256", "checksumValue": requirement["wheel"]["sha256"]}],
                "licenseConcluded": requirement["license"],
                "licenseDeclared": requirement["license"],
                "comment": "scope=core-runtime",
                "externalRefs": [{
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{requirement['name']}@{requirement['version']}",
                }],
            }
            for index, requirement in enumerate(requirements, start=1)
        ] + [
            {
                "name": wheel["name"],
                "SPDXID": f"SPDXRef-AI-Package-{index}",
                "versionInfo": wheel["version"],
                "downloadLocation": wheel["url"],
                "filesAnalyzed": False,
                "checksums": [{"algorithm": "SHA256", "checksumValue": wheel["sha256"]}],
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "comment": "scope=ai-worker-runtime",
                "externalRefs": [{
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{wheel['name']}@{wheel['version']}",
                }],
            }
            for index, wheel in enumerate(ai_wheels, start=1)
        ],
        "files": [
            {
                "fileName": model["path"],
                "SPDXID": f"SPDXRef-Model-{index}",
                "checksums": [{"algorithm": "SHA256", "checksumValue": model["sha256"]}],
                "licenseConcluded": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
            for index, model in enumerate(model_manifest["models"], start=1)
        ] + [
            {
                "fileName": path.name,
                "SPDXID": f"SPDXRef-AI-Bound-{index}",
                "checksums": [{"algorithm": "SHA256", "checksumValue": digest_file(path).split(":", 1)[1]}],
                "licenseConcluded": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
            for index, path in enumerate(
                (worker_executable, ai_runtime_descriptor, ai_requirements_lock, source_descriptor, model_pack_descriptor),
                start=1,
            )
        ],
    }
    sbom_path = output / "vision-sbom.spdx.json"
    write_json(sbom_path, sbom)

    provenance = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": bundle.name, "digest": {"sha256": digest_file(bundle).split(":", 1)[1]}}],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://github.com/vending-vision/windows-candidate/v1",
                "externalParameters": {"releaseVersion": args.version},
                "internalParameters": {},
                "resolvedDependencies": [
                    {"uri": f"git+https://github.com/{args.repository}@{args.commit}", "digest": {"gitCommit": args.commit}},
                    {"uri": "legacy-bundle:vending-vision.zip", "digest": {"sha256": model_manifest["sourceArtifact"]["digest"].split(":", 1)[1]}},
                    {"uri": f"candidate-manifest:{candidate_manifest.name}", "digest": {"sha256": digest_file(candidate_manifest).split(":", 1)[1]}},
                    {"uri": f"github-attestation:{github_attestation.name}", "digest": {"sha256": digest_file(github_attestation).split(":", 1)[1]}},
                    {"uri": f"trusted-builder-evidence:{trusted_builder_evidence.name}", "digest": {"sha256": digest_file(trusted_builder_evidence).split(":", 1)[1]}},
                    {"uri": f"ai-lock:{ai_requirements_lock.name}", "digest": {"sha256": digest_file(ai_requirements_lock).split(":", 1)[1]}},
                    {"uri": f"ai-runtime:{ai_runtime_descriptor.name}", "digest": {"sha256": digest_file(ai_runtime_descriptor).split(":", 1)[1]}},
                    {"uri": f"ai-source:{source_descriptor.name}", "digest": {"sha256": digest_file(source_descriptor).split(":", 1)[1]}},
                    {"uri": f"ai-model-pack:{model_pack_descriptor.name}", "digest": {"sha256": digest_file(model_pack_descriptor).split(":", 1)[1]}},
                    {"uri": f"ai-worker:{worker_executable.name}", "digest": {"sha256": digest_file(worker_executable).split(":", 1)[1]}},
                ],
            },
            "runDetails": {
                "builder": {"id": "https://github.com/actions/runner/windows"},
                "metadata": {"invocationId": args.commit},
            },
        },
    }
    provenance_path = output / "vision-provenance.json"
    write_json(provenance_path, provenance)

    descriptor_without_identity = {
        "schemaVersion": "vem-vision-release-descriptor/v1",
        "kind": "vision-release-descriptor",
        "releaseVersion": args.version,
        "bundle": {
            "digest": digest_file(bundle),
            "bytes": bundle.stat().st_size,
            "platform": {"os": "windows", "architecture": "x86_64"},
            "format": "zip",
            "extractor": {"contractVersion": "vem-vision-extractor/v1", "handler": "zip-safe-v1"},
        },
        "entrypoint": {"command": "vending-vision/vending-vision.exe", "arguments": ["--no-browser"]},
        "candidateManifest": evidence_ref(candidate_manifest, schemaVersion="vending-vision-candidate-artifact/v3"),
        "githubArtifactAttestation": evidence_ref(github_attestation, format="sigstore-bundle"),
        "trustedBuilderEvidence": evidence_ref(
            trusted_builder_evidence,
            schemaVersion="vending-vision-trusted-builder-evidence/v1",
        ),
        "aiRuntime": {
            "requirementsLock": evidence_ref(ai_requirements_lock),
            "runtimeDescriptor": evidence_ref(ai_runtime_descriptor),
            "sourceDescriptor": evidence_ref(source_descriptor),
            "modelPackDescriptor": evidence_ref(model_pack_descriptor),
            "workerExecutable": evidence_ref(worker_executable),
        },
        "lifecycle": {"requiresInteractiveSession": True, "shutdownTimeoutMs": 10000},
        "configuration": {"format": "json", "schemaVersion": "vending-vision-site-config/v1", "argument": "--config"},
        "health": {"port": 7892, "path": "/health", "expectedStatus": 200, "timeoutMs": 30000},
        "protocol": {
            "version": contract_identity.protocol,
            "schemaVersion": contract_identity.schema_version,
            "bundleVersion": contract_identity.bundle_version,
            "contractDigest": contract_identity.contract_digest,
            "webSocketPath": "/ws",
        },
        "sbom": evidence_ref(sbom_path, format="spdx-json"),
        "provenance": evidence_ref(provenance_path, predicateType="https://slsa.dev/provenance/v1"),
    }
    descriptor = {
        **descriptor_without_identity,
        "identity": digest_bytes(canonical_bytes(descriptor_without_identity)),
    }
    descriptor_path = output / "vision-release-descriptor.json"
    write_json(descriptor_path, descriptor)

    attestation = {
        "schemaVersion": "vem-vision-artifact-attestation/v1",
        "kind": "vision-artifact-attestation",
        "bundleDigest": descriptor["bundle"]["digest"],
        "descriptorDigest": descriptor["identity"],
        "sbomDigest": descriptor["sbom"]["digest"],
        "provenanceDigest": descriptor["provenance"]["digest"],
        "candidateManifestDigest": descriptor["candidateManifest"]["digest"],
        "githubArtifactAttestationDigest": descriptor["githubArtifactAttestation"]["digest"],
        "trustedBuilderEvidenceDigest": descriptor["trustedBuilderEvidence"]["digest"],
        "attestedSubjectDigest": descriptor["bundle"]["digest"],
        "workerExecutableDigest": descriptor["aiRuntime"]["workerExecutable"]["digest"],
        "signerIdentity": args.signer_identity,
    }
    write_json(output / "vision-artifact-attestation.json", attestation)


if __name__ == "__main__":
    main()
