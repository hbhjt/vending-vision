from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


SEMVER_RC = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-rc\.[0-9A-Za-z.-]+$"
)
FULL_SHA = re.compile(r"^[a-f0-9]{40}$")
SIGNER_IDENTITY = re.compile(r"^spki-sha256:[a-f0-9]{64}$")
TRUSTED_BUILDER_COMMIT = "691b5056e8b9bf2667bc527b2170780b05863946"
TRUSTED_BUILDER_WORKFLOW = ".github/workflows/trusted-ai-candidate-builder.yml"
SOURCE_FILES = (
    "requirements.txt",
    "requirements-ai.lock.json",
    "ai-runtime-descriptor.json",
    "official-ai-source-descriptor.json",
    "official-ai-model-pack-descriptor.json",
    "models/model-manifest.json",
    "contracts/vem_vision_v2/manifest.json",
)
AI_PAYLOADS = {
    "requirements-ai.lock.json": "vending-vision-ai-worker/_internal/requirements-ai.lock.json",
    "ai-runtime-descriptor.json": "vending-vision-ai-worker/_internal/ai-runtime-descriptor.json",
    "official-ai-source-descriptor.json": "vending-vision-ai-worker/_internal/official-ai-source-descriptor.json",
    "official-ai-model-pack-descriptor.json": "vending-vision-ai-worker/_internal/official-ai-model-pack-descriptor.json",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_ref(path: Path, **extra: object) -> dict[str, object]:
    digest = sha256_file(path)
    return {
        "identity": f"factory-evidence://sha256/{digest}",
        "digest": f"sha256:{digest}",
        **extra,
    }


def git_data(git_dir: Path, commit: str, relative: str) -> bytes:
    if relative not in SOURCE_FILES:
        raise AssertionError("unapproved source data path")
    completed = subprocess.run(
        ["git", "--git-dir", str(git_dir), "show", f"{commit}:{relative}"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AssertionError(f"approved source data missing: {relative}")
    return completed.stdout


def _json_data(source: dict[str, bytes], relative: str) -> object:
    try:
        return json.loads(source[relative])
    except (UnicodeError, ValueError) as exc:
        raise AssertionError(f"approved source JSON invalid: {relative}") from exc


def _core_packages(raw: bytes) -> list[dict[str, object]]:
    text = raw.decode("utf-8")
    starts = list(
        re.finditer(
            r"(?m)^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^ ;\\\r\n]+)(?:\s*;[^\\\r\n]+)?\s*\\\s*$",
            text,
        )
    )
    packages: list[dict[str, object]] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        hashes = sorted(set(re.findall(r"--hash=sha256:([a-f0-9]{64})", text[match.end():end])))
        if not hashes:
            raise AssertionError(f"core requirement has no hashes: {match.group('name')}")
        packages.append(
            {
                "name": re.sub(r"[-_.]+", "-", match.group("name")).lower(),
                "version": match.group("version"),
                "hashes": hashes,
            }
        )
    if not packages:
        raise AssertionError("core requirements lock has no packages")
    return packages


def _binding_path(manifest: dict, name: str) -> str:
    binding = (manifest.get("bindings") or {}).get(name)
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise AssertionError(f"candidate manifest binding missing: {name}")
    return binding["path"]


def generate(args: argparse.Namespace) -> None:
    if not FULL_SHA.fullmatch(args.source_commit):
        raise AssertionError("source commit must be a full lowercase SHA")
    if args.source_ref != f"refs/tags/v{args.version}" or not SEMVER_RC.fullmatch(args.version):
        raise AssertionError("source ref and RC version do not match")
    if args.repository != "hbhjt/vending-vision":
        raise AssertionError("unexpected source repository")
    if not SIGNER_IDENTITY.fullmatch(args.signer_identity):
        raise AssertionError("supplier signer identity is invalid")

    bundle = args.bundle.resolve()
    candidate_manifest_path = args.candidate_manifest.resolve()
    github_attestation = args.github_attestation.resolve()
    trusted_builder_evidence = args.trusted_builder_evidence.resolve()
    verified_root = args.verified_root.resolve()
    git_dir = args.git_dir.resolve()
    output = args.output.resolve()
    for path in (
        bundle,
        candidate_manifest_path,
        github_attestation,
        trusted_builder_evidence,
    ):
        if not path.is_file() or path.is_symlink():
            raise AssertionError("trusted evidence input missing")
    if not verified_root.is_dir() or not git_dir.is_dir():
        raise AssertionError("verified candidate or approved Git data missing")

    source = {relative: git_data(git_dir, args.source_commit, relative) for relative in SOURCE_FILES}
    manifest = json.loads(candidate_manifest_path.read_text("utf-8"))
    if canonical_bytes(manifest) != candidate_manifest_path.read_bytes():
        raise AssertionError("candidate manifest is not canonical")
    if manifest.get("sourceCommit") != args.source_commit:
        raise AssertionError("candidate manifest source commit mismatch")
    ai_lock = _json_data(source, "requirements-ai.lock.json")
    if not isinstance(ai_lock, dict) or not isinstance(ai_lock.get("wheels"), list) or not ai_lock["wheels"]:
        raise AssertionError("AI lock has no wheel closure")
    if canonical_bytes(ai_lock) != source["requirements-ai.lock.json"].rstrip(b"\n"):
        raise AssertionError("AI lock source data is not canonical")

    for relative, payload_relative in AI_PAYLOADS.items():
        payload = verified_root / payload_relative
        if not payload.is_file() or sha256_file(payload) != sha256_bytes(source[relative]):
            raise AssertionError(f"candidate payload does not match approved source data: {relative}")
    worker_relative = _binding_path(manifest, "workerExecutable")
    worker = verified_root / worker_relative
    if not worker.is_file():
        raise AssertionError("verified worker executable missing")

    builder = json.loads(trusted_builder_evidence.read_text("utf-8"))
    expected_builder = {
        "schemaVersion": "vending-vision-trusted-builder-evidence/v1",
        "builderRepository": args.repository,
        "builderWorkflow": TRUSTED_BUILDER_WORKFLOW,
        "builderWorkflowSha": TRUSTED_BUILDER_COMMIT,
        "sourceCommit": args.source_commit,
        "subjectSha256": sha256_file(bundle),
        "embeddedManifestSha256": sha256_file(candidate_manifest_path),
        "attestationBundleSha256": sha256_file(github_attestation),
    }
    if builder != expected_builder:
        raise AssertionError("trusted builder evidence mismatch")

    core_packages = _core_packages(source["requirements.txt"])
    model_manifest = _json_data(source, "models/model-manifest.json")
    contract = _json_data(source, "contracts/vem_vision_v2/manifest.json")
    if not isinstance(model_manifest, dict) or not isinstance(model_manifest.get("models"), list):
        raise AssertionError("model manifest source data is invalid")
    if not isinstance(contract, dict):
        raise AssertionError("contract source data is invalid")

    sbom_packages = [
        {
            "name": item["name"],
            "SPDXID": f"SPDXRef-Core-{index}",
            "versionInfo": item["version"],
            "downloadLocation": f"pkg:pypi/{item['name']}@{item['version']}",
            "filesAnalyzed": False,
            "checksums": [{"algorithm": "SHA256", "checksumValue": digest} for digest in item["hashes"]],
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "comment": "scope=core-runtime; source=approved-hash-lock",
        }
        for index, item in enumerate(core_packages, start=1)
    ] + [
        {
            "name": wheel["name"],
            "SPDXID": f"SPDXRef-AI-{index}",
            "versionInfo": wheel["version"],
            "downloadLocation": wheel["url"],
            "filesAnalyzed": False,
            "checksums": [{"algorithm": "SHA256", "checksumValue": wheel["sha256"]}],
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "comment": "scope=ai-worker-runtime; source=approved-wheel-lock",
        }
        for index, wheel in enumerate(ai_lock["wheels"], start=1)
    ]
    sbom_files = [
        {
            "fileName": item["path"],
            "SPDXID": f"SPDXRef-Candidate-{index}",
            "checksums": [{"algorithm": "SHA256", "checksumValue": item["sha256"]}],
            "licenseConcluded": "NOASSERTION",
            "copyrightText": "NOASSERTION",
        }
        for index, item in enumerate(manifest["files"], start=1)
    ]
    output.mkdir(parents=True, exist_ok=False)
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"vending-vision-{args.version}",
        "documentNamespace": f"https://github.com/{args.repository}/releases/{args.version}/{sha256_file(bundle)}",
        "creationInfo": {"creators": ["Tool: trusted-ai-candidate-signer"], "created": "1970-01-01T00:00:00Z"},
        "packages": sbom_packages,
        "files": sbom_files,
    }
    sbom_path = output / "vision-sbom.spdx.json"
    write_json(sbom_path, sbom)

    source_dependencies = [
        {
            "uri": f"git+https://github.com/{args.repository}@{args.source_commit}:{relative}",
            "digest": {"sha256": sha256_bytes(raw)},
        }
        for relative, raw in source.items()
    ]
    provenance = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": bundle.name, "digest": {"sha256": sha256_file(bundle)}}],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://github.com/vending-vision/trusted-supplier-signing/v1",
                "externalParameters": {
                    "releaseVersion": args.version,
                    "approvedSourceRef": args.source_ref,
                },
                "internalParameters": {},
                "resolvedDependencies": source_dependencies + [
                    {"uri": f"candidate-manifest:{candidate_manifest_path.name}", "digest": {"sha256": sha256_file(candidate_manifest_path)}},
                    {"uri": f"github-attestation:{github_attestation.name}", "digest": {"sha256": sha256_file(github_attestation)}},
                    {"uri": f"trusted-builder-evidence:{trusted_builder_evidence.name}", "digest": {"sha256": sha256_file(trusted_builder_evidence)}},
                ],
            },
            "runDetails": {
                "builder": {"id": "https://github.com/hbhjt/vending-vision/.github/workflows/trusted-ai-candidate-signer.yml"},
                "metadata": {"invocationId": args.source_commit},
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
            "digest": f"sha256:{sha256_file(bundle)}",
            "bytes": bundle.stat().st_size,
            "platform": {"os": "windows", "architecture": "x86_64"},
            "format": "zip",
            "extractor": {"contractVersion": "vem-vision-extractor/v1", "handler": "zip-safe-v1"},
        },
        "entrypoint": {"command": "vending-vision/vending-vision.exe", "arguments": ["--no-browser"]},
        "sourceApproval": {
            "repository": args.repository,
            "commit": args.source_commit,
            "ref": args.source_ref,
            "protectedMainAncestor": True,
        },
        "candidateManifest": evidence_ref(candidate_manifest_path, schemaVersion="vending-vision-candidate-artifact/v3"),
        "githubArtifactAttestation": evidence_ref(github_attestation, format="sigstore-bundle"),
        "trustedBuilderEvidence": evidence_ref(trusted_builder_evidence, schemaVersion="vending-vision-trusted-builder-evidence/v1"),
        "aiRuntime": {
            "requirementsLock": {"digest": f"sha256:{sha256_bytes(source['requirements-ai.lock.json'])}"},
            "runtimeDescriptor": {"digest": f"sha256:{sha256_bytes(source['ai-runtime-descriptor.json'])}"},
            "sourceDescriptor": {"digest": f"sha256:{sha256_bytes(source['official-ai-source-descriptor.json'])}"},
            "modelPackDescriptor": {"digest": f"sha256:{sha256_bytes(source['official-ai-model-pack-descriptor.json'])}"},
            "workerExecutable": {"digest": f"sha256:{sha256_file(worker)}"},
        },
        "lifecycle": {"requiresInteractiveSession": True, "shutdownTimeoutMs": 10000},
        "configuration": {"format": "json", "schemaVersion": "vending-vision-site-config/v1", "argument": "--config"},
        "health": {"port": 7892, "path": "/health", "expectedStatus": 200, "timeoutMs": 30000},
        "protocol": {
            "version": contract["protocol"],
            "schemaVersion": contract["schemaVersion"],
            "bundleVersion": contract["bundleVersion"],
            "contractDigest": contract["bundleDigest"],
            "webSocketPath": "/ws",
        },
        "sbom": evidence_ref(sbom_path, format="spdx-json"),
        "provenance": evidence_ref(provenance_path, predicateType="https://slsa.dev/provenance/v1"),
    }
    descriptor = {
        **descriptor_without_identity,
        "identity": f"sha256:{sha256_bytes(canonical_bytes(descriptor_without_identity))}",
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
        "approvedSourceCommit": args.source_commit,
        "approvedSourceRef": args.source_ref,
        "signerIdentity": args.signer_identity,
    }
    write_json(output / "vision-artifact-attestation.json", attestation)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--github-attestation", required=True, type=Path)
    parser.add_argument("--trusted-builder-evidence", required=True, type=Path)
    parser.add_argument("--verified-root", required=True, type=Path)
    parser.add_argument("--git-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--signer-identity", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        generate(args)
    except (AssertionError, KeyError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"TRUSTED_CANDIDATE_EVIDENCE=FAIL:{exc}")
        return 1
    print("TRUSTED_CANDIDATE_EVIDENCE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
