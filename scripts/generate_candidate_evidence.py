from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from vision.v2_contract_bundle import load_v2_contract_identity

try:  # direct ``python scripts/...`` and package import both remain supported
    from scripts.dependency_lock import verify_dependency_closure
except ModuleNotFoundError:  # pragma: no cover - exercised by release workflow
    from dependency_lock import verify_dependency_closure


SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?$")


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
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--signer-identity", required=True)
    parser.add_argument("--requirements-lock", default="requirements.txt")
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not SEMVER.fullmatch(args.version) or "-rc." not in args.version:
        raise SystemExit("candidate version must be prerelease SemVer containing -rc.")
    if not re.fullmatch(r"spki-sha256:[a-f0-9]{64}", args.signer_identity):
        raise SystemExit("signer identity must be spki-sha256:<64 lowercase hex>")

    root = Path(__file__).resolve().parents[1]
    bundle = Path(args.bundle).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_manifest = json.loads((root / "models/model-manifest.json").read_text(encoding="utf-8"))
    contract_identity = load_v2_contract_identity()
    requirements = verify_dependency_closure(args.requirements_lock, args.wheelhouse, args.python)

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
                "externalRefs": [{
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{requirement['name']}@{requirement['version']}",
                }],
            }
            for index, requirement in enumerate(requirements, start=1)
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
        "signerIdentity": args.signer_identity,
    }
    write_json(output / "vision-artifact-attestation.json", attestation)


if __name__ == "__main__":
    main()
