from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from candidate_artifact_manifest import EMBEDDED_MANIFEST, verify_candidate_archive


TRUSTED_REPOSITORY = "hbhjt/vending-vision"
TRUSTED_BUILDER_COMMIT = "c90a965d117fea49f318b18e0fcd50aa047bc41"
TRUSTED_BUILDER_WORKFLOW = ".github/workflows/trusted-ai-candidate-builder.yml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: str, name: str) -> None:
    if re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise AssertionError(f"{name} digest is invalid")


def verify_inputs(
    *,
    artifact: Path,
    candidate_manifest: Path,
    github_attestation: Path,
    trusted_builder_evidence: Path,
    destination: Path,
    subject_sha256: str,
    manifest_sha256: str,
    attestation_bundle_sha256: str,
    source_commit: str,
) -> None:
    for value, name in (
        (subject_sha256, "subject"),
        (manifest_sha256, "manifest"),
        (attestation_bundle_sha256, "attestation bundle"),
    ):
        _require_digest(value, name)
    if re.fullmatch(r"[a-f0-9]{40}", source_commit) is None:
        raise AssertionError("source commit is invalid")
    paths = (artifact, candidate_manifest, github_attestation, trusted_builder_evidence)
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        raise AssertionError("trusted builder input is missing or not a regular file")
    input_root = artifact.parent.resolve()
    if any(path.parent.resolve() != input_root for path in paths):
        raise AssertionError("trusted builder inputs must share one artifact directory")
    expected_names = {path.name for path in paths}
    actual_names = {path.name for path in input_root.iterdir()}
    if actual_names != expected_names:
        raise AssertionError("trusted builder artifact member set mismatch")
    if sha256_file(github_attestation) != attestation_bundle_sha256:
        raise AssertionError("attestation bundle digest mismatch")
    if sha256_file(candidate_manifest) != manifest_sha256:
        raise AssertionError("external candidate manifest digest mismatch")

    verify_candidate_archive(
        artifact,
        destination,
        expected_subject_sha256=subject_sha256,
        expected_manifest_sha256=manifest_sha256,
        expected_source_commit=source_commit,
    )
    with zipfile.ZipFile(artifact) as archive:
        if archive.read(EMBEDDED_MANIFEST) != candidate_manifest.read_bytes():
            raise AssertionError("external and embedded candidate manifests differ")

    try:
        evidence = json.loads(trusted_builder_evidence.read_text("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise AssertionError("trusted builder evidence is invalid") from exc
    expected_evidence = {
        "schemaVersion": "vending-vision-trusted-builder-evidence/v1",
        "builderRepository": TRUSTED_REPOSITORY,
        "builderWorkflow": TRUSTED_BUILDER_WORKFLOW,
        "builderWorkflowSha": TRUSTED_BUILDER_COMMIT,
        "sourceCommit": source_commit,
        "subjectSha256": subject_sha256,
        "embeddedManifestSha256": manifest_sha256,
        "attestationBundleSha256": attestation_bundle_sha256,
    }
    if evidence != expected_evidence:
        raise AssertionError("trusted builder evidence binding mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--github-attestation", required=True, type=Path)
    parser.add_argument("--trusted-builder-evidence", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--subject-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--attestation-bundle-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    try:
        verify_inputs(
            artifact=args.artifact.resolve(),
            candidate_manifest=args.candidate_manifest.resolve(),
            github_attestation=args.github_attestation.resolve(),
            trusted_builder_evidence=args.trusted_builder_evidence.resolve(),
            destination=args.destination.resolve(),
            subject_sha256=args.subject_sha256,
            manifest_sha256=args.manifest_sha256,
            attestation_bundle_sha256=args.attestation_bundle_sha256,
            source_commit=args.source_commit,
        )
    except (AssertionError, OSError, zipfile.BadZipFile) as exc:
        print(f"TRUSTED_CANDIDATE_INPUTS=FAIL:{exc}")
        return 1
    print("TRUSTED_CANDIDATE_INPUTS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
