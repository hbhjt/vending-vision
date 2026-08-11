from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys


ROOT = Path(__file__).parents[1]
SIGNER = ROOT / ".github" / "workflows" / "trusted-ai-candidate-signer.yml"
SOURCE_APPROVAL = ROOT / "scripts" / "approve_candidate_source.py"
VERIFY_INPUTS = ROOT / "scripts" / "verify_trusted_candidate_inputs.py"
GENERATE_EVIDENCE = ROOT / "scripts" / "generate_trusted_candidate_evidence.py"


def _workflow_call_inputs(source: str) -> set[str]:
    match = re.search(
        r"(?ms)^  workflow_call:\n    inputs:\n(?P<body>.*?)(?=^    outputs:)", source
    )
    assert match, "trusted signer must declare workflow_call inputs and outputs"
    return set(re.findall(r"(?m)^      ([a-z][a-z0-9_]*):$", match.group("body")))


def test_trusted_signer_has_only_data_inputs_and_isolates_the_supplier_key():
    workflow = SIGNER.read_text("utf-8")

    assert _workflow_call_inputs(workflow) == {
        "source_commit",
        "source_ref",
        "artifact_name",
        "artifact_file",
        "subject_sha256",
        "manifest_sha256",
        "attestation_bundle_file",
        "attestation_bundle_sha256",
        "builder_evidence_file",
    }
    assert "environment: experimental-candidate" in workflow
    assert "runs-on: windows-latest" in workflow
    assert "repository: ${{ job.workflow_repository }}" in workflow
    assert "ref: ${{ job.workflow_sha }}" in workflow
    assert "path: trusted-signer" in workflow
    assert "source_commit" in workflow and "source_ref" in workflow
    assert '"${{ inputs.source_commit }}" -cne "${{ github.sha }}"' in workflow
    assert '"${{ inputs.source_ref }}" -cne "${{ github.ref }}"' in workflow
    assert "--repo \"hbhjt/vending-vision\"" in workflow
    assert "--signer-repo" not in workflow
    assert "--deny-self-hosted-runners" in workflow
    assert "VISION_SUPPLIER_PRIVATE_KEY_PEM" in workflow
    assert "actions/checkout@v4" in workflow
    assert workflow.count("actions/checkout@v4") == 1
    assert "path: source" not in workflow
    assert "actions/setup-python" not in workflow
    for forbidden in (
        "artifact_path",
        "source_path",
        "custom_command",
        "predicate",
        "private_key",
        "source/scripts",
        "source/*.py",
        "*.spec",
        "*.exe",
    ):
        assert forbidden not in workflow


def _git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=directory, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_source_approval_rejects_an_attested_commit_outside_protected_main(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.email", "stage26@example.test")
    _git(repository, "config", "user.name", "Stage 26")
    (repository / "tracked.txt").write_text("approved\n", "utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "approved")
    approved = _git(repository, "rev-parse", "HEAD")
    _git(repository, "checkout", "--orphan", "unapproved")
    (repository / "tracked.txt").write_text("fake exact JSON worker\n", "utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "unapproved")
    unapproved = _git(repository, "rev-parse", "HEAD")
    _git(repository, "tag", "v0.2.1-rc.1", unapproved)
    git_dir = repository / ".git"

    rejected = subprocess.run(
        [
            sys.executable,
            str(SOURCE_APPROVAL),
            "--git-dir",
            str(git_dir),
            "--source-commit",
            unapproved,
            "--source-ref",
            "refs/tags/v0.2.1-rc.1",
            "--protected-main",
            "main",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "not an ancestor" in rejected.stdout

    _git(repository, "tag", "-f", "v0.2.1-rc.1", approved)
    accepted = subprocess.run(
        [
            sys.executable,
            str(SOURCE_APPROVAL),
            "--git-dir",
            str(git_dir),
            "--source-commit",
            approved,
            "--source-ref",
            "refs/tags/v0.2.1-rc.1",
            "--protected-main",
            "main",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr


def test_trusted_signer_generates_bound_evidence_from_zip_and_approved_git_data(tmp_path):
    from scripts.candidate_artifact_manifest import write_candidate_archive

    source_commit = _git(ROOT, "rev-parse", "HEAD")
    dist = tmp_path / "dist"
    main = dist / "vending-vision" / "vending-vision.exe"
    worker = dist / "vending-vision-ai-worker" / "vending-vision-ai-worker.exe"
    internal = worker.parent / "_internal"
    main.parent.mkdir(parents=True)
    internal.mkdir(parents=True)
    main.write_bytes(b"trusted main fixture")
    worker.write_bytes(b"trusted worker fixture")
    for relative in (
        "requirements-ai.lock.json",
        "ai-runtime-descriptor.json",
        "official-ai-source-descriptor.json",
        "official-ai-model-pack-descriptor.json",
    ):
        payload = subprocess.run(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        (internal / relative).write_bytes(payload)

    builder_input = tmp_path / "builder-input"
    builder_input.mkdir()
    artifact = builder_input / "candidate.zip"
    manifest = builder_input / "candidate-manifest.json"
    identity = write_candidate_archive(
        dist, artifact, manifest, source_commit=source_commit
    )
    bundle = builder_input / "github-build-provenance.sigstore.json"
    bundle.write_text('{"test":"verified-attestation-fixture"}', "utf-8")
    bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    builder_evidence = builder_input / "trusted-builder-evidence.json"
    builder_evidence.write_text(
        json.dumps(
            {
                "schemaVersion": "vending-vision-trusted-builder-evidence/v1",
                "builderRepository": "hbhjt/vending-vision",
                "builderWorkflow": ".github/workflows/trusted-ai-candidate-builder.yml",
                "builderWorkflowSha": "fbb97d16f42b2c20a04831750c639fda6db1a3e9",
                "sourceCommit": source_commit,
                "subjectSha256": identity["subjectSha256"],
                "embeddedManifestSha256": identity["embeddedManifestSha256"],
                "attestationBundleSha256": bundle_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "utf-8",
    )
    verified = tmp_path / "verified"
    verification = subprocess.run(
        [
            sys.executable,
            str(VERIFY_INPUTS),
            "--artifact", str(artifact),
            "--candidate-manifest", str(manifest),
            "--github-attestation", str(bundle),
            "--trusted-builder-evidence", str(builder_evidence),
            "--destination", str(verified),
            "--subject-sha256", identity["subjectSha256"],
            "--manifest-sha256", identity["embeddedManifestSha256"],
            "--attestation-bundle-sha256", bundle_sha,
            "--source-commit", source_commit,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verification.returncode == 0, verification.stdout + verification.stderr

    output = tmp_path / "signed-evidence"
    generation = subprocess.run(
        [
            sys.executable,
            str(GENERATE_EVIDENCE),
            "--bundle", str(artifact),
            "--candidate-manifest", str(manifest),
            "--github-attestation", str(bundle),
            "--trusted-builder-evidence", str(builder_evidence),
            "--verified-root", str(verified),
            "--git-dir", str(ROOT / ".git"),
            "--source-commit", source_commit,
            "--source-ref", "refs/tags/v0.2.1-rc.1",
            "--version", "0.2.1-rc.1",
            "--repository", "hbhjt/vending-vision",
            "--signer-identity", "spki-sha256:" + "a" * 64,
            "--output", str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert generation.returncode == 0, generation.stdout + generation.stderr
    descriptor = json.loads((output / "vision-release-descriptor.json").read_text("utf-8"))
    supplier = json.loads((output / "vision-artifact-attestation.json").read_text("utf-8"))
    sbom = json.loads((output / "vision-sbom.spdx.json").read_text("utf-8"))
    assert descriptor["sourceApproval"] == {
        "repository": "hbhjt/vending-vision",
        "commit": source_commit,
        "ref": "refs/tags/v0.2.1-rc.1",
        "protectedMainAncestor": True,
    }
    assert supplier["approvedSourceCommit"] == source_commit
    assert supplier["approvedSourceRef"] == "refs/tags/v0.2.1-rc.1"
    assert {item["comment"].split(";", 1)[0] for item in sbom["packages"]} == {
        "scope=core-runtime",
        "scope=ai-worker-runtime",
    }
