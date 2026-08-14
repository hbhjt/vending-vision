from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import pytest

from scripts.workflow_yaml import load_workflow_yaml


ROOT = Path(__file__).parents[1]
TRUSTED_BUILDER = ROOT / ".github" / "workflows" / "trusted-ai-candidate-builder.yml"
PUBLISHER = ROOT / ".github" / "workflows" / "publish-candidate.yml"
TRUST_POLICY = ROOT / "scripts" / "check_trusted_candidate_workflows.py"
TRUSTED_BUILDER_COMMIT = "691b5056e8b9bf2667bc527b2170780b05863946"
sys.path.insert(0, str(ROOT / "scripts"))


def _check_policy(publisher: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TRUST_POLICY),
            "--builder",
            str(TRUSTED_BUILDER),
            "--publisher",
            str(publisher),
            "--repository-root",
            str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")},
    )


def test_trusted_builder_remains_the_immutable_full_verifier_and_oidc_evidence_owner():
    source = TRUSTED_BUILDER.read_text("utf-8")
    assert f"trusted-ai-candidate-builder.yml@{TRUSTED_BUILDER_COMMIT}" in PUBLISHER.read_text("utf-8")
    assert source.index("--require-ai-worker") < source.index("actions/attest-build-provenance@v4")
    assert source.index("actions/attest-build-provenance@v4") < source.index("Record trusted builder evidence")
    assert "id-token: write" in source
    assert "attestations: write" in source
    assert "trusted-builder-evidence.json" in source


def test_publisher_is_an_exact4_only_builder_artifact_consumer():
    """The publisher may release only the four immutable builder outputs."""
    source = PUBLISHER.read_text("utf-8")
    workflow = load_workflow_yaml(source)
    assert set(workflow["jobs"]) == {"trusted_builder", "publish"}

    publish = workflow["jobs"]["publish"]
    assert publish["needs"] == "trusted_builder"
    assert publish["environment"] == "production"
    assert source.count("actions/download-artifact@v4") == 1
    assert "verify_trusted_candidate_inputs.py" in source
    assert "gh attestation verify" in source
    assert source.count("gh release create ") == 1
    for member in (
        "$env:ARTIFACT_FILE",
        "candidate-manifest.json",
        "github-build-provenance.sigstore.json",
        "trusted-builder-evidence.json",
    ):
        assert member in source
    for forbidden in (
        "trusted-ai-candidate-signer.yml",
        "sign_evidence:",
        "VISION_SUPPLIER_",
        "secrets.",
        "path: source",
        "refs/tags/",
        "refs/heads/main",
        "signed-evidence",
        "release/*",
    ):
        assert forbidden not in source


def test_exact4_policy_accepts_the_publisher():
    completed = _check_policy(PUBLISHER)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    ("name", "old", "new", "error"),
    [
        ("verify-job", "  publish:\n", "  verify:\n    runs-on: windows-latest\n    steps: []\n\n  publish:\n", "publisher_jobs_exact4"),
        ("signer-job", "  publish:\n", "  trusted_signer:\n    uses: owner/repo/.github/workflows/signer.yml@deadbeef\n\n  publish:\n", "publisher_jobs_exact4"),
        ("direct-sign-job", "  publish:\n", "  sign_evidence:\n    runs-on: windows-latest\n    steps: []\n\n  publish:\n", "publisher_jobs_exact4"),
        ("supplier-secret", "    environment: production\n", "    environment: production\n    env:\n      KEY: ${{ secrets.VISION_SUPPLIER_PRIVATE_KEY_PEM }}\n", "publisher_forbidden_capability"),
        ("bracket-secret", "    environment: production\n", "    environment: production\n    env:\n      KEY: ${{ secrets['REINTRODUCED_SECRET'] }}\n", "publisher_forbidden_capability"),
        ("wrong-environment", "environment: production", "environment: experimental-candidate", "publisher_production_environment"),
        ("source-checkout", "path: trusted-builder", "path: source", "publisher_forbidden_capability"),
        ("sidecar", "trusted-builder-evidence.json", "supplier-signed-evidence.json", "publisher_forbidden_capability"),
        ("missing-artifact", "candidate-manifest.json", "candidate-manifest-missing.json", "publisher_exact4_member"),
        ("wrong-digest", "SUBJECT_SHA256: ${{ needs.trusted_builder.outputs.subject_sha256 }}", "SUBJECT_SHA256: ${{ needs.trusted_builder.outputs.manifest_sha256 }}", "publisher_output_binding"),
        ("publishes-without-builder", "needs: trusted_builder", "needs: []", "publisher_requires_builder"),
    ],
)
def test_exact4_policy_rejects_reintroduced_authority_or_invalid_input_binding(
    tmp_path, name, old, new, error
):
    source = PUBLISHER.read_text("utf-8")
    assert old in source
    candidate = tmp_path / f"{name}.yml"
    candidate.write_text(source.replace(old, new, 1), "utf-8")

    completed = _check_policy(candidate)

    assert completed.returncode != 0
    assert error in completed.stdout


def test_exact4_policy_rejects_glob_or_extra_release_assets(tmp_path):
    source = PUBLISHER.read_text("utf-8")
    candidate = tmp_path / "glob.yml"
    candidate.write_text(
        source.replace(
            '"candidate-input/trusted-builder-evidence.json"',
            '"candidate-input/*"',
            1,
        ),
        "utf-8",
    )

    completed = _check_policy(candidate)

    assert completed.returncode != 0
    assert "publisher_release_assets_exact4" in completed.stdout


def test_exact4_policy_keeps_the_691_builder_closure_and_rejects_run_interpolation(tmp_path):
    changed_builder = tmp_path / "trusted-ai-candidate-builder.yml"
    changed_builder.write_text(TRUSTED_BUILDER.read_text("utf-8") + "\n# drift\n", "utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(TRUST_POLICY),
            "--builder", str(changed_builder),
            "--publisher", str(PUBLISHER),
            "--repository-root", str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")},
    )
    assert completed.returncode != 0
    assert "trusted_builder_bytes_changed" in completed.stdout

    injected = tmp_path / "injected.yml"
    injected.write_text(
        PUBLISHER.read_text("utf-8").replace(
            "run: |\n          if ($env:ARTIFACT_FILE",
            'run: |\n          Write-Output "${{ needs.trusted_builder.outputs.artifact_file }}"\n          if ($env:ARTIFACT_FILE',
            1,
        ),
        "utf-8",
    )
    completed = _check_policy(injected)
    assert completed.returncode != 0
    assert "publisher_workflow_expression_in_run" in completed.stdout


def _exact4_fixture(root: Path) -> dict[str, object]:
    from scripts.candidate_artifact_manifest import BINDING_PATHS, write_candidate_archive

    source_commit = "a" * 40
    dist = root / "dist"
    for index, relative in enumerate(BINDING_PATHS.values()):
        path = dist / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"payload-{index}".encode())
    inputs = root / "inputs"
    artifact = inputs / "vending-vision-0.2.1-rc.1-windows-x86_64.zip"
    manifest = inputs / "candidate-manifest.json"
    digests = write_candidate_archive(dist, artifact, manifest, source_commit=source_commit)
    bundle = inputs / "github-build-provenance.sigstore.json"
    bundle.write_text("oidc bundle", "utf-8")
    evidence = inputs / "trusted-builder-evidence.json"
    bundle_digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    evidence.write_text(json.dumps({
        "schemaVersion": "vending-vision-trusted-builder-evidence/v1",
        "builderRepository": "hbhjt/vending-vision",
        "builderWorkflow": ".github/workflows/trusted-ai-candidate-builder.yml",
        "builderWorkflowSha": TRUSTED_BUILDER_COMMIT,
        "sourceCommit": source_commit,
        "subjectSha256": digests["subjectSha256"],
        "embeddedManifestSha256": digests["embeddedManifestSha256"],
        "attestationBundleSha256": bundle_digest,
    }), "utf-8")
    return {
        "artifact": artifact,
        "manifest": manifest,
        "bundle": bundle,
        "evidence": evidence,
        "source_commit": source_commit,
        "subject": digests["subjectSha256"],
        "manifest_digest": digests["embeddedManifestSha256"],
        "bundle_digest": bundle_digest,
    }


def test_public_exact4_verifier_rejects_missing_inputs_wrong_digest_and_source_extra(tmp_path):
    from scripts.verify_trusted_candidate_inputs import verify_inputs

    for mutation, expected in (("missing", "missing"), ("digest", "digest"), ("source", "member set")):
        fixture = _exact4_fixture(tmp_path / mutation)
        if mutation == "missing":
            fixture["evidence"].unlink()
        elif mutation == "digest":
            fixture["subject"] = "0" * 64
        else:
            (fixture["artifact"].parent / "source").mkdir()
        with pytest.raises(AssertionError, match=expected):
            verify_inputs(
                artifact=fixture["artifact"],
                candidate_manifest=fixture["manifest"],
                github_attestation=fixture["bundle"],
                trusted_builder_evidence=fixture["evidence"],
                destination=tmp_path / mutation / "extracted",
                subject_sha256=fixture["subject"],
                manifest_sha256=fixture["manifest_digest"],
                attestation_bundle_sha256=fixture["bundle_digest"],
                source_commit=fixture["source_commit"],
            )
