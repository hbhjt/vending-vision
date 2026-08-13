from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

import pytest

from scripts.verify_release_tag_ruleset import github_ref_name_matches, verify_rulesets
from scripts.verify_trusted_script_set import verify as verify_trusted_script_set
from scripts.workflow_yaml import load_workflow_yaml


ROOT = Path(__file__).parents[1]
SIGNER = ROOT / ".github" / "workflows" / "trusted-ai-candidate-signer.yml"
SOURCE_APPROVAL = ROOT / "scripts" / "approve_candidate_source.py"
VERIFY_INPUTS = ROOT / "scripts" / "verify_trusted_candidate_inputs.py"
GENERATE_EVIDENCE = ROOT / "scripts" / "generate_trusted_candidate_evidence.py"
TAG_RULESET = ROOT / "scripts" / "verify_release_tag_ruleset.py"
TRUSTED_BUILDER_COMMIT = "691b5056e8b9bf2667bc527b2170780b05863946"


def _assert_signer_environment_authority(source: str) -> None:
    jobs = load_workflow_yaml(source)["jobs"]
    verify = jobs["verify_evidence"]
    sign = jobs["sign_evidence"]
    assert "environment" not in verify
    assert sign["environment"] == "experimental-candidate"
    verify_steps = {step["name"]: step for step in verify["steps"]}
    sign_steps = {step["name"]: step for step in sign["steps"]}
    assert "VISION_SUPPLIER_PRIVATE_KEY_PEM" not in str(verify)
    assert verify_steps["Validate identities without shell interpolation"]["env"][
        "VISION_SUPPLIER_SIGNER_IDENTITY"
    ] == "${{ inputs.signer_identity }}"
    assert '"VISION_SUPPLIER_SIGNER_IDENTITY=$env:VISION_SUPPLIER_SIGNER_IDENTITY" >> $env:GITHUB_ENV' in verify_steps[
        "Validate identities without shell interpolation"
    ]["run"]
    assert sign_steps["Revalidate signer identity on the fresh runner"]["env"][
        "VISION_SUPPLIER_SIGNER_IDENTITY"
    ] == "${{ inputs.signer_identity }}"
    assert '"VISION_SUPPLIER_SIGNER_IDENTITY=$env:VISION_SUPPLIER_SIGNER_IDENTITY" >> $env:GITHUB_ENV' in sign_steps[
        "Revalidate signer identity on the fresh runner"
    ]["run"]
    assert sign_steps["Sign only verified evidence with the protected supplier key"]["env"] == {
        "VISION_SUPPLIER_PRIVATE_KEY_PEM": "${{ secrets.VISION_SUPPLIER_PRIVATE_KEY_PEM }}"
    }


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
        "subject_sha256",
        "manifest_sha256",
        "attestation_bundle_sha256",
        "signer_identity",
    }
    parsed = load_workflow_yaml(workflow)
    verify = parsed["jobs"]["verify_evidence"]
    sign = parsed["jobs"]["sign_evidence"]
    sign_source = workflow[workflow.index("  sign_evidence:\n"):]
    assert workflow.count("runs-on: windows-latest") == 2
    assert "environment" not in verify
    assert sign["environment"] == "experimental-candidate"
    assert sign["needs"] == "verify_evidence"
    verify_steps = {step["name"]: step for step in verify["steps"]}
    sign_steps = {step["name"]: step for step in sign["steps"]}
    _assert_signer_environment_authority(workflow)
    assert "repository: ${{ job.workflow_repository }}" in workflow
    assert "ref: ${{ job.workflow_sha }}" in workflow
    assert "path: trusted-signer" in workflow
    assert "source_commit" in workflow and "source_ref" in workflow
    assert "--repo \"hbhjt/vending-vision\"" in workflow
    assert "--signer-repo" not in workflow
    assert "--deny-self-hosted-runners" in workflow
    assert f'--signer-digest "{TRUSTED_BUILDER_COMMIT}"' in workflow
    assert "actions/checkout@v4" in workflow
    assert workflow.count("actions/checkout@v4") == 2
    assert "path: source" not in workflow
    assert "actions/setup-python" not in workflow
    assert "scripts/evidence_artifact.py" in workflow
    assert "--expected-digest $env:UNSIGNED_EVIDENCE_SHA256" in sign_source
    assert "scripts/verify_trusted_script_set.py" in sign_source
    assert "candidate-input" not in sign_source
    assert "verified-candidate" not in sign_source
    assert "source-approval" not in sign_source
    assert ".venv" not in sign_source
    assert "& $env:TRUSTED_PYTHON" in sign_source
    assert "--openssl $env:TRUSTED_OPENSSL" in sign_source
    input_lines = [line.strip() for line in workflow.splitlines() if "${{ inputs." in line]
    assert input_lines
    assert all(
        re.fullmatch(r"[A-Z][A-Z0-9_]*: \$\{\{ inputs\.[a-z][a-z0-9_]* \}\}", line)
        for line in input_lines
    )
    run_blocks = re.findall(
        r"(?ms)^\s+run: \|\n(?P<body>.*?)(?=^\s+- name:|^\s+- uses:|^  [a-z_]+:|\Z)",
        workflow,
    )
    assert run_blocks
    assert all("${{ inputs." not in block for block in run_blocks)
    assert all("${{ github.event.inputs" not in block for block in run_blocks)
    for forbidden in (
        "artifact_path",
        "artifact_file",
        "attestation_bundle_file",
        "builder_evidence_file",
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


@pytest.mark.parametrize(
    "old,new,count",
    [
        ("    environment: experimental-candidate\n    permissions:", "    permissions:", 1),
        ("    environment: experimental-candidate\n    permissions:", "    environment: staging\n    permissions:", 1),
        (
            "VISION_SUPPLIER_SIGNER_IDENTITY: ${{ inputs.signer_identity }}",
            "VISION_SUPPLIER_SIGNER_IDENTITY: ${{ vars.WRONG_IDENTITY }}",
            1,
        ),
        (
            "VISION_SUPPLIER_PRIVATE_KEY_PEM: ${{ secrets.VISION_SUPPLIER_PRIVATE_KEY_PEM }}",
            "VISION_SUPPLIER_PRIVATE_KEY_PEM: ${{ vars.VISION_SUPPLIER_PRIVATE_KEY_PEM }}",
            1,
        ),
    ],
)
def test_trusted_signer_environment_authority_rejects_missing_or_mis_scoped_inputs(
    old, new, count
):
    source = SIGNER.read_text("utf-8")
    with pytest.raises((AssertionError, KeyError)):
        _assert_signer_environment_authority(source.replace(old, new, count))


def test_trusted_signer_descriptor_binds_the_path_aware_ruleset_authority():
    verify_trusted_script_set(ROOT, ROOT / "trusted-signer-scripts.json")
    descriptor = json.loads((ROOT / "trusted-signer-scripts.json").read_text("utf-8"))
    ruleset = next(
        item
        for item in descriptor["scripts"]
        if item["path"] == "scripts/verify_release_tag_ruleset.py"
    )
    assert ruleset["sha256"] == hashlib.sha256(TAG_RULESET.read_bytes()).hexdigest()
    assert not github_ref_name_matches("refs/*", "refs/tags/v1.2.3-rc.1")


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


def test_source_approval_rejects_a_tag_moved_away_from_the_claimed_commit(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.email", "stage27@example.test")
    _git(repository, "config", "user.name", "Stage 27")
    (repository / "tracked.txt").write_text("first\n", "utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "first")
    claimed = _git(repository, "rev-parse", "HEAD")
    (repository / "tracked.txt").write_text("moved\n", "utf-8")
    _git(repository, "commit", "-am", "moved")
    moved = _git(repository, "rev-parse", "HEAD")
    _git(repository, "tag", "v0.2.1-rc.1", moved)

    completed = subprocess.run(
        [
            sys.executable,
            str(SOURCE_APPROVAL),
            "--git-dir",
            str(repository / ".git"),
            "--source-commit",
            claimed,
            "--source-ref",
            "refs/tags/v0.2.1-rc.1",
            "--protected-main",
            "main",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "does not identify the claimed commit" in completed.stdout


def test_source_ref_injection_is_data_and_cannot_execute(tmp_path):
    marker = tmp_path / "injected"
    malicious_ref = f'refs/tags/v0.2.1-rc.1";touch {marker};#'

    completed = subprocess.run(
        [
            sys.executable,
            str(SOURCE_APPROVAL),
            "--git-dir",
            str(ROOT / ".git"),
            "--source-commit",
            "a" * 40,
            "--source-ref",
            malicious_ref,
            "--protected-main",
            "main",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "not an RC tag" in completed.stdout
    assert not marker.exists()


def test_unsigned_evidence_digest_survives_only_an_exact_cross_job_copy(tmp_path):
    from scripts.evidence_artifact import DOCUMENTS, seal, verify

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in DOCUMENTS:
        (evidence / name).write_text(json.dumps({"name": name}), "utf-8")

    digest = seal(evidence, "unsigned")
    verify(evidence, "unsigned", digest)
    (evidence / DOCUMENTS[0]).write_text("tampered", "utf-8")

    try:
        verify(evidence, "unsigned", digest)
    except AssertionError as exc:
        assert "payload binding mismatch" in str(exc)
    else:
        raise AssertionError("tampered cross-job evidence was accepted")


def test_release_tag_ruleset_fails_closed_without_active_non_bypass_update_and_delete_rules(tmp_path):
    protected = {
        "id": 27,
        "target": "tag",
        "enforcement": "active",
        "bypass_actors": [],
        "conditions": {
            "ref_name": {
                "include": ["refs/tags/v*.*.*-rc.*"],
                "exclude": [],
            }
        },
        "rules": [{"type": "update"}, {"type": "deletion"}],
    }
    fixture = tmp_path / "rulesets.json"

    for value, accepted in (([], False), ([{**protected, "rules": []}], False), ([protected], True)):
        fixture.write_text(json.dumps(value), "utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(TAG_RULESET),
                "--rulesets",
                str(fixture),
                "--repository",
                "hbhjt/vending-vision",
                "--source-ref",
                "refs/tags/v0.2.1-rc.1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert (completed.returncode == 0) is accepted, completed.stdout


def _protecting_ruleset(*, include: list[str], exclude: list[str] | None = None):
    return {
        "id": 73,
        "target": "tag",
        "enforcement": "active",
        "bypass_actors": [],
        "conditions": {
            "ref_name": {"include": include, "exclude": exclude or []},
        },
        "rules": [{"type": "update"}, {"type": "deletion"}],
    }


def test_ruleset_single_star_does_not_cross_ref_path_separators():
    with pytest.raises(AssertionError, match="no active"):
        verify_rulesets(
            [_protecting_ruleset(include=["refs/*"])],
            repository="hbhjt/vending-vision",
            source_ref="refs/tags/v1.2.3-rc.1",
        )


@pytest.mark.parametrize(
    "pattern",
    [
        "refs/tags/v*",
        "refs/tags/**",
        "refs/tags/v1.2.3-rc.1",
    ],
)
def test_ruleset_glob_and_literal_patterns_cover_exact_rc_tag(pattern):
    assert (
        verify_rulesets(
            [_protecting_ruleset(include=[pattern])],
            repository="hbhjt/vending-vision",
            source_ref="refs/tags/v1.2.3-rc.1",
        )
        == 73
    )


def test_ruleset_double_star_crosses_nested_ref_levels():
    assert not github_ref_name_matches(
        "refs/tags/v*", "refs/tags/releases/v1.2.3-rc.1"
    )
    assert github_ref_name_matches(
        "refs/tags/**", "refs/tags/releases/v1.2.3-rc.1"
    )


def test_ruleset_exclude_pattern_takes_precedence_over_include():
    with pytest.raises(AssertionError, match="no active"):
        verify_rulesets(
            [
                _protecting_ruleset(
                    include=["refs/tags/**"],
                    exclude=["refs/tags/v1.2.3-rc.1"],
                )
            ],
            repository="hbhjt/vending-vision",
            source_ref="refs/tags/v1.2.3-rc.1",
        )


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
                "builderWorkflowSha": TRUSTED_BUILDER_COMMIT,
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
