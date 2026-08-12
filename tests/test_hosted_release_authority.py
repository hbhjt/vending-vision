from __future__ import annotations

from pathlib import Path

import pytest

from scripts.verify_hosted_release_authority import AuthorityError, verify_release_fence
from scripts.verify_hosted_environment import verify_environment


REPOSITORY = "hbhjt/vending-vision"
SOURCE_REF = "refs/tags/v1.2.3-rc.4"
SOURCE_COMMIT = "a" * 40
ROOT = Path(__file__).parents[1]
AUTHORITY_COMMIT = "41afbd9bd07b67df9f93de1dea1a9f9b0cea0228"


def _release(value: object) -> dict:
    return {"data": {"repository": {"release": value}}}


def test_publish_admission_accepts_only_an_absent_release_for_the_exact_rc_tag():
    verify_release_fence(
        _release(None),
        mode="publish-admission",
        repository=REPOSITORY,
        source_ref=SOURCE_REF,
        source_commit=SOURCE_COMMIT,
    )

    with pytest.raises(AuthorityError, match="already_exists"):
        verify_release_fence(
            _release(
                {
                    "tagName": "v1.2.3-rc.4",
                    "targetCommitish": SOURCE_COMMIT,
                    "isDraft": False,
                    "isPrerelease": True,
                }
            ),
            mode="publish-admission",
            repository=REPOSITORY,
            source_ref=SOURCE_REF,
            source_commit=SOURCE_COMMIT,
        )


def test_published_and_proof_fences_bind_a_nondraft_prerelease_to_the_exact_commit():
    value = {
        "tagName": "v1.2.3-rc.4",
        "targetCommitish": SOURCE_COMMIT,
        "isDraft": False,
        "isPrerelease": True,
    }
    for mode in ("publish-complete", "proof"):
        verify_release_fence(
            value,
            mode=mode,
            repository=REPOSITORY,
            source_ref=SOURCE_REF,
            source_commit=SOURCE_COMMIT,
        )


@pytest.mark.parametrize(
    "release",
    [
        None,
        {
            "tagName": "v1.2.3-rc.5",
            "targetCommitish": SOURCE_COMMIT,
            "isDraft": False,
            "isPrerelease": True,
        },
        {
            "tagName": "v1.2.3-rc.4",
            "targetCommitish": "b" * 40,
            "isDraft": False,
            "isPrerelease": True,
        },
        {
            "tagName": "v1.2.3-rc.4",
            "targetCommitish": SOURCE_COMMIT,
            "isDraft": True,
            "isPrerelease": True,
        },
        {
            "tagName": "v1.2.3-rc.4",
            "targetCommitish": SOURCE_COMMIT,
            "isDraft": False,
            "isPrerelease": False,
        },
    ],
)
def test_proof_fails_closed_for_missing_or_rewritten_release_identity(release):
    with pytest.raises(AuthorityError):
        verify_release_fence(
            _release(release),
            mode="proof",
            repository=REPOSITORY,
            source_ref=SOURCE_REF,
            source_commit=SOURCE_COMMIT,
        )


@pytest.mark.parametrize(
    ("repository", "source_ref", "source_commit"),
    [
        ("fork/vending-vision", SOURCE_REF, SOURCE_COMMIT),
        (REPOSITORY, "refs/heads/main", SOURCE_COMMIT),
        (REPOSITORY, SOURCE_REF, "A" * 40),
    ],
)
def test_authority_rejects_repository_ref_and_digest_scope_drift(
    repository, source_ref, source_commit
):
    with pytest.raises(AuthorityError):
        verify_release_fence(
            _release(None),
            mode="publish-admission",
            repository=repository,
            source_ref=source_ref,
            source_commit=source_commit,
        )


def test_publish_and_proof_use_the_hosted_environment_and_release_fence_not_rulesets():
    publisher = (ROOT / ".github/workflows/publish-candidate.yml").read_text("utf-8")
    proof = (
        ROOT / ".github/workflows/trusted-precutover-companion-proof.yml"
    ).read_text("utf-8")

    assert "rulesets?targets=tag" not in publisher
    assert "rulesets?targets=tag" not in proof
    publish_job = publisher[publisher.index("  publish:\n") :]
    assert "environment: experimental-candidate" in publish_job.split("steps:", 1)[0]
    assert f"ref: {AUTHORITY_COMMIT}" in publish_job
    assert "verify_hosted_release_authority.py" in publish_job
    assert "--mode publish-admission" in publish_job
    assert "--mode publish-complete" in publish_job
    assert publish_job.index("--mode publish-admission") < publish_job.index(
        "gh release create"
    ) < publish_job.index("--mode publish-complete")

    execute = proof[proof.index("  execute:\n") : proof.index("  sign:\n")]
    sign = proof[proof.index("  sign:\n") : proof.index("  verify:\n")]
    for job in (execute, sign):
        assert "environment: experimental-candidate" in job.split("steps:", 1)[0]
        assert f"ref: {AUTHORITY_COMMIT}" in job
        assert "verify_hosted_release_authority.py" in job
        assert "--mode proof" in job
        candidate_attestation = (
            "gh attestation verify proof-input/candidate/candidate.zip"
            if "  execute:\n" in job
            else "attestation verify signer-proof-input/candidate/candidate.zip"
        )
        assert job.index("approve_candidate_source.py") < job.index(
            "verify_hosted_release_authority.py"
        ) < job.index(candidate_attestation)

    combined = publisher + proof
    for forbidden in (
        "git tag -f",
        "git push --delete",
        "gh release delete",
        "gh release edit",
    ):
        assert forbidden not in combined


def test_operator_preflight_requires_the_exact_existing_rc_tag_environment_policy():
    preflight = (ROOT / "scripts/verify_hosted_environment.py").read_text("utf-8")
    assert "experimental-candidate" in preflight
    assert "v*.*.*-rc.*" in preflight
    assert "PUT" not in preflight and "POST" not in preflight and "DELETE" not in preflight

    environment = {
        "name": "experimental-candidate",
        "deployment_branch_policy": {
            "protected_branches": False,
            "custom_branch_policies": True,
        },
    }
    policies = {
        "branch_policies": [{"id": 1, "type": "tag", "name": "v*.*.*-rc.*"}]
    }
    verify_environment(environment, policies)

    for drift in (
        {"branch_policies": []},
        {"branch_policies": [*policies["branch_policies"], policies["branch_policies"][0]]},
        {"branch_policies": [{"id": 1, "type": "branch", "name": "v*.*.*-rc.*"}]},
    ):
        with pytest.raises(AssertionError, match="configuration_drift"):
            verify_environment(environment, drift)
