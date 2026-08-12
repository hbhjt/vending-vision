from __future__ import annotations

import pytest

from scripts.verify_hosted_release_authority import AuthorityError, verify_release_fence


REPOSITORY = "hbhjt/vending-vision"
SOURCE_REF = "refs/tags/v1.2.3-rc.4"
SOURCE_COMMIT = "a" * 40


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
