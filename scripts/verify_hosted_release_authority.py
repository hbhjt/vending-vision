from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


TRUSTED_REPOSITORY = "hbhjt/vending-vision"
RC_REF = re.compile(
    r"^refs/tags/v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)-rc\.[0-9A-Za-z.-]+$"
)
COMMIT = re.compile(r"^[a-f0-9]{40}$")
MODES = {"publish-admission", "publish-complete", "proof"}


class AuthorityError(RuntimeError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise AuthorityError(reason)


def _release_from_query(value: object) -> object:
    _require(isinstance(value, dict), "response_shape")
    if set(value) == {"isDraft", "isPrerelease", "tagName", "targetCommitish"}:
        return value
    data = value.get("data")
    _require(isinstance(data, dict), "response_data")
    repository = data.get("repository")
    _require(isinstance(repository, dict), "response_repository")
    return repository.get("release")


def verify_release_fence(
    value: object,
    *,
    mode: str,
    repository: str,
    source_ref: str,
    source_commit: str,
) -> None:
    _require(mode in MODES, "mode")
    _require(repository == TRUSTED_REPOSITORY, "repository")
    _require(RC_REF.fullmatch(source_ref) is not None, "source_ref")
    _require(COMMIT.fullmatch(source_commit) is not None, "source_commit")
    release = _release_from_query(value)
    if mode == "publish-admission":
        _require(release is None, "release_already_exists")
        return

    _require(isinstance(release, dict), "release_missing")
    expected_tag = source_ref.removeprefix("refs/tags/")
    _require(release.get("tagName") == expected_tag, "release_tag")
    _require(release.get("targetCommitish") == source_commit, "release_target")
    _require(release.get("isDraft") is False, "release_draft")
    _require(release.get("isPrerelease") is True, "release_prerelease")
    _require(set(release) == {"isDraft", "isPrerelease", "tagName", "targetCommitish"}, "release_shape")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-query", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--repository", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    try:
        verify_release_fence(
            json.loads(args.release_query.read_text("utf-8")),
            mode=args.mode,
            repository=args.repository,
            source_ref=args.source_ref,
            source_commit=args.source_commit,
        )
    except (AuthorityError, OSError, UnicodeError, ValueError) as exc:
        print(f"HOSTED_RELEASE_AUTHORITY=FAIL:{exc}")
        return 1
    print(
        "HOSTED_RELEASE_AUTHORITY=PASS:"
        f"{args.mode}:{args.source_ref}:{args.source_commit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
