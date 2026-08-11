from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


FULL_SHA = re.compile(r"^[a-f0-9]{40}$")
RC_REF = re.compile(
    r"^refs/tags/v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-rc\.[0-9A-Za-z.-]+$"
)
SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def _git(git_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "--git-dir", str(git_dir), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def approve_source(
    *, git_dir: Path, source_commit: str, source_ref: str, protected_main: str
) -> None:
    if not git_dir.is_dir():
        raise AssertionError("source approval Git database is missing")
    if not FULL_SHA.fullmatch(source_commit):
        raise AssertionError("source commit is invalid")
    if not RC_REF.fullmatch(source_ref):
        raise AssertionError("source ref is not an RC tag")
    if not SAFE_REF.fullmatch(protected_main) or protected_main.startswith("-"):
        raise AssertionError("protected main ref is invalid")
    commit = _git(git_dir, "rev-parse", f"{source_commit}^{{commit}}")
    tag = _git(git_dir, "rev-parse", f"{source_ref}^{{commit}}")
    if commit.returncode != 0 or commit.stdout.strip() != source_commit:
        raise AssertionError("source commit object is missing")
    if tag.returncode != 0 or tag.stdout.strip() != source_commit:
        raise AssertionError("source ref does not identify the claimed commit")
    ancestry = _git(git_dir, "merge-base", "--is-ancestor", source_commit, protected_main)
    if ancestry.returncode != 0:
        raise AssertionError("source commit is not an ancestor of protected main")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--protected-main", required=True)
    args = parser.parse_args()
    try:
        approve_source(
            git_dir=args.git_dir.resolve(),
            source_commit=args.source_commit,
            source_ref=args.source_ref,
            protected_main=args.protected_main,
        )
    except (AssertionError, OSError) as exc:
        print(f"CANDIDATE_SOURCE_APPROVAL=FAIL:{exc}")
        return 1
    print(f"CANDIDATE_SOURCE_APPROVAL=PASS:{args.source_commit}:{args.source_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
