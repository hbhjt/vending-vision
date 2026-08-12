# Trusted hosted release authority

The private `hbhjt/vending-vision` repository does not treat repository
rulesets as an executable release prerequisite. The repository plan returns
`403` for the rulesets REST API, so a successful ruleset lookup cannot be part
of either candidate publication or pre-cutover proof.

## Authority

The hosted authority is the conjunction of these independently checked facts:

1. The privileged jobs share the existing `experimental-candidate` GitHub
   environment with the candidate signer. Its deployment branch policy has
   `protected_branches=false`, `custom_branch_policies=true`, and exactly one
   custom policy `{type: tag, name: v*.*.*-rc.*}`. A branch dispatch cannot
   enter the job.
2. The job fetches `main`, the exact 40-character source commit, and the exact
   `refs/tags/vX.Y.Z-rc.N` ref into a fresh bare repository. The tag must peel to
   the claimed commit and that commit must be an ancestor of fetched `main`.
3. GitHub build provenance is verified with the same exact source ref and
   source digest. Candidate bytes cannot substitute a different source claim.
4. Publication is create-only. Before `gh release create`, a GraphQL lookup
   must return no release for the tag. The workflow never creates, updates, or
   deletes a Git tag and never edits or deletes a release.
5. After publication, and independently before proof execution and signing,
   the release must be a non-draft prerelease whose `tag_name` and immutable
   `target_commitish` are the exact tag and source commit. The exact tag is
   fetched again in each fresh job. A moved tag therefore disagrees with the
   existing release target and fails closed.

`experimental-candidate` is an admission authority, not a secret-transfer
boundary. The manual caller has read-only repository permission and invokes
only the SHA-pinned reusable proof. GitHub does not pass environment secrets
through a reusable-workflow call, and the proof workflow never references
`VISION_SUPPLIER_PRIVATE_KEY_PEM` or any other supplier-signing secret.

Before release operation, an administrator or operator with permission to read
environment policies should run the repository's read-only preflight. The
workflow does not perform this API call because its ordinary `github.token`
may receive `403` even though environment admission itself works.

## Repository governance constraint

This is the minimum trusted alternative available on the current private
repository plan. It does not claim that GitHub mechanically makes the tag ref
or release undeletable. Repository administrators and credentials with
out-of-band write authority could delete both objects and recreate them. Such
administrative recovery is prohibited by repository policy, must be audited,
and requires a new RC tag rather than reuse of a published name. If GitHub tag
rulesets with non-bypass update and deletion rules become available, they
should replace this governance constraint while retaining exact Git,
attestation, and release-target checks.
