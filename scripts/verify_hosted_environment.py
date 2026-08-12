from __future__ import annotations

import json
import subprocess
import sys


REPOSITORY = "hbhjt/vending-vision"
ENVIRONMENT = "experimental-candidate"
EXPECTED_POLICY = {"type": "tag", "name": "v*.*.*-rc.*"}


def verify_environment(environment: object, policies: object) -> None:
    if not isinstance(environment, dict) or not isinstance(policies, dict):
        raise AssertionError("response_shape")
    branch_policy = environment.get("deployment_branch_policy")
    policy_values = policies.get("branch_policies")
    valid = (
        environment.get("name") == ENVIRONMENT
        and branch_policy
        == {"protected_branches": False, "custom_branch_policies": True}
        and isinstance(policy_values, list)
        and len(policy_values) == 1
        and isinstance(policy_values[0], dict)
        and {key: policy_values[0].get(key) for key in EXPECTED_POLICY}
        == EXPECTED_POLICY
    )
    if not valid:
        raise AssertionError("configuration_drift")


def main() -> int:
    endpoints = (
        f"repos/{REPOSITORY}/environments/{ENVIRONMENT}",
        f"repos/{REPOSITORY}/environments/{ENVIRONMENT}/deployment-branch-policies",
    )
    values = []
    for endpoint in endpoints:
        completed = subprocess.run(
            ["gh", "api", "--method", "GET", endpoint],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            print(completed.stderr.strip() or "environment policy API unavailable")
            return 2
        values.append(json.loads(completed.stdout))
    try:
        verify_environment(*values)
    except AssertionError as exc:
        print(f"HOSTED_ENVIRONMENT=FAIL:{exc}")
        return 1
    print(f"HOSTED_ENVIRONMENT=PASS:{ENVIRONMENT}:{EXPECTED_POLICY['name']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"HOSTED_ENVIRONMENT=FAIL:{exc}")
        raise SystemExit(2)
