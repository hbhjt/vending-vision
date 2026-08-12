from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


RC_REF = re.compile(
    r"^refs/tags/v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-rc\.[0-9A-Za-z.-]+$"
)


def _flatten_pages(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise AssertionError("tag ruleset API response is not a list")
    if value and all(isinstance(page, list) for page in value):
        value = [item for page in value for item in page]
    if not all(isinstance(item, dict) for item in value):
        raise AssertionError("tag ruleset API response contains invalid entries")
    return value


def github_ref_name_matches(pattern: str, source_ref: str) -> bool:
    if pattern == "~ALL":
        return True
    expression = ["^"]
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                expression.append(".*")
                index += 2
                continue
            expression.append("[^/]*")
        elif character == "?":
            expression.append("[^/]")
        else:
            expression.append(re.escape(character))
        index += 1
    expression.append("$")
    return re.fullmatch("".join(expression), source_ref) is not None


def verify_rulesets(value: object, *, repository: str, source_ref: str) -> int:
    if repository != "hbhjt/vending-vision":
        raise AssertionError("unexpected release repository")
    if RC_REF.fullmatch(source_ref) is None:
        raise AssertionError("release source ref is not an RC tag")
    rulesets = _flatten_pages(value)
    for ruleset in rulesets:
        if ruleset.get("target") != "tag" or ruleset.get("enforcement") != "active":
            continue
        if ruleset.get("bypass_actors") != []:
            continue
        conditions = ruleset.get("conditions")
        ref_name = conditions.get("ref_name") if isinstance(conditions, dict) else None
        includes = ref_name.get("include") if isinstance(ref_name, dict) else None
        excludes = ref_name.get("exclude") if isinstance(ref_name, dict) else None
        if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
            continue
        if not isinstance(excludes, list) or not all(isinstance(item, str) for item in excludes):
            continue
        if not any(github_ref_name_matches(pattern, source_ref) for pattern in includes):
            continue
        if any(github_ref_name_matches(pattern, source_ref) for pattern in excludes):
            continue
        rules = ruleset.get("rules")
        if not isinstance(rules, list) or not all(isinstance(item, dict) for item in rules):
            continue
        rule_types = {item.get("type") for item in rules}
        if {"deletion", "update"}.issubset(rule_types):
            identifier = ruleset.get("id")
            if type(identifier) is not int or identifier <= 0:
                raise AssertionError("protecting tag ruleset identity is invalid")
            return identifier
    raise AssertionError("no active non-bypass tag ruleset prevents update and deletion")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rulesets", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    try:
        value = json.loads(args.rulesets.read_text("utf-8"))
        identifier = verify_rulesets(
            value, repository=args.repository, source_ref=args.source_ref
        )
    except (AssertionError, OSError, UnicodeError, ValueError) as exc:
        print(f"RELEASE_TAG_RULESET=FAIL:{exc}")
        return 1
    print(f"RELEASE_TAG_RULESET=PASS:{identifier}:{args.source_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
