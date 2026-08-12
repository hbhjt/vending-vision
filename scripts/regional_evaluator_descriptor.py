#!/usr/bin/env python3
"""Generate or verify the canonical regional evaluator descriptor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from vision.regional_evaluator_provenance import (  # noqa: E402
    REGIONAL_EVALUATOR_DESCRIPTOR_PATH,
    build_regional_evaluator_descriptor,
    canonical_regional_evaluator_descriptor_json,
    verify_regional_evaluator_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        if not verify_regional_evaluator_provenance():
            print("regional_evaluator_descriptor_mismatch", file=sys.stderr)
            return 1
        return 0
    REGIONAL_EVALUATOR_DESCRIPTOR_PATH.write_text(
        canonical_regional_evaluator_descriptor_json(build_regional_evaluator_descriptor())
        + "\n",
        "utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
