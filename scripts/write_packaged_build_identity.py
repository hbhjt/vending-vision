#!/usr/bin/env python3
"""Write the canonical packaged build identity without editing tracked sources."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.build_identity import write_packaged_build_identity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-marker", required=True, type=Path)
    parser.add_argument("--identity-output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    write_packaged_build_identity(
        args.version_marker.resolve(),
        args.identity_output.resolve(),
        args.source_commit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
