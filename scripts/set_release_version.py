from __future__ import annotations

import argparse
import re
from pathlib import Path


SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    args = parser.parse_args()
    if not SEMVER.fullmatch(args.version):
        raise SystemExit("release version must be strict SemVer")
    target = Path(__file__).resolve().parents[1] / "vision" / "_build_version.py"
    target.write_text(f'APP_VERSION = "{args.version}"\n', encoding="utf-8")


if __name__ == "__main__":
    main()
