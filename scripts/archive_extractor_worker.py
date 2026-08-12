"""Stdlib-only child entrypoint for bounded archive extraction."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def _load_archive_module():
    module_path = Path(__file__).with_name("download_verified_archive.py")
    spec = importlib.util.spec_from_file_location("_trusted_archive_downloader", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("archive_worker_import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--max-extracted-bytes", required=True, type=int)
    parser.add_argument("--max-members", required=True, type=int)
    args = parser.parse_args()
    module = _load_archive_module()
    try:
        module._extract_archive(
            args.archive,
            args.destination,
            max_extracted_bytes=args.max_extracted_bytes,
            max_members=args.max_members,
        )
    except module.ArchiveError as exc:
        sys.stderr.write(str(exc))
        return module._EXTRACT_EXIT_CODES.get(str(exc), 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
