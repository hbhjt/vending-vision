from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_ai_wheelhouse import _validate_download_url, canonical_json
from vision.ai_runtime_descriptor import load_ai_runtime_descriptor


class MaterializeError(RuntimeError):
    pass


def materialize_ai_wheelhouse(
    descriptor_path: Path,
    destination: Path,
    *,
    opener=urlopen,
    runtime_descriptor_path: Path | None = None,
) -> None:
    if destination.exists():
        raise MaterializeError("ai_wheelhouse_destination_exists")
    raw = descriptor_path.read_text("utf-8")
    descriptor = json.loads(raw)
    if canonical_json(descriptor) != raw.rstrip("\n"):
        raise MaterializeError("ai_wheelhouse_descriptor_noncanonical")
    if descriptor.get("schemaVersion") != "vem-ai-worker-wheelhouse-release/v1":
        raise MaterializeError("ai_wheelhouse_descriptor_schema")
    if descriptor.get("source") != "pip-report-locked-wheelhouse":
        raise MaterializeError("ai_wheelhouse_descriptor_source")
    if runtime_descriptor_path is not None:
        runtime_descriptor = load_ai_runtime_descriptor(runtime_descriptor_path)
        if hashlib.sha256(descriptor_path.read_bytes()).hexdigest() != runtime_descriptor["requirementsAiLockSha256"]:
            raise MaterializeError("ai_wheelhouse_runtime_lock_mismatch")
    wheels = descriptor.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        raise MaterializeError("ai_wheelhouse_release_descriptor_required")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    seen: set[str] = set()
    try:
        for wheel in wheels:
            file_name = wheel.get("fileName")
            if not isinstance(file_name, str) or PurePosixPath(file_name).name != file_name:
                raise MaterializeError("ai_wheelhouse_path")
            if file_name in seen:
                raise MaterializeError("ai_wheelhouse_duplicate")
            seen.add(file_name)
            try:
                source = _validate_download_url(wheel["url"], file_name)
            except (KeyError, RuntimeError) as exc:
                raise MaterializeError("ai_wheelhouse_download_url") from exc
            if source != wheel.get("source"):
                raise MaterializeError("ai_wheelhouse_download_source")

            request = Request(wheel["url"], headers={"User-Agent": "vem-ai-wheelhouse-materializer/1"})
            target = staging / file_name
            digest = hashlib.sha256()
            byte_size = 0
            with opener(request, timeout=120.0) as response:
                if response.geturl() != wheel["url"]:
                    raise MaterializeError("ai_wheelhouse_redirect_identity")
                with target.open("xb") as output:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
                        digest.update(chunk)
                        byte_size += len(chunk)
            if byte_size != wheel.get("byteSize") or digest.hexdigest() != wheel.get("sha256"):
                raise MaterializeError("ai_wheelhouse_digest")
        if {path.name for path in staging.iterdir()} != seen:
            raise MaterializeError("ai_wheelhouse_extra_or_missing")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", default="requirements-ai.lock.json")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--runtime-descriptor", default="ai-runtime-descriptor.json")
    args = parser.parse_args()
    materialize_ai_wheelhouse(
        Path(args.descriptor),
        Path(args.destination).resolve(),
        runtime_descriptor_path=Path(args.runtime_descriptor),
    )
    print("AI wheelhouse materialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
