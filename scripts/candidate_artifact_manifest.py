from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "vending-vision-candidate-artifact/v2"
LAYOUT = {
    "mainOnedir": "vending-vision",
    "mainExecutable": "vending-vision/vending-vision.exe",
    "workerOnedir": "vending-vision-ai-worker",
    "workerExecutable": "vending-vision-ai-worker/vending-vision-ai-worker.exe",
    "workerInternal": "vending-vision-ai-worker/_internal",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_candidate_artifact_manifest(dist_root: Path, artifact: Path) -> dict:
    paths = {
        "mainExecutableSha256": dist_root / LAYOUT["mainExecutable"],
        "workerExecutableSha256": dist_root / LAYOUT["workerExecutable"],
        "runtimeDescriptorSha256": dist_root / LAYOUT["workerInternal"] / "ai-runtime-descriptor.json",
        "aiWheelhouseManifestSha256": dist_root / LAYOUT["workerInternal"] / "requirements-ai.lock.json",
        "sourceDescriptorSha256": dist_root / LAYOUT["workerInternal"] / "official-ai-source-descriptor.json",
        "modelPackDescriptorSha256": dist_root / LAYOUT["workerInternal"] / "official-ai-model-pack-descriptor.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing or not artifact.is_file():
        raise RuntimeError(f"candidate_artifact_input_missing:{missing}")
    return {
        "schemaVersion": SCHEMA,
        "artifactSha256": _sha256(artifact),
        "layout": LAYOUT,
        **{name: _sha256(path) for name, path in paths.items()},
    }


def write_candidate_artifact_manifest(dist_root: Path, artifact: Path, output: Path) -> str:
    output.write_text(canonical_json(build_candidate_artifact_manifest(dist_root, artifact)), "utf-8")
    digest = _sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "\n", "ascii")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-root", default="dist")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    digest = write_candidate_artifact_manifest(
        Path(args.dist_root).resolve(), Path(args.artifact).resolve(), Path(args.output).resolve()
    )
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
