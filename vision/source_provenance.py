from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCE_DESCRIPTOR_PATH = REPO_ROOT / "official-ai-source-descriptor.json"
SOURCE_DESCRIPTOR_SCHEMA_VERSION = "vem-official-ai-source-descriptor/v1"
OFFICIAL_CATVTON_SOURCE_REVISION = "3b795364a4d2f3b5adb365f39cdea376d20bc53c"
OFFICIAL_SOURCE_PATHS = (
    "run_ai_attempt_worker.py",
    "vision/ai_attempt_process.py",
    "vision/ai_attempt_worker.py",
    "vision/ai_model_pack.py",
    "vision/ai_runtime_descriptor.py",
    "vision/catvton_pose_masks.py",
    "vision/catvton_preprocess.py",
    "vision/process_supervisor.py",
    "vision/source_provenance.py",
    "vision/vendor/catvton/LICENSE",
    "vision/vendor/catvton/PROVENANCE.md",
    "vision/vendor/catvton/__init__.py",
    "vision/vendor/catvton/model/SCHP/__init__.py",
    "vision/vendor/catvton/model/SCHP/networks/AugmentCE2P.py",
    "vision/vendor/catvton/model/SCHP/networks/__init__.py",
    "vision/vendor/catvton/model/SCHP/utils/__init__.py",
    "vision/vendor/catvton/model/SCHP/utils/transforms.py",
    "vision/vendor/catvton/model/__init__.py",
    "vision/vendor/catvton/model/attn_processor.py",
    "vision/vendor/catvton/model/pipeline.py",
    "vision/vendor/catvton/model/utils.py",
    "vision/vendor/catvton/utils.py",
)


class SourceProvenanceError(RuntimeError):
    pass


def canonical_source_descriptor_json(descriptor: dict) -> str:
    return json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_official_source_descriptor() -> dict:
    sources = []
    for relative in sorted(OFFICIAL_SOURCE_PATHS):
        path = (REPO_ROOT / relative).resolve()
        if REPO_ROOT not in path.parents or not path.is_file():
            raise SourceProvenanceError("official_source_descriptor_source")
        sources.append({"path": relative, "sha256": _sha256(path)})
    return {
        "catvtonSourceRevision": OFFICIAL_CATVTON_SOURCE_REVISION,
        "schemaVersion": SOURCE_DESCRIPTOR_SCHEMA_VERSION,
        "sources": sources,
    }


def load_official_source_descriptor(root: Path | None = None) -> dict:
    descriptor_path = (root / "official-ai-source-descriptor.json") if root is not None else OFFICIAL_SOURCE_DESCRIPTOR_PATH
    try:
        raw = descriptor_path.read_text("utf-8")
        descriptor = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise SourceProvenanceError("official_source_descriptor_missing_or_invalid") from exc
    if canonical_source_descriptor_json(descriptor) != raw.rstrip("\n"):
        raise SourceProvenanceError("official_source_descriptor_noncanonical")
    if set(descriptor) != {"schemaVersion", "catvtonSourceRevision", "sources"}:
        raise SourceProvenanceError("official_source_descriptor_shape")
    if descriptor["schemaVersion"] != SOURCE_DESCRIPTOR_SCHEMA_VERSION:
        raise SourceProvenanceError("official_source_descriptor_schema")
    if descriptor["catvtonSourceRevision"] != OFFICIAL_CATVTON_SOURCE_REVISION:
        raise SourceProvenanceError("official_source_revision_mismatch")
    sources = descriptor["sources"]
    if not isinstance(sources, list) or not sources:
        raise SourceProvenanceError("official_source_descriptor_sources")
    paths = [source.get("path") for source in sources if isinstance(source, dict)]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise SourceProvenanceError("official_source_descriptor_sources")
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise SourceProvenanceError("official_source_descriptor_source")
        if (
            not isinstance(source["path"], str)
            or source["path"].startswith("/")
            or ".." in Path(source["path"]).parts
            or "\\" in source["path"]
            or not isinstance(source["sha256"], str)
            or len(source["sha256"]) != 64
            or source["sha256"] != source["sha256"].lower()
        ):
            raise SourceProvenanceError("official_source_descriptor_source")
    return descriptor


def verify_official_source_provenance_at_root(root: str | Path) -> bool:
    root_path = Path(root).resolve()
    try:
        descriptor = load_official_source_descriptor(root_path)
        for source in descriptor["sources"]:
            path = (root_path / source["path"]).resolve()
            if root_path not in path.parents or not path.is_file() or _sha256(path) != source["sha256"]:
                return False
        return True
    except SourceProvenanceError:
        return False


def verify_official_source_provenance() -> bool:
    return verify_official_source_provenance_at_root(REPO_ROOT)


def official_source_digest() -> str:
    descriptor = load_official_source_descriptor()
    digest = hashlib.sha256(canonical_source_descriptor_json(descriptor).encode("utf-8"))
    for source in descriptor["sources"]:
        path = (REPO_ROOT / source["path"]).resolve()
        digest.update(source["path"].encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()
