"""Strict provenance for the independent regional-evidence evaluator."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REGIONAL_EVALUATOR_DESCRIPTOR_PATH = REPO_ROOT / "regional-evaluator-descriptor.json"
REGIONAL_EVALUATOR_DESCRIPTOR_SCHEMA_VERSION = "vem-ai-regional-evaluator-descriptor/v1"
REGIONAL_EVALUATOR_SOURCE_PATHS = (
    "vision/catvton_pose_masks.py",
    "vision/catvton_preprocess.py",
    "vision/config.py",
    "vision/pose_estimator.py",
    "vision/regional_evaluator.py",
    "vision/regional_evaluator_provenance.py",
    "vision/vendor/__init__.py",
    "vision/vendor/catvton/__init__.py",
    "vision/vendor/catvton/model/__init__.py",
    "vision/vendor/catvton/model/SCHP/__init__.py",
    "vision/vendor/catvton/model/SCHP/networks/AugmentCE2P.py",
    "vision/vendor/catvton/model/SCHP/networks/__init__.py",
    "vision/vendor/catvton/model/SCHP/utils/__init__.py",
    "vision/vendor/catvton/model/SCHP/utils/transforms.py",
    "vision/vendor/catvton/model/attn_processor.py",
    "vision/vendor/catvton/model/pipeline.py",
    "vision/vendor/catvton/model/utils.py",
    "vision/vendor/catvton/utils.py",
)
REGIONAL_EVALUATOR_ENTRY_MODULES = (
    "vision.regional_evaluator",
    "vision.regional_evaluator_provenance",
)
REGIONAL_EVALUATOR_SEMANTICS = {
    "algorithm": "rgb-absolute-delta-rle/v1",
    "atr": "schp-atr",
    "lip": "schp-lip",
    "pose": "mediapipe-pose-or-frame-proportional",
}


class RegionalEvaluatorProvenanceError(RuntimeError):
    pass


def canonical_regional_evaluator_descriptor_json(descriptor: dict[str, Any]) -> str:
    return json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_relative(module: str) -> str | None:
    if not module.startswith("vision"):
        return None
    return module.replace(".", "/") + ".py"


def _module_name(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root)
    if relative.name == "__init__.py":
        return ".".join(relative.parent.parts)
    return ".".join(relative.with_suffix("").parts)


def _module_path(root: Path, module: str) -> Path | None:
    relative = _module_relative(module)
    if relative is None:
        return None
    candidate = root / relative
    if candidate.is_file():
        return candidate
    package = root / module.replace(".", "/") / "__init__.py"
    return package if package.is_file() else None


def _package_initializers(root: Path, module: str) -> set[Path]:
    parts = module.split(".")
    return {
        initializer
        for length in range(1, len(parts))
        if (initializer := root / "/".join(parts[:length]) / "__init__.py").is_file()
    }


def _local_import_modules(root: Path, path: Path) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    current = _module_name(root, path)
    package = current if path.name == "__init__.py" else current.rsplit(".", maxsplit=1)[0]
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names if alias.name.startswith("vision"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".")
                base = ".".join(base_parts[: len(base_parts) - node.level + 1])
                module = ".".join(part for part in (base, node.module or "") if part)
            else:
                module = node.module or ""
            if module.startswith("vision"):
                if node.module:
                    result.add(module)
                else:
                    result.update(f"{module}.{alias.name}" for alias in node.names)
    return result


def discovered_regional_evaluator_source_paths(root: str | Path = REPO_ROOT) -> tuple[str, ...]:
    root_path = Path(root).resolve()
    pending = list(REGIONAL_EVALUATOR_ENTRY_MODULES)
    visited: set[str] = set()
    paths: set[Path] = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = _module_path(root_path, module)
        if path is None:
            raise RegionalEvaluatorProvenanceError(
                "regional_evaluator_local_import_missing"
            )
        paths.add(path)
        paths.update(_package_initializers(root_path, module))
        pending.extend(_local_import_modules(root_path, path))
    return tuple(
        sorted(
            path.relative_to(root_path).as_posix()
            for path in paths
            if path.stat().st_size > 0
        )
    )


def build_regional_evaluator_descriptor(root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root_path = Path(root).resolve()
    sources = []
    for relative in sorted(REGIONAL_EVALUATOR_SOURCE_PATHS):
        path = (root_path / relative).resolve(strict=False)
        if root_path not in path.parents or not path.is_file():
            raise RegionalEvaluatorProvenanceError("regional_evaluator_descriptor_source")
        sources.append(
            {"byteSize": path.stat().st_size, "path": relative, "sha256": _sha256(path)}
        )
    return {
        "schemaVersion": REGIONAL_EVALUATOR_DESCRIPTOR_SCHEMA_VERSION,
        "semantics": REGIONAL_EVALUATOR_SEMANTICS,
        "sources": sources,
    }


def load_regional_evaluator_descriptor(root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root_path = Path(root).resolve()
    descriptor_path = root_path / "regional-evaluator-descriptor.json"
    try:
        raw = descriptor_path.read_text("utf-8")
        descriptor = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise RegionalEvaluatorProvenanceError(
            "regional_evaluator_descriptor_missing_or_invalid"
        ) from exc
    if canonical_regional_evaluator_descriptor_json(descriptor) != raw.rstrip("\n"):
        raise RegionalEvaluatorProvenanceError("regional_evaluator_descriptor_noncanonical")
    if set(descriptor) != {"schemaVersion", "semantics", "sources"}:
        raise RegionalEvaluatorProvenanceError("regional_evaluator_descriptor_shape")
    if descriptor["schemaVersion"] != REGIONAL_EVALUATOR_DESCRIPTOR_SCHEMA_VERSION:
        raise RegionalEvaluatorProvenanceError("regional_evaluator_descriptor_schema")
    if descriptor["semantics"] != REGIONAL_EVALUATOR_SEMANTICS:
        raise RegionalEvaluatorProvenanceError("regional_evaluator_descriptor_semantics")
    sources = descriptor["sources"]
    if not isinstance(sources, list) or len(sources) != len(REGIONAL_EVALUATOR_SOURCE_PATHS):
        raise RegionalEvaluatorProvenanceError("regional_evaluator_descriptor_sources")
    paths = [source.get("path") for source in sources if isinstance(source, dict)]
    if paths != sorted(REGIONAL_EVALUATOR_SOURCE_PATHS) or len(paths) != len(sources):
        raise RegionalEvaluatorProvenanceError("regional_evaluator_descriptor_sources")
    for source in sources:
        if (
            not isinstance(source, dict)
            or set(source) != {"byteSize", "path", "sha256"}
            or not isinstance(source["byteSize"], int)
            or source["byteSize"] <= 0
            or not isinstance(source["sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"])
        ):
            raise RegionalEvaluatorProvenanceError("regional_evaluator_descriptor_source")
    return descriptor


def verify_regional_evaluator_provenance_at_root(root: str | Path) -> bool:
    root_path = Path(root).resolve()
    try:
        descriptor = load_regional_evaluator_descriptor(root_path)
        if tuple(source["path"] for source in descriptor["sources"]) != discovered_regional_evaluator_source_paths(root_path):
            return False
        return descriptor == build_regional_evaluator_descriptor(root_path)
    except RegionalEvaluatorProvenanceError:
        return False


def verify_regional_evaluator_provenance() -> bool:
    return verify_regional_evaluator_provenance_at_root(REPO_ROOT)


def regional_evaluator_descriptor_sha256() -> str:
    return _sha256(REGIONAL_EVALUATOR_DESCRIPTOR_PATH)
