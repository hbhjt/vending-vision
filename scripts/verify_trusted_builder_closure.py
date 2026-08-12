"""Verify the immutable source closure used by the trusted companion builder."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat


SCHEMA = "vending-vision-trusted-companion-builder-closure/v1"
EXPECTED_PATHS = (
    ".github/workflows/trusted-precutover-companion-builder.yml",
    ".python-version",
    "requirements.txt",
    "run_precutover_verifier.py",
    "scripts/ai_model_pack_release.py",
    "scripts/archive_extractor_worker.py",
    "scripts/candidate_artifact_manifest.py",
    "scripts/dependency_lock.py",
    "scripts/download_verified_archive.py",
    "scripts/precutover_companion_descriptor.py",
    "scripts/verify_trusted_builder_closure.py",
    "scripts/verify_trusted_candidate_inputs.py",
    "vending_vision_precutover_verifier.spec",
    "vision/__init__.py",
    "vision/ai_model_pack.py",
    "vision/ai_runtime_descriptor.py",
    "vision/precutover_companion.py",
    "vision/process_supervisor.py",
)
_LOCAL_MODULES = {
    path.removesuffix(".py").replace("/", "."): path
    for path in EXPECTED_PATHS
    if path.endswith(".py")
}


class ClosureError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for name in names:
            for module, relative in _LOCAL_MODULES.items():
                if name == module or name.startswith(module + "."):
                    imported.add(relative)
    return imported


def _spec_local_modules(root: Path) -> set[str]:
    source = (root / "vending_vision_precutover_verifier.spec").read_text("utf-8")
    modules = set(re.findall(r'["\']((?:scripts|vision)\.[A-Za-z0-9_.]+)["\']', source))
    paths = set()
    for module in modules:
        relative = _LOCAL_MODULES.get(module)
        if relative is None:
            raise ClosureError(f"trusted_builder_closure_unlisted_spec_module:{module}")
        paths.add(relative)
    return paths


def verify(root: Path, manifest_path: Path) -> None:
    raw = manifest_path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClosureError("trusted_builder_closure_json") from exc
    if type(value) is not dict or set(value) != {"files", "schemaVersion"}:
        raise ClosureError("trusted_builder_closure_schema")
    if value["schemaVersion"] != SCHEMA or raw != _canonical(value) + b"\n":
        raise ClosureError("trusted_builder_closure_noncanonical")
    files = value["files"]
    if type(files) is not list or len(files) != len(EXPECTED_PATHS):
        raise ClosureError("trusted_builder_closure_file_set")
    paths = []
    for item in files:
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            raise ClosureError("trusted_builder_closure_entry")
        path = item["path"]
        digest = item["sha256"]
        if (
            type(path) is not str
            or PurePosixPath(path).as_posix() != path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or type(digest) is not str
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
        ):
            raise ClosureError("trusted_builder_closure_entry")
        paths.append(path)
    if tuple(paths) != EXPECTED_PATHS or len(set(paths)) != len(paths):
        raise ClosureError("trusted_builder_closure_file_set")
    for item in files:
        candidate = root / item["path"]
        facts = candidate.lstat()
        if not stat.S_ISREG(facts.st_mode) or candidate.is_symlink():
            raise ClosureError(f"trusted_builder_closure_nonregular:{item['path']}")
        if _sha256(candidate) != item["sha256"]:
            raise ClosureError(f"trusted_builder_closure_digest:{item['path']}")

    discovered = _spec_local_modules(root)
    for relative in EXPECTED_PATHS:
        if relative.endswith(".py"):
            discovered.update(_local_imports(root / relative))
    if not discovered.issubset(set(EXPECTED_PATHS)):
        raise ClosureError("trusted_builder_closure_dependency")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    try:
        verify(args.root.resolve(), args.manifest.resolve())
    except (ClosureError, OSError, SyntaxError) as exc:
        print(f"TRUSTED_BUILDER_CLOSURE=FAIL:{exc}")
        return 1
    print("TRUSTED_BUILDER_CLOSURE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
