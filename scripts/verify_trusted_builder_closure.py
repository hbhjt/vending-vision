"""Verify the immutable source closure used by the trusted companion builder."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess


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
    "vision/ai_attempt_process.py",
    "vision/ai_model_pack.py",
    "vision/ai_runtime_descriptor.py",
    "vision/precutover_companion.py",
    "vision/process_supervisor.py",
)
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


def _tracked_python_modules(root: Path) -> dict[str, str]:
    completed = subprocess.run(
        ["git", "ls-files", "-s", "-z", "--", "*.py"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ClosureError("trusted_builder_closure_git_index")
    modules: dict[str, str] = {}
    for record in completed.stdout.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode = metadata.split(b" ", 1)[0]
            relative = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ClosureError("trusted_builder_closure_git_index") from exc
        if mode != b"100644" and mode != b"100755":
            raise ClosureError(f"trusted_builder_closure_tracked_nonregular:{relative}")
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise ClosureError("trusted_builder_closure_git_path")
        candidate = root / relative
        facts = candidate.lstat()
        if not stat.S_ISREG(facts.st_mode) or candidate.is_symlink():
            raise ClosureError(f"trusted_builder_closure_tracked_nonregular:{relative}")
        module = relative.removesuffix(".py").replace("/", ".")
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        if module in modules:
            raise ClosureError(f"trusted_builder_closure_module_collision:{module}")
        modules[module] = relative
    return modules


def _resolve_module(name: str, modules: dict[str, str]) -> str | None:
    while name:
        relative = modules.get(name)
        if relative is not None:
            return relative
        name = name.rpartition(".")[0]
    return None


def _local_imports(path: Path, modules: dict[str, str]) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for name in names:
            relative = _resolve_module(name, modules)
            if relative is not None:
                imported.add(relative)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__import__"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            relative = _resolve_module(node.args[0].value, modules)
            if relative is not None:
                imported.add(relative)
    return imported


def _spec_local_modules(root: Path, modules: dict[str, str]) -> set[str]:
    spec_path = root / "vending_vision_precutover_verifier.spec"
    source = spec_path.read_text("utf-8")
    tree = ast.parse(source, filename=str(spec_path))
    analyses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Analysis"
    ]
    if len(analyses) != 1:
        raise ClosureError("trusted_builder_closure_spec_analysis")
    keywords = {item.arg: item.value for item in analyses[0].keywords if item.arg}
    binaries = keywords.get("binaries")
    datas = keywords.get("datas")
    hiddenimports = keywords.get("hiddenimports")
    if not isinstance(binaries, ast.List) or binaries.elts:
        raise ClosureError("trusted_builder_closure_spec_binaries")
    if not isinstance(datas, ast.Name) or datas.id != "datas":
        raise ClosureError("trusted_builder_closure_spec_datas")
    if not isinstance(hiddenimports, ast.Name) or hiddenimports.id != "hiddenimports":
        raise ClosureError("trusted_builder_closure_spec_hiddenimports")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literal = node.value.replace("\\", "/")
            if literal.endswith(".py") and literal != "run_precutover_verifier.py":
                raise ClosureError(
                    f"trusted_builder_closure_spec_local_data:{literal}"
                )
    declared = set(re.findall(r'["\']((?:scripts|vision)\.[A-Za-z0-9_.]+)["\']', source))
    paths = set()
    for module in declared:
        relative = _resolve_module(module, modules)
        if relative is None or (
            module not in modules
            and not any(name.startswith(module + ".") for name in modules)
        ):
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

    modules = _tracked_python_modules(root)
    expected = set(EXPECTED_PATHS)
    pending = {
        relative for relative in EXPECTED_PATHS if relative.endswith(".py")
    } | _spec_local_modules(root, modules)
    visited: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in visited:
            continue
        if relative not in expected:
            raise ClosureError(f"trusted_builder_closure_dependency:{relative}")
        visited.add(relative)
        pending.update(_local_imports(root / relative, modules) - visited)


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
