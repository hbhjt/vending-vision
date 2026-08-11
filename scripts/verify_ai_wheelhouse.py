from __future__ import annotations

import argparse
import email
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from vision.ai_runtime_descriptor import load_ai_runtime_descriptor


class WheelhouseError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_wheel_filename(path: Path) -> dict[str, str]:
    name = path.name
    if not name.endswith(".whl"):
        raise WheelhouseError("ai_wheelhouse_not_wheel")
    parts = name[:-4].split("-")
    if len(parts) < 5:
        raise WheelhouseError("ai_wheelhouse_wheel_name")
    return {
        "name": canonicalize_name(parts[0]),
        "version": parts[1],
        "tags": "-".join(parts[2:]),
    }


def _normalise_requirement(value: str) -> str:
    if "==" not in value:
        raise WheelhouseError("ai_wheelhouse_requirement")
    name, version = value.split("==", 1)
    return f"{canonicalize_name(name)}=={version}"


def _target_marker_environment(descriptor: dict) -> dict[str, str]:
    env = default_environment()
    if descriptor["target"] == "windows-x86_64":
        env.update(
            {
                "implementation_name": "cpython",
                "platform_machine": "AMD64",
                "platform_system": "Windows",
                "python_full_version": descriptor["python"],
                "python_version": ".".join(descriptor["python"].split(".")[:2]),
                "sys_platform": "win32",
                "os_name": "nt",
            }
        )
    return env


def _wheel_tag_compatible(tags: str, descriptor: dict) -> bool:
    tag_parts = tags.split("-")
    if len(tag_parts) != 3:
        return False
    python_tag, abi_tag, platform_tag = tag_parts
    if descriptor["target"] == "windows-x86_64":
        if platform_tag not in {"win_amd64", "any"}:
            return False
        python_ok = python_tag in {"py3", "py2.py3", "cp311"}
        abi_ok = abi_tag in {"none", "abi3", "cp311"}
        return python_ok and abi_ok
    return False


def _read_wheel_metadata(path: Path) -> tuple[str, str, list[str]]:
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA") and "/" in name
            ]
            if len(metadata_names) != 1:
                raise WheelhouseError("ai_wheelhouse_metadata")
            message = email.message_from_bytes(archive.read(metadata_names[0]))
    except (OSError, zipfile.BadZipFile) as exc:
        raise WheelhouseError("ai_wheelhouse_metadata") from exc
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise WheelhouseError("ai_wheelhouse_metadata")
    return canonicalize_name(name), version, message.get_all("Requires-Dist") or []


def _requirement_satisfied(requirement: Requirement, wheel: dict) -> bool:
    if canonicalize_name(requirement.name) != wheel["name"]:
        return False
    if requirement.specifier and Version(wheel["version"]) not in requirement.specifier:
        return False
    return True


def build_ai_wheelhouse_descriptor(
    wheelhouse_root: Path,
    *,
    requirements: list[str],
    python: str = "cp311",
    platform: str = "win_amd64",
) -> dict:
    wheels = []
    seen: set[str] = set()
    direct_names = {_normalise_requirement(requirement).split("==", 1)[0] for requirement in requirements}
    for path in sorted(wheelhouse_root.glob("*.whl")):
        parsed = _parse_wheel_filename(path)
        relative = path.relative_to(wheelhouse_root).as_posix()
        if parsed["name"] in seen:
            raise WheelhouseError("ai_wheelhouse_duplicate")
        seen.add(parsed["name"])
        wheels.append(
            {
                **parsed,
                "fileName": relative,
                "byteSize": path.stat().st_size,
                "sha256": _digest(path),
                "direct": parsed["name"] in direct_names,
                "transitive": parsed["name"] not in direct_names,
            }
        )
    if not wheels:
        raise WheelhouseError("ai_wheelhouse_release_descriptor_required")
    return {
        "schemaVersion": "vem-ai-worker-wheelhouse-release/v1",
        "target": "windows-x86_64" if platform == "win_amd64" else platform,
        "python": "3.11.9" if python == "cp311" else python,
        "source": "release-provided-wheelhouse",
        "directRequirements": sorted(_normalise_requirement(requirement) for requirement in requirements),
        "wheels": wheels,
    }


def _load_runtime_descriptor(path: Path | None) -> dict:
    if path is None:
        return load_ai_runtime_descriptor()
    return json.loads(path.read_text("utf-8"))


def _validate_release_descriptor(descriptor: dict, runtime_descriptor: dict) -> list[dict]:
    if set(descriptor) != {"schemaVersion", "target", "python", "source", "directRequirements", "wheels"}:
        raise WheelhouseError("ai_wheelhouse_descriptor_shape")
    if descriptor["schemaVersion"] != "vem-ai-worker-wheelhouse-release/v1":
        raise WheelhouseError("ai_wheelhouse_descriptor_schema")
    if descriptor["target"] != runtime_descriptor["target"] or descriptor["python"] != runtime_descriptor["python"]:
        raise WheelhouseError("ai_wheelhouse_target_mismatch")
    expected_direct = sorted(_normalise_requirement(requirement) for requirement in runtime_descriptor["directRequirements"])
    actual_direct = sorted(_normalise_requirement(requirement) for requirement in descriptor["directRequirements"])
    if actual_direct != expected_direct:
        raise WheelhouseError("ai_wheelhouse_direct_requirements_mismatch")
    wheels = descriptor["wheels"]
    if not isinstance(wheels, list) or not wheels:
        raise WheelhouseError("ai_wheelhouse_release_descriptor_required")
    return wheels


def generate_hashed_requirements(descriptor: dict, wheelhouse_root: Path) -> str:
    wheels = sorted(descriptor["wheels"], key=lambda wheel: (wheel["name"], wheel["version"], wheel["fileName"]))
    lines = []
    for wheel in wheels:
        path = wheelhouse_root / wheel["fileName"]
        if not path.is_file():
            raise WheelhouseError("ai_wheelhouse_missing")
        lines.append(f"{wheel['name']}=={wheel['version']} --hash=sha256:{wheel['sha256']}")
    return "\n".join(lines) + "\n"


def verify_ai_wheelhouse(
    descriptor_path: Path,
    wheelhouse_root: Path,
    *,
    runtime_descriptor_path: Path | None = None,
    requirements_output: Path | None = None,
) -> None:
    raw = descriptor_path.read_text("utf-8")
    descriptor = json.loads(raw)
    if canonical_json(descriptor) != raw.rstrip("\n"):
        raise WheelhouseError("ai_wheelhouse_descriptor_noncanonical")
    runtime_descriptor = _load_runtime_descriptor(runtime_descriptor_path)
    wheels = _validate_release_descriptor(descriptor, runtime_descriptor)
    seen: set[str] = set()
    seen_names: set[str] = set()
    by_name: dict[str, dict] = {}
    all_files = {
        path.relative_to(wheelhouse_root).as_posix()
        for path in wheelhouse_root.rglob("*")
        if path.is_file()
    }
    for wheel in wheels:
        if set(wheel) != {"fileName", "name", "version", "tags", "byteSize", "sha256", "direct", "transitive"}:
            raise WheelhouseError("ai_wheelhouse_entry_shape")
        relative = wheel["fileName"]
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "\\" in relative or ":" in relative or pure.as_posix() != relative:
            raise WheelhouseError("ai_wheelhouse_path")
        parsed = _parse_wheel_filename(Path(relative))
        if (
            parsed["name"] != wheel["name"]
            or parsed["version"] != wheel["version"]
            or parsed["tags"] != wheel["tags"]
        ):
            raise WheelhouseError("ai_wheelhouse_wheel_name")
        if not _wheel_tag_compatible(wheel["tags"], descriptor):
            raise WheelhouseError("ai_wheelhouse_target_mismatch")
        if relative in seen or wheel["name"] in seen_names:
            raise WheelhouseError("ai_wheelhouse_duplicate")
        seen.add(relative)
        seen_names.add(wheel["name"])
        path = (wheelhouse_root / pure).resolve()
        if wheelhouse_root.resolve() not in path.parents or not path.is_file():
            raise WheelhouseError("ai_wheelhouse_missing")
        if path.stat().st_size != wheel["byteSize"] or _digest(path) != wheel["sha256"]:
            raise WheelhouseError("ai_wheelhouse_digest")
        metadata_name, metadata_version, requirements = _read_wheel_metadata(path)
        if metadata_name != wheel["name"] or metadata_version != wheel["version"]:
            raise WheelhouseError("ai_wheelhouse_metadata")
        wheel["_requires_dist"] = requirements
        by_name[wheel["name"]] = wheel
    direct_names = {
        _normalise_requirement(requirement).split("==", 1)[0]
        for requirement in descriptor["directRequirements"]
    }
    wheel_direct_names = {wheel["name"] for wheel in wheels if wheel["direct"] and not wheel["transitive"]}
    if wheel_direct_names != direct_names:
        raise WheelhouseError("ai_wheelhouse_direct_requirements_mismatch")
    actual = all_files
    expected = {wheel["fileName"] for wheel in wheels}
    if actual != expected:
        raise WheelhouseError("ai_wheelhouse_extra_or_missing")
    env = _target_marker_environment(descriptor)
    needed = set(direct_names)
    queue = list(direct_names)
    while queue:
        name = queue.pop(0)
        wheel = by_name.get(name)
        if wheel is None:
            raise WheelhouseError("ai_wheelhouse_missing_dependency")
        for requirement_text in wheel.pop("_requires_dist", []):
            requirement = Requirement(requirement_text)
            if requirement.marker is not None and not requirement.marker.evaluate(env):
                continue
            dependency_name = canonicalize_name(requirement.name)
            dependency_wheel = by_name.get(dependency_name)
            if dependency_wheel is None or not _requirement_satisfied(requirement, dependency_wheel):
                raise WheelhouseError("ai_wheelhouse_missing_dependency")
            if dependency_name not in needed:
                needed.add(dependency_name)
                queue.append(dependency_name)
    if set(by_name) != needed:
        raise WheelhouseError("ai_wheelhouse_extra_unrelated")
    if requirements_output is not None:
        requirements_output.parent.mkdir(parents=True, exist_ok=True)
        requirements_output.write_text(generate_hashed_requirements(descriptor, wheelhouse_root), "utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-descriptor", action="store_true")
    parser.add_argument("--descriptor", default="requirements-ai.lock.json")
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--runtime-descriptor", default=None)
    parser.add_argument("--requirements-output", default=None)
    args = parser.parse_args()
    if args.build_descriptor:
        requirements = [
            line.strip()
            for line in Path("requirements-ai.txt").read_text("utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        descriptor = build_ai_wheelhouse_descriptor(
            Path(args.wheelhouse).resolve(),
            requirements=requirements,
        )
        Path(args.descriptor).write_text(canonical_json(descriptor), "utf-8")
    verify_ai_wheelhouse(
        Path(args.descriptor),
        Path(args.wheelhouse).resolve(),
        runtime_descriptor_path=Path(args.runtime_descriptor) if args.runtime_descriptor else None,
        requirements_output=Path(args.requirements_output) if args.requirements_output else None,
    )
    print("AI wheelhouse verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
