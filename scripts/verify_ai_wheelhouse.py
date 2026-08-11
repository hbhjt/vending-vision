from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


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
        "name": parts[0].replace("_", "-").lower(),
        "version": parts[1],
        "tag": "-".join(parts[2:]),
    }


def build_ai_wheelhouse_descriptor(
    wheelhouse_root: Path,
    *,
    requirements: list[str],
    python: str = "cp311",
    platform: str = "win_amd64",
) -> dict:
    wheels = []
    seen: set[str] = set()
    for path in sorted(wheelhouse_root.glob("*.whl")):
        parsed = _parse_wheel_filename(path)
        relative = path.relative_to(wheelhouse_root).as_posix()
        if parsed["name"] in seen:
            raise WheelhouseError("ai_wheelhouse_duplicate")
        seen.add(parsed["name"])
        wheels.append(
            {
                **parsed,
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _digest(path),
            }
        )
    if not wheels:
        raise WheelhouseError("ai_wheelhouse_release_descriptor_required")
    return {
        "schemaVersion": "vem-ai-worker-wheelhouse/v1",
        "platform": platform,
        "python": python,
        "source": "materialized-release-wheelhouse",
        "requirements": requirements,
        "wheels": wheels,
    }


def verify_ai_wheelhouse(descriptor_path: Path, wheelhouse_root: Path) -> None:
    raw = descriptor_path.read_text("utf-8")
    descriptor = json.loads(raw)
    if canonical_json(descriptor) != raw.rstrip("\n"):
        raise WheelhouseError("ai_wheelhouse_descriptor_noncanonical")
    if set(descriptor) != {"schemaVersion", "platform", "python", "source", "requirements", "wheels"}:
        raise WheelhouseError("ai_wheelhouse_descriptor_shape")
    if descriptor["schemaVersion"] != "vem-ai-worker-wheelhouse/v1":
        raise WheelhouseError("ai_wheelhouse_descriptor_schema")
    wheels = descriptor["wheels"]
    if not isinstance(wheels, list) or not wheels:
        raise WheelhouseError("ai_wheelhouse_release_descriptor_required")
    seen: set[str] = set()
    for wheel in wheels:
        if set(wheel) != {"name", "version", "tag", "path", "size", "sha256"}:
            raise WheelhouseError("ai_wheelhouse_entry_shape")
        relative = wheel["path"]
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "\\" in relative or ":" in relative or pure.as_posix() != relative:
            raise WheelhouseError("ai_wheelhouse_path")
        if relative in seen:
            raise WheelhouseError("ai_wheelhouse_duplicate")
        seen.add(relative)
        path = (wheelhouse_root / pure).resolve()
        if wheelhouse_root.resolve() not in path.parents or not path.is_file():
            raise WheelhouseError("ai_wheelhouse_missing")
        if path.stat().st_size != wheel["size"] or _digest(path) != wheel["sha256"]:
            raise WheelhouseError("ai_wheelhouse_digest")
    actual = {path.name for path in wheelhouse_root.glob("*.whl")}
    expected = {Path(wheel["path"]).name for wheel in wheels}
    if actual != expected:
        raise WheelhouseError("ai_wheelhouse_extra_or_missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-descriptor", action="store_true")
    parser.add_argument("--descriptor", default="requirements-ai.lock.json")
    parser.add_argument("--wheelhouse", required=True)
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
    verify_ai_wheelhouse(Path(args.descriptor), Path(args.wheelhouse).resolve())
    print("AI wheelhouse verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
