from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


class MaterializeError(RuntimeError):
    pass


_SAFE_NAME = re.compile(r"^[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*$")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9]+(?:[.!+_-][A-Za-z0-9]+)*$")
_MAX_TOTAL_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_download_url(url: str, file_name: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise MaterializeError("ai_wheelhouse_download_url")
    if unquote(PurePosixPath(parsed.path).name) != file_name:
        raise MaterializeError("ai_wheelhouse_download_identity")
    host = (parsed.hostname or "").lower()
    if host == "files.pythonhosted.org":
        return "pypi"
    if host in {"download.pytorch.org", "download-r2.pytorch.org"}:
        return "pytorch-cpu"
    raise MaterializeError("ai_wheelhouse_download_source")


def _load_bootstrap_descriptor(
    descriptor_path: Path,
    runtime_descriptor_path: Path | None,
) -> tuple[dict, list[dict]]:
    try:
        raw = descriptor_path.read_text("utf-8")
        descriptor = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise MaterializeError("ai_wheelhouse_descriptor_invalid") from exc
    if _canonical_json(descriptor) != raw.rstrip("\n"):
        raise MaterializeError("ai_wheelhouse_descriptor_noncanonical")
    if set(descriptor) != {
        "schemaVersion", "target", "python", "source", "directRequirements", "wheels"
    }:
        raise MaterializeError("ai_wheelhouse_descriptor_shape")
    if descriptor["schemaVersion"] != "vem-ai-worker-wheelhouse-release/v1":
        raise MaterializeError("ai_wheelhouse_descriptor_schema")
    if descriptor["source"] != "pip-report-locked-wheelhouse":
        raise MaterializeError("ai_wheelhouse_descriptor_source")
    wheels = descriptor["wheels"]
    if not isinstance(wheels, list) or not wheels:
        raise MaterializeError("ai_wheelhouse_release_descriptor_required")

    if runtime_descriptor_path is not None:
        try:
            runtime_raw = runtime_descriptor_path.read_text("utf-8")
            runtime = json.loads(runtime_raw)
        except (OSError, ValueError) as exc:
            raise MaterializeError("ai_wheelhouse_runtime_descriptor_invalid") from exc
        if _canonical_json(runtime) != runtime_raw.rstrip("\n"):
            raise MaterializeError("ai_wheelhouse_runtime_descriptor_noncanonical")
        if set(runtime) != {
            "schemaVersion", "target", "python", "directRequirements",
            "requirementsAiSha256", "requirementsAiLockSha256", "workerLayout",
        } or runtime["schemaVersion"] != "vem-ai-runtime-descriptor/v1":
            raise MaterializeError("ai_wheelhouse_runtime_descriptor_shape")
        if hashlib.sha256(descriptor_path.read_bytes()).hexdigest() != runtime["requirementsAiLockSha256"]:
            raise MaterializeError("ai_wheelhouse_runtime_lock_mismatch")
        requirements_path = runtime_descriptor_path.with_name("requirements-ai.txt")
        try:
            requirements_digest = hashlib.sha256(requirements_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise MaterializeError("ai_wheelhouse_runtime_requirements_missing") from exc
        if requirements_digest != runtime["requirementsAiSha256"]:
            raise MaterializeError("ai_wheelhouse_runtime_requirements_mismatch")
        if descriptor["target"] != runtime["target"] or descriptor["python"] != runtime["python"]:
            raise MaterializeError("ai_wheelhouse_runtime_target_mismatch")
        if sorted(descriptor["directRequirements"]) != sorted(runtime["directRequirements"]):
            raise MaterializeError("ai_wheelhouse_runtime_requirements_semantics")

    seen: set[str] = set()
    for wheel in wheels:
        if not isinstance(wheel, dict) or set(wheel) != {
            "fileName", "name", "version", "tags", "byteSize", "sha256",
            "direct", "transitive", "url", "source",
        }:
            raise MaterializeError("ai_wheelhouse_entry_shape")
        file_name = wheel["fileName"]
        if (
            not isinstance(file_name, str)
            or PurePosixPath(file_name).name != file_name
            or not file_name.endswith(".whl")
        ):
            raise MaterializeError("ai_wheelhouse_path")
        if file_name in seen:
            raise MaterializeError("ai_wheelhouse_duplicate")
        seen.add(file_name)
        if (
            not isinstance(wheel["name"], str)
            or _SAFE_NAME.fullmatch(wheel["name"]) is None
            or not isinstance(wheel["version"], str)
            or _SAFE_VERSION.fullmatch(wheel["version"]) is None
            or type(wheel["byteSize"]) is not int
            or wheel["byteSize"] < 0
            or not isinstance(wheel["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"]) is None
            or type(wheel["direct"]) is not bool
            or type(wheel["transitive"]) is not bool
            or wheel["direct"] == wheel["transitive"]
        ):
            raise MaterializeError("ai_wheelhouse_entry_value")
        source = _validate_download_url(wheel["url"], file_name)
        if source != wheel["source"]:
            raise MaterializeError("ai_wheelhouse_download_source")
    return descriptor, wheels


def _hashed_requirements(wheels: list[dict]) -> str:
    return "".join(
        f"{wheel['name']}=={wheel['version']} --hash=sha256:{wheel['sha256']}\n"
        for wheel in sorted(wheels, key=lambda item: (item["name"], item["version"], item["fileName"]))
    )


def _write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_existing_ai_wheelhouse(
    descriptor_path: Path,
    wheelhouse: Path,
    runtime_descriptor_path: Path,
    requirements_output: Path,
) -> None:
    _descriptor, wheels = _load_bootstrap_descriptor(descriptor_path, runtime_descriptor_path)
    if not wheelhouse.is_dir():
        raise MaterializeError("ai_wheelhouse_missing")
    expected = {wheel["fileName"] for wheel in wheels}
    actual = {path.name for path in wheelhouse.iterdir()}
    if actual != expected:
        raise MaterializeError("ai_wheelhouse_extra_or_missing")
    for wheel in wheels:
        path = wheelhouse / wheel["fileName"]
        if path.is_symlink() or not path.is_file():
            raise MaterializeError("ai_wheelhouse_file_type")
        if path.stat().st_size != wheel["byteSize"] or _sha256_file(path) != wheel["sha256"]:
            raise MaterializeError("ai_wheelhouse_digest")
    _write_atomic(requirements_output, _hashed_requirements(wheels))


def materialize_ai_wheelhouse(
    descriptor_path: Path,
    destination: Path,
    *,
    opener=urlopen,
    runtime_descriptor_path: Path | None = None,
    max_total_download_bytes: int = _MAX_TOTAL_DOWNLOAD_BYTES,
) -> None:
    if destination.exists():
        raise MaterializeError("ai_wheelhouse_destination_exists")
    _descriptor, wheels = _load_bootstrap_descriptor(descriptor_path, runtime_descriptor_path)
    if (
        type(max_total_download_bytes) is not int
        or max_total_download_bytes <= 0
        or sum(wheel["byteSize"] for wheel in wheels) > max_total_download_bytes
    ):
        raise MaterializeError("ai_wheelhouse_download_size")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    seen: set[str] = set()
    total_downloaded = 0
    try:
        for wheel in wheels:
            file_name = wheel["fileName"]
            seen.add(file_name)

            request = Request(wheel["url"], headers={"User-Agent": "vem-ai-wheelhouse-materializer/1"})
            target = staging / file_name
            digest = hashlib.sha256()
            byte_size = 0
            with opener(request, timeout=120.0) as response:
                if response.geturl() != wheel["url"]:
                    raise MaterializeError("ai_wheelhouse_redirect_identity")
                with target.open("xb") as output:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        if (
                            byte_size + len(chunk) > wheel["byteSize"]
                            or total_downloaded + len(chunk) > max_total_download_bytes
                        ):
                            raise MaterializeError("ai_wheelhouse_download_size")
                        output.write(chunk)
                        digest.update(chunk)
                        byte_size += len(chunk)
                        total_downloaded += len(chunk)
            if byte_size != wheel["byteSize"]:
                raise MaterializeError("ai_wheelhouse_download_size")
            if digest.hexdigest() != wheel["sha256"]:
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
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--destination")
    destination.add_argument("--wheelhouse")
    parser.add_argument("--runtime-descriptor", default="ai-runtime-descriptor.json")
    parser.add_argument("--requirements-output")
    args = parser.parse_args()
    descriptor_path = Path(args.descriptor)
    runtime_path = Path(args.runtime_descriptor)
    if args.destination:
        materialize_ai_wheelhouse(
            descriptor_path,
            Path(args.destination).resolve(),
            runtime_descriptor_path=runtime_path,
        )
        if args.requirements_output:
            prepare_existing_ai_wheelhouse(
                descriptor_path,
                Path(args.destination).resolve(),
                runtime_path,
                Path(args.requirements_output),
            )
        print("AI wheelhouse materialized")
    else:
        if not args.requirements_output:
            parser.error("--wheelhouse requires --requirements-output")
        prepare_existing_ai_wheelhouse(
            descriptor_path,
            Path(args.wheelhouse).resolve(),
            runtime_path,
            Path(args.requirements_output),
        )
        print("AI wheelhouse bootstrap verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
