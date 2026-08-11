"""Frozen Windows entrypoint for pre-cutover AI artifact verification."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from PyInstaller.archive.readers import CArchiveReader

from scripts.ai_model_pack_release import install_model_pack_zip

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from scripts.verify_trusted_candidate_inputs import verify_inputs
from vision.ai_model_pack import canonical_ai_model_manifest_json
from vision.process_supervisor import run_supervised


PROOF_SCHEMA = "vending-vision-precutover-proof/v1"
MAX_CANDIDATE_BYTES = 8 * 1024 * 1024 * 1024
MAX_MODEL_BYTES = 8 * 1024 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_regular(source: Path, destination: Path, *, maximum_bytes: int, label: str) -> dict[str, object]:
    if not source.is_absolute() or source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise RuntimeError(f"{label}_regular_file")
    before = source.stat()
    if before.st_size <= 0 or before.st_size > maximum_bytes:
        raise RuntimeError(f"{label}_size")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as input_stream, destination.open("xb") as output:
            for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                size += len(chunk)
                if size > maximum_bytes:
                    raise RuntimeError(f"{label}_size")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = source.stat()
    identity = lambda stat: (
        stat.st_dev,
        stat.st_ino,
        stat.st_mode,
        stat.st_nlink,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    if identity(before) != identity(opened) or identity(before) != identity(after) or size != before.st_size:
        raise RuntimeError(f"{label}_changed")
    return {"byteSize": size, "sha256": digest.hexdigest()}


def _load_canonical(path: Path, label: str) -> dict:
    raw = path.read_text("utf-8")
    if len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raise RuntimeError(f"{label}_size")
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise RuntimeError(f"{label}_json") from exc
    if not isinstance(value, dict) or _canonical(value) != raw.rstrip("\n") or raw not in {_canonical(value), _canonical(value) + "\n"}:
        raise RuntimeError(f"{label}_noncanonical")
    return value


def _require_exact(value: dict, keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise RuntimeError(f"{label}_shape")


def _runtime_requirements(runtime: dict, lock_path: Path) -> dict[str, Requirement]:
    _require_exact(
        runtime,
        {
            "directRequirements",
            "python",
            "requirementsAiLockSha256",
            "requirementsAiSha256",
            "schemaVersion",
            "target",
            "workerLayout",
        },
        "runtime_descriptor",
    )
    if (
        runtime["schemaVersion"] != "vem-ai-runtime-descriptor/v1"
        or runtime["target"] != "windows-x86_64"
        or runtime["requirementsAiLockSha256"] != _sha256(lock_path)
        or not isinstance(runtime["directRequirements"], list)
    ):
        raise RuntimeError("runtime_descriptor_identity")
    result = {}
    for raw in runtime["directRequirements"]:
        requirement = Requirement(raw)
        name = canonicalize_name(requirement.name)
        if requirement.url or requirement.extras or requirement.marker or not requirement.specifier or name in result:
            raise RuntimeError("runtime_descriptor_requirement")
        result[name] = requirement
    return result


def _probe_payload(result, *, expected_probe: str, source_revision: str, requirements: dict[str, Requirement]) -> dict:
    if (
        result.returncode != 0
        or result.stdout_total > MAX_OUTPUT_BYTES
        or result.stderr_total > MAX_OUTPUT_BYTES
        or result.stderr_tail
    ):
        raise RuntimeError("precutover_worker_probe_failed")
    try:
        text = result.stdout_tail.decode("utf-8")
        lines = text.splitlines()
        if len(lines) != 1:
            raise ValueError("one JSON line required")
        payload = json.loads(lines[0])
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("precutover_worker_probe_json") from exc
    expected_keys = {"catvtonSourceRevision", "probe", *requirements}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RuntimeError("precutover_worker_probe_shape")
    if payload["probe"] != expected_probe or payload["catvtonSourceRevision"] != source_revision:
        raise RuntimeError("precutover_worker_probe_identity")
    for name, requirement in requirements.items():
        try:
            valid = isinstance(payload[name], str) and requirement.specifier.contains(
                Version(payload[name]), prereleases=True
            )
        except InvalidVersion:
            valid = False
        if not valid:
            raise RuntimeError(f"precutover_worker_dependency:{name}")
    return payload


def verify_frozen_worker_archive(worker: Path) -> None:
    try:
        archive = CArchiveReader(str(worker))
        entries = set(archive.toc)
        for name in tuple(entries):
            if name.lower().endswith(".pyz"):
                embedded = archive.open_embedded_archive(name)
                entries.update(embedded.toc)
    except Exception as exc:
        raise RuntimeError("precutover_worker_not_frozen") from exc
    required = {
        "vision.ai_attempt_worker",
        "vision.ai_model_pack",
        "vision.ai_runtime_descriptor",
        "vision.process_supervisor",
        "vision.source_provenance",
        "vision.vendor.catvton.model.pipeline",
    }
    if not required.issubset(entries):
        raise RuntimeError("precutover_worker_archive_modules")


def _write_exclusive(path: Path, value: dict) -> None:
    if not path.is_absolute() or path.exists() or path.parent.resolve() != path.parent:
        raise RuntimeError("precutover_report_path")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_precutover(
    *,
    candidate_artifact: Path,
    candidate_manifest: Path,
    github_attestation: Path,
    trusted_builder_evidence: Path,
    subject_sha256: str,
    manifest_sha256: str,
    attestation_bundle_sha256: str,
    source_commit: str,
    model_pack_archive: Path,
    model_pack_byte_size: int,
    model_pack_sha256: str,
    model_descriptor_sha256: str,
    private_parent: Path,
    report_output: Path,
    timeout: float,
    require_frozen_worker: bool = False,
) -> dict:
    for value, pattern, label in (
        (subject_sha256, SHA256_RE, "candidate_subject"),
        (manifest_sha256, SHA256_RE, "candidate_manifest"),
        (attestation_bundle_sha256, SHA256_RE, "candidate_attestation"),
        (model_pack_sha256, SHA256_RE, "model_pack"),
        (model_descriptor_sha256, SHA256_RE, "model_descriptor"),
        (source_commit, COMMIT_RE, "source_commit"),
    ):
        if pattern.fullmatch(value) is None:
            raise RuntimeError(f"{label}_identity")
    if (
        type(model_pack_byte_size) is not int
        or model_pack_byte_size <= 0
        or model_pack_byte_size > MAX_MODEL_BYTES
        or not isinstance(timeout, (int, float))
        or timeout <= 0
        or timeout > 300
        or not private_parent.is_absolute()
        or not private_parent.is_dir()
        or private_parent.is_symlink()
        or private_parent.resolve() != private_parent
    ):
        raise RuntimeError("precutover_external_identity")
    candidate_inputs = (
        candidate_artifact,
        candidate_manifest,
        github_attestation,
        trusted_builder_evidence,
    )
    candidate_parent = candidate_artifact.parent.resolve()
    if (
        any(path.parent.resolve() != candidate_parent for path in candidate_inputs)
        or {path.name for path in candidate_inputs}
        != {path.name for path in candidate_parent.iterdir()}
    ):
        raise RuntimeError("candidate_exact4_member_set")
    private_root = Path(tempfile.mkdtemp(prefix=".precutover-", dir=private_parent))
    try:
        inputs = private_root / "candidate-inputs"
        inputs.mkdir()
        staged = {}
        for name, source, cap in (
            ("artifact", candidate_artifact, MAX_CANDIDATE_BYTES),
            ("candidate-manifest.json", candidate_manifest, MAX_JSON_BYTES),
            ("github-build-provenance.sigstore.json", github_attestation, MAX_JSON_BYTES),
            ("trusted-builder-evidence.json", trusted_builder_evidence, MAX_JSON_BYTES),
        ):
            target = inputs / ("candidate.zip" if name == "artifact" else name)
            staged[name] = (target, _stage_regular(source, target, maximum_bytes=cap, label=f"candidate_{name}"))
        extracted = private_root / "candidate"
        verify_inputs(
            artifact=staged["artifact"][0],
            candidate_manifest=staged["candidate-manifest.json"][0],
            github_attestation=staged["github-build-provenance.sigstore.json"][0],
            trusted_builder_evidence=staged["trusted-builder-evidence.json"][0],
            destination=extracted,
            subject_sha256=subject_sha256,
            manifest_sha256=manifest_sha256,
            attestation_bundle_sha256=attestation_bundle_sha256,
            source_commit=source_commit,
        )
        model_archive = private_root / "model-pack.zip"
        model_facts = _stage_regular(
            model_pack_archive,
            model_archive,
            maximum_bytes=MAX_MODEL_BYTES,
            label="model_pack_archive",
        )
        if model_facts != {"byteSize": model_pack_byte_size, "sha256": model_pack_sha256}:
            raise RuntimeError("model_pack_external_identity")
        manifest = _load_canonical(staged["candidate-manifest.json"][0], "candidate_manifest")
        bindings = manifest.get("bindings")
        if not isinstance(bindings, dict):
            raise RuntimeError("candidate_bindings")
        worker = extracted / bindings["workerExecutable"]["path"]
        internal = worker.parent / "_internal"
        model_descriptor_path = extracted / bindings["modelPackDescriptor"]["path"]
        runtime_path = extracted / bindings["runtimeDescriptor"]["path"]
        lock_path = extracted / bindings["aiLock"]["path"]
        source_path = extracted / bindings["sourceDescriptor"]["path"]
        if _sha256(model_descriptor_path) != model_descriptor_sha256:
            raise RuntimeError("model_descriptor_external_identity")
        if require_frozen_worker:
            verify_frozen_worker_archive(worker)
        model_descriptor = _load_canonical(model_descriptor_path, "model_descriptor")
        if canonical_ai_model_manifest_json(model_descriptor) != model_descriptor_path.read_text("utf-8"):
            raise RuntimeError("model_descriptor_noncanonical")
        source = _load_canonical(source_path, "source_descriptor")
        _require_exact(source, {"catvtonSourceRevision", "schemaVersion", "sources"}, "source_descriptor")
        source_revision = source["catvtonSourceRevision"]
        if (
            source["schemaVersion"] != "vem-official-ai-source-descriptor/v1"
            or model_descriptor.get("catvtonSourceRevision") != source_revision
        ):
            raise RuntimeError("model_source_identity")
        runtime = _load_canonical(runtime_path, "runtime_descriptor")
        requirements = _runtime_requirements(runtime, lock_path)
        installed = install_model_pack_zip(
            model_archive,
            private_root / "model-install",
            model_descriptor,
            outer_sha256=model_pack_sha256,
        )
        worker_command = [str(worker)] if os.name == "nt" else [sys.executable, str(worker)]
        runtime_result = asyncio.run(
            run_supervised([*worker_command, "--probe-runtime"], timeout=timeout)
        )
        runtime_probe = _probe_payload(
            runtime_result,
            expected_probe="official-catvton-worker-runtime",
            source_revision=source_revision,
            requirements=requirements,
        )
        model_result = asyncio.run(
            run_supervised(
                [*worker_command, "--model-pack", str(installed), "--probe"],
                timeout=timeout,
            )
        )
        model_probe = _probe_payload(
            model_result,
            expected_probe="official-catvton-worker",
            source_revision=source_revision,
            requirements=requirements,
        )
        report = {
            "candidate": {
                "attestationBundleSha256": attestation_bundle_sha256,
                "embeddedManifestSha256": manifest_sha256,
                "sourceCommit": source_commit,
                "subjectSha256": subject_sha256,
                "workerExecutableSha256": _sha256(worker),
                "workerMode": "frozen-windows" if require_frozen_worker else "source-test-only",
            },
            "modelPack": {
                "archive": model_facts,
                "descriptorSha256": model_descriptor_sha256,
                "sourceRevision": source_revision,
            },
            "probes": {"model": model_probe, "runtime": runtime_probe},
            "resources": {
                "aiLockSha256": _sha256(lock_path),
                "runtimeDescriptorSha256": _sha256(runtime_path),
                "sourceDescriptorSha256": _sha256(source_path),
            },
            "schemaVersion": PROOF_SCHEMA,
        }
        _write_exclusive(report_output, report)
        return report
    finally:
        shutil.rmtree(private_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--github-attestation", required=True)
    parser.add_argument("--trusted-builder-evidence", required=True)
    parser.add_argument("--subject-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--attestation-bundle-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-pack-archive", required=True)
    parser.add_argument("--model-pack-byte-size", required=True, type=int)
    parser.add_argument("--model-pack-sha256", required=True)
    parser.add_argument("--model-descriptor-sha256", required=True)
    parser.add_argument("--private-parent", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    if os.name != "nt":
        raise SystemExit("PRECUTOVER_COMPANION=FAIL:windows_required")
    try:
        verify_precutover(
            candidate_artifact=Path(args.candidate_artifact).resolve(),
            candidate_manifest=Path(args.candidate_manifest).resolve(),
            github_attestation=Path(args.github_attestation).resolve(),
            trusted_builder_evidence=Path(args.trusted_builder_evidence).resolve(),
            subject_sha256=args.subject_sha256,
            manifest_sha256=args.manifest_sha256,
            attestation_bundle_sha256=args.attestation_bundle_sha256,
            source_commit=args.source_commit,
            model_pack_archive=Path(args.model_pack_archive).resolve(),
            model_pack_byte_size=args.model_pack_byte_size,
            model_pack_sha256=args.model_pack_sha256,
            model_descriptor_sha256=args.model_descriptor_sha256,
            private_parent=Path(args.private_parent).resolve(),
            report_output=Path(args.report_output).resolve(),
            timeout=args.timeout,
            require_frozen_worker=True,
        )
    except Exception as exc:
        raise SystemExit(f"PRECUTOVER_COMPANION=FAIL:{exc}") from exc
    print("PRECUTOVER_COMPANION=PASS")
    return 0
