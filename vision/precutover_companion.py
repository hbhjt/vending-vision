"""Frozen Windows entrypoint for pre-cutover AI artifact verification."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Callable

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
MAX_HELD_FILES = 20_000
MAX_HELD_BYTES = MAX_CANDIDATE_BYTES + MAX_MODEL_BYTES
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


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _fd_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    if not hasattr(os, "pread"):
        duplicate = os.dup(descriptor)
        with os.fdopen(duplicate, "rb") as stream:
            stream.seek(0)
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


class _WindowsFileApi:
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x00000080

    def __init__(self, kernel32=None):
        if kernel32 is None:
            if os.name != "nt":
                raise RuntimeError("precutover_windows_file_api")
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateFileW.argtypes = [
                ctypes.c_wchar_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
            ]
            kernel32.CreateFileW.restype = ctypes.c_void_p
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
        self._kernel32 = kernel32

    def open_read_lease(self, path: Path):
        handle = self._kernel32.CreateFileW(
            str(path),
            self.GENERIC_READ,
            self.FILE_SHARE_READ,
            None,
            self.OPEN_EXISTING,
            self.FILE_ATTRIBUTE_NORMAL,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in {None, 0, -1, invalid}:
            raise RuntimeError("precutover_integrity_handle_open")
        return handle

    def close(self, handle) -> None:
        if not self._kernel32.CloseHandle(handle):
            raise RuntimeError("precutover_integrity_handle_close")


class _PosixChangeMonitor:
    _MASK = (
        0x00000002
        | 0x00000004
        | 0x00000008
        | 0x00000040
        | 0x00000080
        | 0x00000100
        | 0x00000200
        | 0x00000400
        | 0x00000800
    )

    def __init__(self):
        if not sys.platform.startswith("linux"):
            raise RuntimeError("precutover_integrity_monitor_unavailable")
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.inotify_init1.argtypes = [ctypes.c_int]
        self._libc.inotify_init1.restype = ctypes.c_int
        self._libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self._libc.inotify_add_watch.restype = ctypes.c_int
        self._descriptor = self._libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self._descriptor < 0:
            raise RuntimeError("precutover_integrity_monitor_open")
        self._labels: dict[int, str] = {}

    def add(self, path: Path, label: str) -> None:
        watch = self._libc.inotify_add_watch(
            self._descriptor,
            os.fsencode(path),
            self._MASK,
        )
        if watch < 0 or watch in self._labels:
            raise RuntimeError(f"precutover_integrity_monitor_add:{label}")
        self._labels[watch] = label

    def verify(self) -> None:
        try:
            changed = os.read(self._descriptor, 1024 * 1024)
        except BlockingIOError:
            return
        if changed:
            raise RuntimeError("precutover_integrity_write_event")

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1


@dataclass(frozen=True)
class _ExpectedFile:
    path: Path
    byte_size: int
    sha256: str
    label: str


@dataclass
class _HeldFile:
    expected: _ExpectedFile
    identity: tuple[int, ...]
    descriptor: int | None = None
    windows_handle: object | None = None


class _IntegrityFence:
    def __init__(self, expected: list[_ExpectedFile], *, exact_root: Path | None = None):
        if (
            not expected
            or len(expected) > MAX_HELD_FILES
            or sum(item.byte_size for item in expected) > MAX_HELD_BYTES
            or len({item.path for item in expected}) != len(expected)
        ):
            raise RuntimeError("precutover_integrity_bounds")
        self._expected = expected
        self._exact_root = exact_root
        self._held: list[_HeldFile] = []
        self._windows = _WindowsFileApi() if os.name == "nt" else None
        self._monitor = _PosixChangeMonitor() if os.name != "nt" else None

    def __enter__(self):
        try:
            if self._exact_root is not None:
                if (
                    not self._exact_root.is_absolute()
                    or self._exact_root.is_symlink()
                    or not self._exact_root.is_dir()
                    or self._exact_root.resolve() != self._exact_root
                ):
                    raise RuntimeError("precutover_integrity_tree_root")
                if self._monitor is not None:
                    directories = [self._exact_root]
                    directories.extend(
                        path for path in self._exact_root.rglob("*") if path.is_dir()
                    )
                    for index, directory in enumerate(directories):
                        self._monitor.add(directory, f"tree_directory:{index}")
            for expected in self._expected:
                path = expected.path
                if (
                    not path.is_absolute()
                    or path.is_symlink()
                    or not path.is_file()
                    or path.resolve() != path
                    or type(expected.byte_size) is not int
                    or expected.byte_size < 0
                    or SHA256_RE.fullmatch(expected.sha256) is None
                ):
                    raise RuntimeError(f"precutover_integrity_open:{expected.label}")
                before = path.stat()
                if not stat.S_ISREG(before.st_mode) or before.st_size != expected.byte_size:
                    raise RuntimeError(f"precutover_integrity_size:{expected.label}")
                if self._windows is not None:
                    handle = self._windows.open_read_lease(path)
                    held = _HeldFile(expected, _stat_identity(before), windows_handle=handle)
                else:
                    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    opened = os.fstat(descriptor)
                    if _stat_identity(before) != _stat_identity(opened):
                        os.close(descriptor)
                        raise RuntimeError(f"precutover_integrity_changed:{expected.label}")
                    held = _HeldFile(expected, _stat_identity(before), descriptor=descriptor)
                self._held.append(held)
                if self._monitor is not None:
                    self._monitor.add(path, expected.label)
            self.verify()
            return self
        except Exception:
            self.close()
            raise

    def verify(self) -> None:
        if self._monitor is not None:
            self._monitor.verify()
        if self._exact_root is not None:
            expected_paths = {
                item.path.relative_to(self._exact_root).as_posix() for item in self._expected
            }
            actual_paths = set()
            for path in self._exact_root.rglob("*"):
                if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                    raise RuntimeError("precutover_integrity_tree_member")
                if path.is_file():
                    actual_paths.add(path.relative_to(self._exact_root).as_posix())
            if actual_paths != expected_paths:
                raise RuntimeError("precutover_integrity_tree_set")
        for held in self._held:
            expected = held.expected
            try:
                current = expected.path.stat()
            except OSError as exc:
                raise RuntimeError(f"precutover_integrity_missing:{expected.label}") from exc
            if _stat_identity(current) != held.identity:
                raise RuntimeError(f"precutover_integrity_changed:{expected.label}")
            if held.descriptor is not None:
                if _stat_identity(os.fstat(held.descriptor)) != held.identity:
                    raise RuntimeError(f"precutover_integrity_changed:{expected.label}")
                digest = _fd_sha256(held.descriptor)
            else:
                digest = _sha256(expected.path)
            if digest != expected.sha256:
                raise RuntimeError(f"precutover_integrity_digest:{expected.label}")

    def close(self) -> None:
        failure = None
        for held in reversed(self._held):
            try:
                if held.descriptor is not None:
                    os.close(held.descriptor)
                elif held.windows_handle is not None and self._windows is not None:
                    self._windows.close(held.windows_handle)
            except Exception as exc:
                failure = failure or exc
        self._held.clear()
        if self._monitor is not None:
            self._monitor.close()
        if failure is not None:
            raise failure

    def __exit__(self, _type, _value, _traceback):
        self.close()


def _snapshot_expectation(
    path: Path,
    label: str,
    *,
    maximum_bytes: int,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> _ExpectedFile:
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise RuntimeError(f"precutover_integrity_source:{label}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size <= 0
            or opened.st_size > maximum_bytes
        ):
            raise RuntimeError(f"precutover_integrity_source:{label}")
        digest = _fd_sha256(descriptor)
    finally:
        os.close(descriptor)
    current = path.stat()
    if _stat_identity(opened) != _stat_identity(current):
        raise RuntimeError(f"precutover_integrity_changed:{label}")
    if expected_size is not None and opened.st_size != expected_size:
        raise RuntimeError(f"precutover_integrity_size:{label}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"precutover_integrity_digest:{label}")
    return _ExpectedFile(path, opened.st_size, digest, label)


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


def _write_exclusive(path: Path, value: dict, *, before_link) -> None:
    if not path.is_absolute() or path.exists() or path.parent.resolve() != path.parent:
        raise RuntimeError("precutover_report_path")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        before_link()
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
    _test_phase_hook: Callable[[str, dict[str, Path]], None] | None = None,
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
    if require_frozen_worker and _test_phase_hook is not None:
        raise RuntimeError("precutover_test_hook_forbidden")
    candidate_inputs = (
        candidate_artifact,
        candidate_manifest,
        github_attestation,
        trusted_builder_evidence,
    )
    candidate_parent = candidate_artifact.parent.resolve()
    candidate_names = {path.name for path in candidate_inputs}
    if (
        any(path.parent.resolve() != candidate_parent for path in candidate_inputs)
        or candidate_names != {path.name for path in candidate_parent.iterdir()}
    ):
        raise RuntimeError("candidate_exact4_member_set")
    source_expectations = [
        _snapshot_expectation(
            candidate_artifact,
            "source_candidate",
            maximum_bytes=MAX_CANDIDATE_BYTES,
            expected_sha256=subject_sha256,
        ),
        _snapshot_expectation(
            candidate_manifest,
            "source_manifest",
            maximum_bytes=MAX_JSON_BYTES,
            expected_sha256=manifest_sha256,
        ),
        _snapshot_expectation(
            github_attestation,
            "source_attestation",
            maximum_bytes=MAX_JSON_BYTES,
            expected_sha256=attestation_bundle_sha256,
        ),
        _snapshot_expectation(
            trusted_builder_evidence,
            "source_builder_evidence",
            maximum_bytes=MAX_JSON_BYTES,
        ),
        _snapshot_expectation(
            model_pack_archive,
            "source_model_archive",
            maximum_bytes=MAX_MODEL_BYTES,
            expected_size=model_pack_byte_size,
            expected_sha256=model_pack_sha256,
        ),
    ]
    private_root = Path(tempfile.mkdtemp(prefix=".precutover-", dir=private_parent))
    try:
        with ExitStack() as leases:
            fences = [leases.enter_context(_IntegrityFence(source_expectations))]

            def verify_all() -> None:
                if {path.name for path in candidate_parent.iterdir()} != candidate_names:
                    raise RuntimeError("precutover_integrity_source_member_set")
                for fence in fences:
                    fence.verify()

            inputs = private_root / "candidate-inputs"
            inputs.mkdir()
            staged = {}
            for name, source_file, cap in (
                ("artifact", candidate_artifact, MAX_CANDIDATE_BYTES),
                ("candidate-manifest.json", candidate_manifest, MAX_JSON_BYTES),
                ("github-build-provenance.sigstore.json", github_attestation, MAX_JSON_BYTES),
                ("trusted-builder-evidence.json", trusted_builder_evidence, MAX_JSON_BYTES),
            ):
                target = inputs / ("candidate.zip" if name == "artifact" else name)
                staged[name] = (
                    target,
                    _stage_regular(source_file, target, maximum_bytes=cap, label=f"candidate_{name}"),
                )
            fences.append(
                leases.enter_context(
                    _IntegrityFence(
                        [
                            _ExpectedFile(path, facts["byteSize"], facts["sha256"], f"staged_{name}")
                            for name, (path, facts) in staged.items()
                        ],
                        exact_root=inputs,
                    )
                )
            )
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
            fences.append(
                leases.enter_context(
                    _IntegrityFence(
                        [
                            _ExpectedFile(
                                model_archive,
                                model_facts["byteSize"],
                                model_facts["sha256"],
                                "staged_model_archive",
                            )
                        ]
                    )
                )
            )
            manifest = _load_canonical(staged["candidate-manifest.json"][0], "candidate_manifest")
            bindings = manifest.get("bindings")
            if not isinstance(bindings, dict):
                raise RuntimeError("candidate_bindings")
            payload_expectations = [
                _ExpectedFile(
                    (extracted / item["path"]).resolve(),
                    item["size"],
                    item["sha256"],
                    f"candidate_payload:{item['path']}",
                )
                for item in manifest["files"]
            ]
            fences.append(
                leases.enter_context(_IntegrityFence(payload_expectations, exact_root=extracted))
            )
            worker_binding = bindings["workerExecutable"]
            worker = extracted / worker_binding["path"]
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
            installed_expectations = [
                _ExpectedFile(
                    (installed / "ai-model-manifest.json").resolve(),
                    model_descriptor_path.stat().st_size,
                    model_descriptor_sha256,
                    "installed_model:ai-model-manifest.json",
                ),
                *[
                    _ExpectedFile(
                        (installed / item["path"]).resolve(),
                        item["byteSize"],
                        item["sha256"],
                        f"installed_model:{item['path']}",
                    )
                    for item in model_descriptor["files"]
                ],
            ]
            fences.append(
                leases.enter_context(
                    _IntegrityFence(installed_expectations, exact_root=installed)
                )
            )
            observed_paths = {
                "ai_lock": lock_path,
                "installed_model": installed / model_descriptor["files"][0]["path"],
                "model_descriptor": model_descriptor_path,
                "runtime_descriptor": runtime_path,
                "source_attestation": github_attestation,
                "source_candidate": candidate_artifact,
                "source_descriptor": source_path,
                "source_evidence": trusted_builder_evidence,
                "source_manifest": candidate_manifest,
                "source_model_archive": model_pack_archive,
                "staged_model_archive": model_archive,
                "worker": worker,
                "worker_resource": internal / "runtime-resource.dll",
            }
            worker_command = [str(worker)] if os.name == "nt" else [sys.executable, str(worker)]
            if _test_phase_hook is not None:
                _test_phase_hook("before-runtime-probe", observed_paths)
            verify_all()
            runtime_result = asyncio.run(
                run_supervised([*worker_command, "--probe-runtime"], timeout=timeout)
            )
            verify_all()
            runtime_probe = _probe_payload(
                runtime_result,
                expected_probe="official-catvton-worker-runtime",
                source_revision=source_revision,
                requirements=requirements,
            )
            if _test_phase_hook is not None:
                _test_phase_hook("between-probes", observed_paths)
            verify_all()
            model_result = asyncio.run(
                run_supervised(
                    [*worker_command, "--model-pack", str(installed), "--probe"],
                    timeout=timeout,
                )
            )
            verify_all()
            model_probe = _probe_payload(
                model_result,
                expected_probe="official-catvton-worker",
                source_revision=source_revision,
                requirements=requirements,
            )
            if _test_phase_hook is not None:
                _test_phase_hook("before-receipt", observed_paths)
            if _sha256(worker) != worker_binding["sha256"]:
                raise RuntimeError("precutover_integrity_worker_binding")
            report = {
                "candidate": {
                    "attestationBundleSha256": attestation_bundle_sha256,
                    "embeddedManifestSha256": manifest_sha256,
                    "sourceCommit": source_commit,
                    "subjectSha256": subject_sha256,
                    "workerExecutableSha256": worker_binding["sha256"],
                    "workerMode": "frozen-windows" if require_frozen_worker else "source-test-only",
                },
                "modelPack": {
                    "archive": model_facts,
                    "descriptorSha256": model_descriptor_sha256,
                    "sourceRevision": source_revision,
                },
                "probes": {"model": model_probe, "runtime": runtime_probe},
                "resources": {
                    "aiLockSha256": bindings["aiLock"]["sha256"],
                    "runtimeDescriptorSha256": bindings["runtimeDescriptor"]["sha256"],
                    "sourceDescriptorSha256": bindings["sourceDescriptor"]["sha256"],
                },
                "schemaVersion": PROOF_SCHEMA,
            }
            verify_all()
            _write_exclusive(report_output, report, before_link=verify_all)
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
