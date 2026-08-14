from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import pytest

from scripts.ai_model_pack_release import build_model_pack_zip
from scripts.candidate_artifact_manifest import write_candidate_archive
import vision.precutover_companion as companion
from vision.ai_model_pack import canonical_ai_model_manifest_json
from vision.precutover_companion import (
    _ExpectedFile,
    _fd_sha256,
    _IntegrityFence,
    _WindowsFileApi,
    _worker_probe_command,
)
from vision.precutover_companion import verify_frozen_worker_archive, verify_precutover
from vision.precutover_companion import main as companion_main


SOURCE_COMMIT = "a" * 40
SOURCE_REVISION = "3b795364a4d2f3b5adb365f39cdea376d20bc53c"
TRUSTED_BUILDER_COMMIT = "691b5056e8b9bf2667bc527b2170780b05863946"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_fixture(
    root: Path,
    *,
    torch_version: str = "2.8.0+cpu",
    worker_script: str | None = None,
    model_descriptor_suffix: str = "",
) -> dict[str, object]:
    dist = root / "dist"
    main = dist / "vending-vision/vending-vision.exe"
    worker = dist / "vending-vision-ai-worker/vending-vision-ai-worker.exe"
    internal = worker.parent / "_internal"
    main.parent.mkdir(parents=True)
    internal.mkdir(parents=True)
    main.write_bytes(b"test main")
    worker.write_text(
        worker_script
        or (
            "#!/usr/bin/env python3\n"
            "import json,sys\n"
            f"payload={{'catvtonSourceRevision':{SOURCE_REVISION!r},"
            f"'torch':{torch_version!r},'torchvision':'0.23.0+cpu',"
            "'diffusers':'0.29.2','transformers':'4.53.3'}\n"
            "payload['probe']='official-catvton-worker-runtime' if '--probe-runtime' in sys.argv else 'official-catvton-worker'\n"
            "print(json.dumps(payload,sort_keys=True))\n"
        ),
        "utf-8",
    )
    worker.chmod(0o700)
    (internal / "runtime-resource.dll").write_bytes(b"test worker runtime resource")
    requirements = [
        "torch==2.8.0+cpu",
        "torchvision==0.23.0+cpu",
        "diffusers==0.29.2",
        "transformers==4.53.3",
    ]
    lock = internal / "requirements-ai.lock.json"
    lock.write_text(canonical({"directRequirements": requirements}) + "\n", "utf-8")
    runtime = internal / "ai-runtime-descriptor.json"
    runtime.write_text(
        canonical(
            {
                "directRequirements": requirements,
                "python": "3.11.9",
                "requirementsAiLockSha256": sha256(lock),
                "requirementsAiSha256": "b" * 64,
                "schemaVersion": "vem-ai-runtime-descriptor/v1",
                "target": "windows-x86_64",
                "workerLayout": {
                    "mainOnedir": "vending-vision",
                    "modelPackEnv": "VEM_AI_MODEL_PACK",
                    "workerExecutable": "vending-vision-ai-worker/vending-vision-ai-worker.exe",
                    "workerOnedir": "vending-vision-ai-worker",
                },
            }
        )
        + "\n",
        "utf-8",
    )
    source = internal / "official-ai-source-descriptor.json"
    source.write_text(
        canonical(
            {
                "catvtonSourceRevision": SOURCE_REVISION,
                "schemaVersion": "vem-official-ai-source-descriptor/v1",
                "sources": [],
            }
        )
        + "\n",
        "utf-8",
    )
    model_source = root / "model-source"
    model_file = model_source / "inpainting/unet/config.json"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("{}", "utf-8")
    model_descriptor = {
        "catvtonSourceRevision": SOURCE_REVISION,
        "files": [
            {
                "byteSize": 2,
                "format": "json",
                "path": "inpainting/unet/config.json",
                "role": "inpainting_unet_config",
                "sha256": sha256(model_file),
                "upstream": "inpainting",
                "upstreamPath": "unet/config.json",
            }
        ],
        "schemaVersion": "vem-official-ai-model-pack-descriptor/v2",
        "totalByteSize": 2,
        "upstreams": [
            {"id": "inpainting", "repository": "example/mini", "revision": "c" * 40}
        ],
    }
    model_descriptor_path = internal / "official-ai-model-pack-descriptor.json"
    model_descriptor_path.write_text(
        canonical_ai_model_manifest_json(model_descriptor) + model_descriptor_suffix, "utf-8"
    )
    candidate_root = root / "candidate-input"
    candidate_root.mkdir()
    candidate_archive = candidate_root / "candidate.zip"
    candidate_manifest = candidate_root / "candidate-manifest.json"
    candidate_facts = write_candidate_archive(
        dist, candidate_archive, candidate_manifest, source_commit=SOURCE_COMMIT
    )
    attestation = candidate_root / "github-build-provenance.sigstore.json"
    attestation.write_text("{}\n", "utf-8")
    evidence = candidate_root / "trusted-builder-evidence.json"
    evidence.write_text(
        canonical(
            {
                "attestationBundleSha256": sha256(attestation),
                "builderRepository": "hbhjt/vending-vision",
                "builderWorkflow": ".github/workflows/trusted-ai-candidate-builder.yml",
                "builderWorkflowSha": TRUSTED_BUILDER_COMMIT,
                "embeddedManifestSha256": candidate_facts["embeddedManifestSha256"],
                "schemaVersion": "vending-vision-trusted-builder-evidence/v1",
                "sourceCommit": SOURCE_COMMIT,
                "subjectSha256": candidate_facts["subjectSha256"],
            }
        ),
        "utf-8",
    )
    model_archive = root / "model-pack.zip"
    model_sha = build_model_pack_zip(model_source, model_archive, model_descriptor)
    return {
        "attestation": attestation,
        "candidate_archive": candidate_archive,
        "candidate_manifest": candidate_manifest,
        "evidence": evidence,
        "manifest_sha": candidate_facts["embeddedManifestSha256"],
        "model_archive": model_archive,
        "model_descriptor_sha": hashlib.sha256(
            (canonical_ai_model_manifest_json(model_descriptor) + model_descriptor_suffix).encode(
                "utf-8"
            )
        ).hexdigest(),
        "model_sha": model_sha,
        "subject_sha": candidate_facts["subjectSha256"],
    }


def test_source_mode_verifies_real_candidate_model_archive_and_both_worker_probes(tmp_path):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"

    report = verify_precutover(
        candidate_artifact=fixture["candidate_archive"],
        candidate_manifest=fixture["candidate_manifest"],
        github_attestation=fixture["attestation"],
        trusted_builder_evidence=fixture["evidence"],
        subject_sha256=fixture["subject_sha"],
        manifest_sha256=fixture["manifest_sha"],
        attestation_bundle_sha256=sha256(fixture["attestation"]),
        source_commit=SOURCE_COMMIT,
        model_pack_archive=fixture["model_archive"],
        model_pack_byte_size=fixture["model_archive"].stat().st_size,
        model_pack_sha256=fixture["model_sha"],
        model_descriptor_sha256=fixture["model_descriptor_sha"],
        private_parent=tmp_path,
        report_output=output,
        timeout=5.0,
    )

    assert output.read_text("utf-8") == canonical(report) + "\n"
    assert report["schemaVersion"] == "vending-vision-precutover-proof/v1"
    assert report["probes"]["runtime"]["probe"] == "official-catvton-worker-runtime"
    assert report["probes"]["model"]["probe"] == "official-catvton-worker"
    assert report["candidate"]["workerMode"] == "source-test-only"
    manifest = json.loads(fixture["candidate_manifest"].read_text("utf-8"))
    assert (
        report["candidate"]["workerExecutableSha256"]
        == manifest["bindings"]["workerExecutable"]["sha256"]
    )
    assert list(tmp_path.glob(".precutover-*")) == []


def test_source_mode_accepts_canonical_model_descriptor_with_trailing_newline(tmp_path):
    fixture = build_fixture(tmp_path, model_descriptor_suffix="\n")
    output = tmp_path / "proof.json"

    report = verify_precutover(
        candidate_artifact=fixture["candidate_archive"],
        candidate_manifest=fixture["candidate_manifest"],
        github_attestation=fixture["attestation"],
        trusted_builder_evidence=fixture["evidence"],
        subject_sha256=fixture["subject_sha"],
        manifest_sha256=fixture["manifest_sha"],
        attestation_bundle_sha256=sha256(fixture["attestation"]),
        source_commit=SOURCE_COMMIT,
        model_pack_archive=fixture["model_archive"],
        model_pack_byte_size=fixture["model_archive"].stat().st_size,
        model_pack_sha256=fixture["model_sha"],
        model_descriptor_sha256=fixture["model_descriptor_sha"],
        private_parent=tmp_path,
        report_output=output,
        timeout=5.0,
    )

    assert report["schemaVersion"] == "vending-vision-precutover-proof/v1"
    assert output.is_file()


def test_source_test_worker_uses_the_interpreter_even_on_windows():
    worker = Path("C:/private/vending-vision-ai-worker.exe")

    assert _worker_probe_command(worker, require_frozen_worker=False) == [
        sys.executable,
        str(worker),
    ]
    assert _worker_probe_command(worker, require_frozen_worker=True) == [str(worker)]


def test_exact_json_emitter_is_not_accepted_as_a_frozen_windows_worker(tmp_path):
    emitter = tmp_path / "vending-vision-ai-worker.exe"
    emitter.write_text(
        "#!/usr/bin/env python3\nprint('{\"probe\":\"official-catvton-worker-runtime\"}')\n",
        "utf-8",
    )

    try:
        verify_frozen_worker_archive(emitter)
    except RuntimeError as exc:
        assert str(exc) == "precutover_worker_not_frozen"
    else:
        raise AssertionError("a JSON emitter must not be a packaged worker")


def test_production_cli_is_windows_only_and_has_no_source_mode_or_fake_worker_flag(tmp_path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    candidate_artifact = candidate_root / "candidate.zip"
    candidate_manifest = candidate_root / "candidate-manifest.json"
    attestation = candidate_root / "attestation.json"
    evidence = candidate_root / "evidence.json"
    for path, content in (
        (candidate_artifact, b"not a candidate archive"),
        (candidate_manifest, b"{}"),
        (attestation, b"{}"),
        (evidence, b"{}"),
    ):
        path.write_bytes(content)
    model_pack = tmp_path / "model.zip"
    model_pack.write_bytes(b"not a model archive")
    arguments = [
        "--candidate-artifact", str(candidate_artifact),
        "--candidate-manifest", str(candidate_manifest),
        "--github-attestation", str(attestation),
        "--trusted-builder-evidence", str(evidence),
        "--subject-sha256", sha256(candidate_artifact),
        "--manifest-sha256", sha256(candidate_manifest),
        "--attestation-bundle-sha256", sha256(attestation),
        "--source-commit", "d" * 40,
        "--model-pack-archive", str(model_pack),
        "--model-pack-byte-size", str(model_pack.stat().st_size),
        "--model-pack-sha256", sha256(model_pack),
        "--model-descriptor-sha256", "f" * 64,
        "--private-parent", str(tmp_path),
        "--report-output", str(tmp_path / "proof.json"),
    ]

    if os.name == "nt":
        with pytest.raises(SystemExit, match=r"PRECUTOVER_COMPANION=FAIL:(?!windows_required)"):
            companion_main(arguments)
    else:
        with pytest.raises(SystemExit, match="windows_required"):
            companion_main(arguments)
    with pytest.raises(SystemExit):
        companion_main([*arguments, "--fake-worker", "emitter.exe"])


def invoke_fixture(fixture: dict[str, object], tmp_path: Path, output: Path, **overrides):
    arguments = {
        "candidate_artifact": fixture["candidate_archive"],
        "candidate_manifest": fixture["candidate_manifest"],
        "github_attestation": fixture["attestation"],
        "trusted_builder_evidence": fixture["evidence"],
        "subject_sha256": fixture["subject_sha"],
        "manifest_sha256": fixture["manifest_sha"],
        "attestation_bundle_sha256": sha256(fixture["attestation"]),
        "source_commit": SOURCE_COMMIT,
        "model_pack_archive": fixture["model_archive"],
        "model_pack_byte_size": fixture["model_archive"].stat().st_size,
        "model_pack_sha256": fixture["model_sha"],
        "model_descriptor_sha256": fixture["model_descriptor_sha"],
        "private_parent": tmp_path,
        "report_output": output,
        "timeout": 5.0,
    }
    arguments.update(overrides)
    return verify_precutover(**arguments)


def test_worker_probe_rejects_a_wrong_pinned_dependency_version_without_partial_output(
    tmp_path,
):
    fixture = build_fixture(tmp_path, torch_version="9.9.0+cpu")
    output = tmp_path / "proof.json"

    with pytest.raises(RuntimeError, match="precutover_worker_dependency:torch"):
        invoke_fixture(fixture, tmp_path, output)

    assert not output.exists()
    assert list(tmp_path.glob(".precutover-*")) == []


def test_corrupt_model_archive_with_self_updated_outer_hash_still_fails_exact_members(
    tmp_path,
):
    fixture = build_fixture(tmp_path)
    source = fixture["model_archive"]
    corrupt = tmp_path / "model-pack-extra.zip"
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(
        corrupt, "w", compression=zipfile.ZIP_STORED
    ) as replacement:
        for info in original.infolist():
            replacement.writestr(info, original.read(info.filename))
        replacement.writestr("extra.bin", b"attacker")
    output = tmp_path / "proof.json"

    with pytest.raises(RuntimeError, match="ai_model_zip_entries"):
        invoke_fixture(
            fixture,
            tmp_path,
            output,
            model_pack_archive=corrupt,
            model_pack_byte_size=corrupt.stat().st_size,
            model_pack_sha256=sha256(corrupt),
        )

    assert not output.exists()


def test_candidate_input_directory_rejects_an_extra_fifth_member_before_extraction(tmp_path):
    fixture = build_fixture(tmp_path)
    fixture["candidate_archive"].parent.joinpath("attacker.txt").write_text("extra")
    output = tmp_path / "proof.json"

    with pytest.raises(RuntimeError, match="candidate_exact4_member_set"):
        invoke_fixture(fixture, tmp_path, output)

    assert not output.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group descendant assertion")
def test_worker_timeout_kills_its_descendant_and_leaves_no_partial_proof(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    script = (
        "#!/usr/bin/env python3\n"
        "import pathlib,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    fixture = build_fixture(tmp_path, worker_script=script)
    output = tmp_path / "proof.json"

    with pytest.raises(RuntimeError, match="supervised_process_timeout"):
        invoke_fixture(fixture, tmp_path, output, timeout=0.2)

    descendant = int(pid_file.read_text("utf-8"))
    deadline = time.monotonic() + 2
    while Path(f"/proc/{descendant}").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not Path(f"/proc/{descendant}").exists()
    assert not output.exists()
    assert list(tmp_path.glob(".precutover-*")) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX in-place worker mutation assertion")
def test_private_worker_rewrite_between_probes_cannot_publish_a_proof(tmp_path):
    script = (
        "#!/usr/bin/env python3\n"
        "import json,pathlib,sys\n"
        "path=pathlib.Path(__file__)\n"
        "if '--probe-runtime' in sys.argv:\n"
        " data=path.read_bytes()\n"
        " with path.open('r+b') as stream:\n"
        "  stream.seek(0); stream.write(data); stream.flush()\n"
        f"payload={{'catvtonSourceRevision':{SOURCE_REVISION!r},"
        "'torch':'2.8.0+cpu','torchvision':'0.23.0+cpu',"
        "'diffusers':'0.29.2','transformers':'4.53.3'}\n"
        "payload['probe']='official-catvton-worker-runtime' if '--probe-runtime' in sys.argv else 'official-catvton-worker'\n"
        "print(json.dumps(payload,sort_keys=True))\n"
    )
    fixture = build_fixture(tmp_path, worker_script=script)
    output = tmp_path / "proof.json"

    with pytest.raises(RuntimeError, match="precutover_integrity"):
        invoke_fixture(fixture, tmp_path, output)

    assert not output.exists()
    assert list(tmp_path.glob(".precutover-*")) == []


@pytest.mark.parametrize(
    "target",
    [
        "worker",
        "worker_resource",
        "runtime_descriptor",
        "ai_lock",
        "source_descriptor",
        "model_descriptor",
        "staged_model_archive",
        "installed_model",
        "source_candidate",
        "source_manifest",
        "source_attestation",
        "source_evidence",
        "source_model_archive",
    ],
)
@pytest.mark.parametrize("mutation", ["atomic-replace", "in-place-rewrite"])
@pytest.mark.parametrize(
    "phase", ["before-runtime-probe", "between-probes", "before-receipt"]
)
@pytest.mark.skipif(os.name == "nt", reason="POSIX mutation-observation matrix")
def test_every_critical_input_is_fenced_through_both_probes_and_receipt_publish(
    tmp_path, target, mutation, phase
):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"

    def mutate(observed_phase: str, paths: dict[str, Path]) -> None:
        if observed_phase != phase:
            return
        path = paths[target]
        content = path.read_bytes()
        if mutation == "atomic-replace":
            replacement = path.with_name(f".{path.name}.replacement")
            replacement.write_bytes(content)
            os.replace(replacement, path)
        else:
            with path.open("r+b") as stream:
                stream.seek(0)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

    with pytest.raises(RuntimeError, match="precutover_integrity"):
        invoke_fixture(fixture, tmp_path, output, _test_phase_hook=mutate)

    assert not output.exists()
    assert list(tmp_path.glob(".precutover-*")) == []


def test_windows_file_lease_allows_only_read_sharing_and_checks_close_result(tmp_path):
    class FakeKernel32:
        def __init__(self):
            self.opened = []
            self.closed = []
            self.close_result = 1

        def CreateFileW(self, *arguments):
            self.opened.append(arguments)
            return 42

        def CloseHandle(self, handle):
            self.closed.append(handle)
            return self.close_result

    kernel32 = FakeKernel32()
    api = _WindowsFileApi(kernel32)
    path = tmp_path / "worker.exe"
    handle = api.open_read_lease(path)

    assert handle == 42
    assert kernel32.opened == [
        (
            str(path),
            _WindowsFileApi.GENERIC_READ,
            _WindowsFileApi.FILE_SHARE_READ,
            None,
            _WindowsFileApi.OPEN_EXISTING,
            _WindowsFileApi.FILE_ATTRIBUTE_NORMAL,
            None,
        )
    ]
    api.close(handle)
    assert kernel32.closed == [42]

    kernel32.close_result = 0
    with pytest.raises(RuntimeError, match="precutover_integrity_handle_close"):
        api.close(handle)
    assert kernel32.closed == [42, 42, 42]


def test_windows_prepared_proof_is_created_and_flushed_with_read_only_sharing(tmp_path):
    class FakeKernel32:
        def __init__(self):
            self.opened = []
            self.written = bytearray()
            self.flushed = []

        def CreateFileW(self, *arguments):
            self.opened.append(arguments)
            return 84

        def WriteFile(self, handle, buffer, size, written, _overlapped):
            assert handle == 84
            self.written.extend(bytes(buffer.raw[:size]))
            written._obj.value = size
            return 1

        def FlushFileBuffers(self, handle):
            self.flushed.append(handle)
            return 1

        def CloseHandle(self, _handle):
            return 1

    kernel32 = FakeKernel32()
    api = _WindowsFileApi(kernel32)
    path = tmp_path / ".proof.random.tmp"
    handle = api.create_prepared_lease(path)
    api.write_all(handle, b"canonical proof\n")

    assert kernel32.opened == [
        (
            str(path),
            _WindowsFileApi.GENERIC_READ | _WindowsFileApi.GENERIC_WRITE,
            _WindowsFileApi.FILE_SHARE_READ,
            None,
            _WindowsFileApi.CREATE_NEW,
            _WindowsFileApi.FILE_ATTRIBUTE_NORMAL,
            None,
        )
    ]
    assert kernel32.written == b"canonical proof\n"
    assert kernel32.flushed == [84]


@pytest.mark.parametrize("invalid_handle", [None, 0, -1])
def test_windows_file_lease_open_failure_is_fail_closed(tmp_path, invalid_handle):
    class FakeKernel32:
        def CreateFileW(self, *_arguments):
            return invalid_handle

        def CloseHandle(self, _handle):
            raise AssertionError("an invalid handle must not be closed")

    with pytest.raises(RuntimeError, match="precutover_integrity_handle_open"):
        _WindowsFileApi(FakeKernel32()).open_read_lease(tmp_path / "worker.exe")


def test_integrity_fence_rejects_unbounded_file_sets_before_opening_paths(tmp_path):
    oversized = [
        _ExpectedFile(tmp_path / str(index), 0, "a" * 64, str(index))
        for index in range(20_001)
    ]

    with pytest.raises(RuntimeError, match="precutover_integrity_bounds"):
        _IntegrityFence(oversized)


def test_held_descriptor_hash_has_a_windows_compatible_fallback(tmp_path, monkeypatch):
    path = tmp_path / "resource.dll"
    path.write_bytes(b"held resource")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        monkeypatch.delattr(os, "pread", raising=False)
        assert _fd_sha256(descriptor) == sha256(path)
    finally:
        os.close(descriptor)


def test_handle_close_failure_after_proof_preparation_leaves_no_final_proof(
    tmp_path, monkeypatch
):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"
    original_close = _IntegrityFence.close
    failed = False

    def fail_one_close(fence):
        nonlocal failed
        original_close(fence)
        if not failed:
            failed = True
            raise RuntimeError("precutover_integrity_handle_close")

    monkeypatch.setattr(companion._IntegrityFence, "close", fail_one_close)

    with pytest.raises(RuntimeError, match="precutover_integrity_handle_close"):
        invoke_fixture(fixture, tmp_path, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".proof.json.*.tmp"))
    assert not list(tmp_path.glob(".precutover-*"))


def test_preexisting_proof_is_preserved_by_exclusive_publish(tmp_path):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"
    output.write_bytes(b"operator-owned proof\n")

    with pytest.raises(RuntimeError, match="precutover_report_path"):
        invoke_fixture(fixture, tmp_path, output)

    assert output.read_bytes() == b"operator-owned proof\n"
    assert not list(tmp_path.glob(".proof.json.*.tmp"))
    assert not list(tmp_path.glob(".precutover-*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX linkat publication seam")
def test_final_proof_link_failure_cleans_prepared_and_private_state(tmp_path, monkeypatch):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"

    def fail_link(_api, _source_descriptor, _parent_descriptor, _name):
        raise OSError("injected proof link failure")

    monkeypatch.setattr(companion._PosixLinkApi, "link_fd", fail_link)

    with pytest.raises(OSError, match="injected proof link failure"):
        invoke_fixture(fixture, tmp_path, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".proof.json.*.tmp"))
    assert not list(tmp_path.glob(".precutover-*"))


def test_private_cleanup_failure_cannot_publish_and_finally_retries_cleanup(
    tmp_path, monkeypatch
):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"
    original_rmtree = companion.shutil.rmtree
    failed = False

    def fail_first_private_cleanup(path, *args, **kwargs):
        nonlocal failed
        if Path(path).name.startswith(".precutover-") and not failed:
            failed = True
            raise OSError("injected private cleanup failure")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(companion.shutil, "rmtree", fail_first_private_cleanup)

    with pytest.raises(OSError, match="injected private cleanup failure"):
        invoke_fixture(fixture, tmp_path, output)

    assert failed
    assert not output.exists()
    assert not list(tmp_path.glob(".proof.json.*.tmp"))
    assert not list(tmp_path.glob(".precutover-*"))


def test_failure_after_final_link_rolls_back_only_this_invocation(tmp_path, monkeypatch):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"
    original_fsync_parent = companion._fsync_parent
    calls = 0

    def fail_first_fsync(path, *, descriptor=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected parent fsync failure")
        return original_fsync_parent(path, descriptor=descriptor)

    monkeypatch.setattr(companion, "_fsync_parent", fail_first_fsync)

    with pytest.raises(OSError, match="injected parent fsync failure"):
        invoke_fixture(fixture, tmp_path, output)

    assert calls == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".proof.json.*.tmp"))
    assert not list(tmp_path.glob(".precutover-*"))


@pytest.mark.parametrize("mutation", ["atomic-replace", "in-place-same-bytes"])
@pytest.mark.skipif(os.name == "nt", reason="POSIX prepared-proof mutation assertion")
def test_prepared_proof_is_fenced_while_inputs_close_and_private_state_cleans(
    tmp_path, monkeypatch, mutation
):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"
    original_prepare = companion._prepare_exclusive

    def mutate_after_prepare(path, value, *, before_close):
        prepared = original_prepare(path, value, before_close=before_close)
        content = prepared.path.read_bytes()
        if mutation == "atomic-replace":
            replacement = prepared.path.with_name(f".{prepared.path.name}.replacement")
            replacement.write_bytes(b"attacker-controlled proof\n")
            os.replace(replacement, prepared.path)
        else:
            with prepared.path.open("r+b") as stream:
                stream.seek(0)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        return prepared

    monkeypatch.setattr(companion, "_prepare_exclusive", mutate_after_prepare)

    with pytest.raises(RuntimeError, match="precutover_integrity"):
        invoke_fixture(fixture, tmp_path, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".proof.json.*.tmp"))
    assert not list(tmp_path.glob(".precutover-*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX linkat publication seam")
def test_prepared_proof_link_rejects_an_instant_source_identity_swap(tmp_path, monkeypatch):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"
    original_link = companion._PosixLinkApi.link_fd

    def swap_source_after_link(api, source_descriptor, parent_descriptor, name):
        original_link(api, source_descriptor, parent_descriptor, name)
        source = Path(os.readlink(f"/proc/self/fd/{source_descriptor}"))
        replacement = source.with_name(f".{source.name}.replacement")
        replacement.write_bytes(b"swapped after link\n")
        os.replace(replacement, source)

    monkeypatch.setattr(companion._PosixLinkApi, "link_fd", swap_source_after_link)

    with pytest.raises(RuntimeError, match="precutover_integrity"):
        invoke_fixture(fixture, tmp_path, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".proof.json.*.tmp"))
    assert not list(tmp_path.glob(".precutover-*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX prepared-proof lease assertion")
def test_prepared_proof_lease_close_failure_rolls_back_linked_final(tmp_path, monkeypatch):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"
    original_close = _IntegrityFence.close
    failed = False

    def fail_prepared_close(fence):
        nonlocal failed
        original_close(fence)
        if (
            not failed
            and fence._expected
            and fence._expected[0].label == "prepared_proof"
        ):
            failed = True
            raise RuntimeError("precutover_integrity_handle_close")

    monkeypatch.setattr(companion._IntegrityFence, "close", fail_prepared_close)

    with pytest.raises(RuntimeError, match="precutover_integrity_handle_close"):
        invoke_fixture(fixture, tmp_path, output)

    assert failed
    assert not output.exists()
    assert not list(tmp_path.glob(".proof.json.*.tmp"))
    assert not list(tmp_path.glob(".precutover-*"))


def test_windows_prepared_closehandle_false_rolls_back_the_linked_final(tmp_path):
    temporary = tmp_path / ".proof.windows.tmp"
    final = tmp_path / "proof.json"
    temporary.write_bytes(b"canonical proof\n")
    expectation = _ExpectedFile(
        temporary,
        temporary.stat().st_size,
        sha256(temporary),
        "prepared_proof",
    )

    class Kernel32:
        def __init__(self):
            self.close_results = [0, 1]

        def CloseHandle(self, handle):
            assert handle == 84
            return self.close_results.pop(0)

    prepared = companion._PreparedProof(
        temporary,
        expectation,
        windows_api=_WindowsFileApi(Kernel32()),
        windows_handle=84,
    )

    with pytest.raises(RuntimeError, match="precutover_integrity_handle_close"):
        companion._publish_prepared(prepared, final)

    assert not final.exists()
    assert not temporary.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX linkat publication seam")
def test_posix_prepared_temp_replace_after_verify_before_link_leaves_no_final(
    tmp_path, monkeypatch
):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"
    original_verify = companion._PreparedProof.verify
    verify_calls = 0

    def replace_after_publish_verify(prepared):
        nonlocal verify_calls
        original_verify(prepared)
        verify_calls += 1
        if verify_calls == 2:
            replacement = prepared.path.with_name(f".{prepared.path.name}.replacement")
            replacement.write_bytes(b"replaced after verify before link\n")
            os.replace(replacement, prepared.path)

    monkeypatch.setattr(companion._PreparedProof, "verify", replace_after_publish_verify)

    with pytest.raises(RuntimeError, match="precutover_integrity"):
        invoke_fixture(fixture, tmp_path, output)

    assert verify_calls >= 2
    assert not output.exists()
    assert not list(tmp_path.glob(".proof.json.*.tmp"))
    assert not list(tmp_path.glob(".precutover-*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX linkat publication seam")
def test_posix_linkat_uses_held_source_fd_parent_dirfd_and_empty_path_flag(tmp_path):
    class FakeLibc:
        def __init__(self):
            self.calls = []

        def linkat(self, *arguments):
            self.calls.append(arguments)
            return 0

    libc = FakeLibc()
    api = companion._PosixLinkApi(libc)
    source = tmp_path / "prepared.tmp"
    source.write_bytes(b"proof")
    source_descriptor = os.open(source, os.O_RDONLY)
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        api.link_fd(source_descriptor, parent_descriptor, "proof.json")
        assert libc.calls == [
            (
                source_descriptor,
                b"",
                parent_descriptor,
                b"proof.json",
                companion._PosixLinkApi.AT_EMPTY_PATH,
            )
        ]
    finally:
        os.close(parent_descriptor)
        os.close(source_descriptor)


@pytest.mark.skipif(os.name == "nt", reason="POSIX parent-directory lease assertion")
def test_parent_directory_close_after_effect_rolls_back_owned_final(tmp_path, monkeypatch):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"
    original_open_parent = companion._PosixLinkApi.open_parent
    original_close = companion.os.close
    parent_descriptor = None
    close_calls = 0

    def capture_parent(api, path):
        nonlocal parent_descriptor
        parent_descriptor, identity = original_open_parent(api, path)
        return parent_descriptor, identity

    def close_then_raise(descriptor):
        nonlocal close_calls
        if descriptor == parent_descriptor:
            close_calls += 1
            if close_calls == 1:
                original_close(descriptor)
                raise OSError("injected parent close failure after effect")
        return original_close(descriptor)

    monkeypatch.setattr(companion._PosixLinkApi, "open_parent", capture_parent)
    monkeypatch.setattr(companion.os, "close", close_then_raise)

    with pytest.raises(OSError, match="injected parent close failure after effect"):
        invoke_fixture(fixture, tmp_path, output)

    # One close for the original lease and one for the independently reopened
    # rollback lease; neither descriptor generation is closed twice.
    assert close_calls == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".proof.json.*.tmp"))
    assert not list(tmp_path.glob(".precutover-*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX parent-directory lease assertion")
def test_rollback_parent_close_after_effect_keeps_owned_final_removed(tmp_path, monkeypatch):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"
    original_open_parent = companion._PosixLinkApi.open_parent
    original_close = companion.os.close
    parent_descriptors = set()
    close_calls = 0

    def capture_parent(api, path):
        descriptor, identity = original_open_parent(api, path)
        parent_descriptors.add(descriptor)
        return descriptor, identity

    def close_then_raise(descriptor):
        nonlocal close_calls
        if descriptor in parent_descriptors:
            close_calls += 1
            original_close(descriptor)
            raise OSError(f"injected parent close failure {close_calls} after effect")
        return original_close(descriptor)

    monkeypatch.setattr(companion._PosixLinkApi, "open_parent", capture_parent)
    monkeypatch.setattr(companion.os, "close", close_then_raise)

    with pytest.raises(OSError, match="injected parent close failure 1 after effect"):
        invoke_fixture(fixture, tmp_path, output)

    assert close_calls == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".proof.json.*.tmp"))
    assert not list(tmp_path.glob(".precutover-*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX parent-directory lease assertion")
def test_rollback_parent_reopen_identity_mismatch_never_unlinks_other_directory(
    tmp_path, monkeypatch
):
    fixture = build_fixture(tmp_path)
    output = tmp_path / "proof.json"
    other_parent = tmp_path / "other-parent"
    other_parent.mkdir()
    other_proof = other_parent / output.name
    other_proof.write_bytes(b"operator-owned proof\n")
    original_open_parent = companion._PosixLinkApi.open_parent
    original_close = companion.os.close
    open_calls = 0
    first_parent_descriptor = None
    close_failed = False

    def redirect_rollback_open(api, path):
        nonlocal open_calls, first_parent_descriptor
        open_calls += 1
        opened = original_open_parent(api, path if open_calls == 1 else other_parent)
        if open_calls == 1:
            first_parent_descriptor = opened[0]
        return opened

    def close_first_parent_then_raise(descriptor):
        nonlocal close_failed
        if descriptor == first_parent_descriptor and not close_failed:
            close_failed = True
            original_close(descriptor)
            raise OSError("injected original parent close failure after effect")
        return original_close(descriptor)

    monkeypatch.setattr(companion._PosixLinkApi, "open_parent", redirect_rollback_open)
    monkeypatch.setattr(companion.os, "close", close_first_parent_then_raise)

    with pytest.raises(OSError, match="original parent close failure after effect"):
        invoke_fixture(fixture, tmp_path, output)

    assert open_calls == 2
    assert close_failed
    assert output.exists()
    assert other_proof.read_bytes() == b"operator-owned proof\n"
    assert not list(tmp_path.glob(".proof.json.*.tmp"))
    assert not list(tmp_path.glob(".precutover-*"))
