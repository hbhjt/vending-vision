from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import pytest

from scripts.ai_model_pack_release import build_model_pack_zip
from scripts.candidate_artifact_manifest import write_candidate_archive
from vision.ai_model_pack import canonical_ai_model_manifest_json
from vision.precutover_companion import verify_frozen_worker_archive, verify_precutover
from vision.precutover_companion import main as companion_main


SOURCE_COMMIT = "a" * 40
SOURCE_REVISION = "3b795364a4d2f3b5adb365f39cdea376d20bc53c"
TRUSTED_BUILDER_COMMIT = "be8fe434855b94f61511e8c6c926e02c54230a38"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_fixture(
    root: Path, *, torch_version: str = "2.8.0+cpu", worker_script: str | None = None
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
        canonical_ai_model_manifest_json(model_descriptor), "utf-8"
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
            canonical_ai_model_manifest_json(model_descriptor).encode("utf-8")
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
    assert list(tmp_path.glob(".precutover-*")) == []


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


def test_production_cli_is_windows_only_and_has_no_source_mode_or_fake_worker_flag():
    arguments = [
        "--candidate-artifact", "candidate.zip",
        "--candidate-manifest", "candidate-manifest.json",
        "--github-attestation", "attestation.json",
        "--trusted-builder-evidence", "evidence.json",
        "--subject-sha256", "a" * 64,
        "--manifest-sha256", "b" * 64,
        "--attestation-bundle-sha256", "c" * 64,
        "--source-commit", "d" * 40,
        "--model-pack-archive", "model.zip",
        "--model-pack-byte-size", "1",
        "--model-pack-sha256", "e" * 64,
        "--model-descriptor-sha256", "f" * 64,
        "--private-parent", ".",
        "--report-output", "proof.json",
    ]

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
