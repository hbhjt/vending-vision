import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from vision.ai_attempt_process import (
    AiAttemptProcess,
    ai_attempt_worker_command,
    probe_ai_attempt_worker,
    probe_ai_attempt_worker_async,
    windows_ai_child_creation_flags,
)
import vision.ai_attempt_process as ai_attempt_process_module


def test_nonofficial_child_probe_fails_closed_and_joins_before_return(tmp_path):
    file = tmp_path / "weights" / "a.bin"
    file.parent.mkdir()
    file.write_bytes(b"w")
    (tmp_path / "ai-model-manifest.json").write_text(json.dumps({"schemaVersion": "vem-catvton-model-pack/v1", "upstream": {"repository": "zhengchong/CatVTON", "revision": "9f415fa"}, "files": [{"path": "weights/a.bin", "byteSize": 1, "sha256": hashlib.sha256(b"w").hexdigest()}]}))
    child = AiAttemptProcess(tmp_path)
    try:
        asyncio.run(child.probe())
    except RuntimeError as exc:
        assert str(exc) == "official_ai_child_failed"
    else:
        raise AssertionError("nonofficial probe pack must fail closed")
    assert child._running is False


def test_windows_production_ai_child_uses_group_and_low_priority_boundary():
    class WindowsSubprocess:
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

    flags = windows_ai_child_creation_flags(WindowsSubprocess)

    assert flags & WindowsSubprocess.CREATE_NEW_PROCESS_GROUP
    assert flags & WindowsSubprocess.BELOW_NORMAL_PRIORITY_CLASS


def test_frozen_official_ai_child_uses_packaged_worker_entrypoint(monkeypatch, tmp_path):
    runtime = tmp_path / "vending-vision"
    runtime.write_text("main", "utf-8")
    worker = tmp_path / "vending-vision-ai-worker"
    worker.write_text("worker", "utf-8")
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys.executable", str(runtime))

    assert ai_attempt_worker_command(tmp_path, probe=True) == [
        str(worker.resolve()),
        "--model-pack",
        str(tmp_path),
        "--probe",
    ]


def test_official_ai_child_attempt_command_carries_template_without_fake_selector(tmp_path):
    command = ai_attempt_worker_command(
        tmp_path / "pack",
        person_png=tmp_path / "person.png",
        garment_png=tmp_path / "garment.png",
        output_png=tmp_path / "output.png",
        template="tshirt_long_sleeve",
    )

    assert command[-2:] == ["--template", "tshirt_long_sleeve"]
    assert "--fake-worker" not in command
    assert "--config" not in command


def test_startup_probe_rejects_worker_dependency_version_mismatch(monkeypatch, tmp_path):
    async def fake_run_supervised(_command, *, timeout):
        return SimpleNamespace(
            returncode=0,
            stdout_tail=b'{"probe":"official-catvton-worker","torch":"0.0.0","torchvision":"0.23.0","diffusers":"0.29.2","transformers":"4.53.3"}\n',
        )

    monkeypatch.setattr(ai_attempt_process_module, "run_supervised", fake_run_supervised)

    try:
        probe_ai_attempt_worker(tmp_path)
    except RuntimeError as exc:
        assert str(exc) == "official_ai_child_probe_failed"
    else:
        raise AssertionError("dependency mismatch must fail closed")


def test_runtime_probe_uses_pep440_specifier_for_cpu_local_version(monkeypatch):
    async def fake_run_supervised(_command, *, timeout):
        return SimpleNamespace(
            returncode=0,
            stdout_tail=b'{"probe":"official-catvton-worker-runtime","torch":"2.8.0+cpu"}\n',
        )

    monkeypatch.setattr(ai_attempt_process_module, "run_supervised", fake_run_supervised)
    monkeypatch.setattr(
        ai_attempt_process_module,
        "expected_dependency_requirements",
        lambda: {"torch": "torch==2.8.0+cpu"},
        raising=False,
    )

    asyncio.run(ai_attempt_process_module.probe_ai_runtime_worker_async())

    async def wrong_run_supervised(_command, *, timeout):
        return SimpleNamespace(
            returncode=0,
            stdout_tail=b'{"probe":"official-catvton-worker-runtime","torch":"9.9.0+cpu"}\n',
        )

    monkeypatch.setattr(ai_attempt_process_module, "run_supervised", wrong_run_supervised)
    with pytest.raises(RuntimeError, match="official_ai_child_runtime_probe_failed"):
        asyncio.run(ai_attempt_process_module.probe_ai_runtime_worker_async())


def test_startup_probe_has_async_api_and_sync_wrapper_rejects_running_loop(monkeypatch, tmp_path):
    async def fake_run_supervised(_command, *, timeout):
        return SimpleNamespace(
            returncode=0,
            stdout_tail=b'{"probe":"official-catvton-worker","torch":"2.8.0+cpu","torchvision":"0.23.0+cpu","diffusers":"0.29.2","transformers":"4.53.3"}\n',
        )

    monkeypatch.setattr(ai_attempt_process_module, "run_supervised", fake_run_supervised)

    async def exercise():
        await probe_ai_attempt_worker_async(tmp_path)
        try:
            probe_ai_attempt_worker(tmp_path)
        except RuntimeError as exc:
            assert str(exc) == "official_ai_child_probe_requires_sync_context"
        else:
            raise AssertionError("sync probe must fail in a running event loop")

    asyncio.run(exercise())


def test_startup_probe_requires_official_probe_marker(monkeypatch, tmp_path):
    async def fake_run_supervised(_command, *, timeout):
        return SimpleNamespace(
            returncode=0,
            stdout_tail=b'{"probe":"runtime-boundary","torch":"2.8.0","torchvision":"0.23.0","diffusers":"0.29.2","transformers":"4.53.3"}\n',
        )

    monkeypatch.setattr(ai_attempt_process_module, "run_supervised", fake_run_supervised)

    try:
        probe_ai_attempt_worker(tmp_path)
    except RuntimeError as exc:
        assert str(exc) == "official_ai_child_probe_failed"
    else:
        raise AssertionError("wrong probe marker must fail closed")
