import asyncio
import hashlib
import json

from vision.ai_attempt_process import (
    AiAttemptProcess,
    ai_attempt_worker_command,
    windows_ai_child_creation_flags,
)


def test_official_child_probe_joins_before_return(tmp_path):
    file = tmp_path / "weights" / "a.bin"
    file.parent.mkdir()
    file.write_bytes(b"w")
    (tmp_path / "ai-model-manifest.json").write_text(json.dumps({"schemaVersion": "vem-catvton-model-pack/v1", "upstream": {"repository": "zhengchong/CatVTON", "revision": "9f415fa"}, "files": [{"path": "weights/a.bin", "byteSize": 1, "sha256": hashlib.sha256(b"w").hexdigest()}]}))
    child = AiAttemptProcess(tmp_path)
    asyncio.run(child.probe())
    assert child._process is None


def test_windows_production_ai_child_uses_group_and_low_priority_boundary():
    class WindowsSubprocess:
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

    flags = windows_ai_child_creation_flags(WindowsSubprocess)

    assert flags & WindowsSubprocess.CREATE_NEW_PROCESS_GROUP
    assert flags & WindowsSubprocess.BELOW_NORMAL_PRIORITY_CLASS


def test_frozen_official_ai_child_uses_packaged_worker_entrypoint(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys.executable", "/opt/vending-vision/vending-vision")

    assert ai_attempt_worker_command(tmp_path, probe=True) == [
        "/opt/vending-vision/vending-vision",
        "--ai-attempt-worker",
        "--model-pack",
        str(tmp_path),
        "--probe",
    ]
