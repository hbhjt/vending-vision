import asyncio
import hashlib
import json

from vision.ai_attempt_process import AiAttemptProcess


def test_official_child_probe_joins_before_return(tmp_path):
    file = tmp_path / "weights" / "a.bin"
    file.parent.mkdir()
    file.write_bytes(b"w")
    (tmp_path / "ai-model-manifest.json").write_text(json.dumps({"schemaVersion": "vem-catvton-model-pack/v1", "upstream": {"repository": "zhengchong/CatVTON", "revision": "fixed"}, "files": [{"path": "weights/a.bin", "byteSize": 1, "sha256": hashlib.sha256(b"w").hexdigest()}]}))
    child = AiAttemptProcess(tmp_path)
    asyncio.run(child.probe())
    assert child._process is None
