import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_ai_model_pack import write_pack
from vision import ai_attempt_worker

ROOT = Path(__file__).parents[1]


def test_worker_probe_accepts_only_official_manifest_and_no_fake_arguments(tmp_path):
    write_pack(tmp_path)

    probe = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision.ai_attempt_worker",
            "--model-pack",
            str(tmp_path),
            "--probe",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert probe.returncode == 0, f"{probe.stdout}{probe.stderr}"
    assert "official-catvton-worker-configured" in probe.stdout

    fake = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision.ai_attempt_worker",
            "--model-pack",
            str(tmp_path),
            "--probe",
            "--fake-worker",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert fake.returncode != 0
    assert "--fake-worker" in fake.stderr


def test_worker_source_hard_guards_downloads_and_probe_does_not_load_or_infer():
    source = inspect.getsource(ai_attempt_worker)

    assert "HF_HUB_OFFLINE" in source
    assert "TRANSFORMERS_OFFLINE" in source
    assert "HF_DATASETS_OFFLINE" in source
    assert "socket.socket = _blocked_socket" in source
    assert "snapshot_download" not in source
    assert "from_pretrained" not in source
    assert "huggingface_hub" not in source
    assert "torch" not in source
    assert "diffusers" not in source
    assert "fake" not in source.lower()


def test_network_guard_blocks_customer_attempt_network_calls():
    original_socket = ai_attempt_worker.socket.socket
    try:
        ai_attempt_worker._deny_downloads()
        with pytest.raises(RuntimeError, match="customer_ai_attempt_network_forbidden"):
            ai_attempt_worker.socket.socket()
        assert ai_attempt_worker.os.environ["HF_HUB_OFFLINE"] == "1"
        assert ai_attempt_worker.os.environ["TRANSFORMERS_OFFLINE"] == "1"
    finally:
        ai_attempt_worker.socket.socket = original_socket
