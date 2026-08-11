import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from scripts.materialize_ai_wheelhouse import MaterializeError, materialize_ai_wheelhouse


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, final_url: str):
        super().__init__(payload)
        self._final_url = final_url

    def geturl(self):
        return self._final_url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _descriptor(path, payload=b"wheel"):
    file_name = "demo-1.0.0-py3-none-any.whl"
    url = f"https://files.pythonhosted.org/packages/test/{file_name}"
    value = {
        "schemaVersion": "vem-ai-worker-wheelhouse-release/v1",
        "target": "windows-x86_64",
        "python": "3.11.9",
        "source": "pip-report-locked-wheelhouse",
        "directRequirements": ["demo==1.0.0"],
        "wheels": [
            {
                "fileName": file_name,
                "name": "demo",
                "version": "1.0.0",
                "tags": "py3-none-any",
                "byteSize": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "direct": True,
                "transitive": False,
                "url": url,
                "source": "pypi",
            }
        ],
    }
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), "utf-8")
    return value, payload, url


def test_materializer_downloads_exact_locked_bytes_into_new_directory(tmp_path):
    descriptor, payload, url = _descriptor(tmp_path / "lock.json")
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return _Response(payload, url)

    destination = tmp_path / "wheelhouse"
    materialize_ai_wheelhouse(tmp_path / "lock.json", destination, opener=opener)

    assert calls == [(url, 120.0)]
    assert (destination / descriptor["wheels"][0]["fileName"]).read_bytes() == payload


def test_materializer_rejects_redirect_identity_and_existing_extra(tmp_path):
    _descriptor(tmp_path / "lock.json")

    def redirected(_request, timeout):
        return _Response(b"wheel", "https://files.pythonhosted.org/packages/test/other.whl")

    with pytest.raises(MaterializeError, match="ai_wheelhouse_redirect_identity"):
        materialize_ai_wheelhouse(tmp_path / "lock.json", tmp_path / "wheelhouse", opener=redirected)
    assert not (tmp_path / "wheelhouse").exists()

    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "extra.txt").write_text("extra", "utf-8")
    with pytest.raises(MaterializeError, match="ai_wheelhouse_destination_exists"):
        materialize_ai_wheelhouse(tmp_path / "lock.json", destination, opener=redirected)


def test_materializer_runs_in_clean_stdlib_only_python(tmp_path):
    descriptor, payload, url = _descriptor(tmp_path / "lock.json")
    requirements_path = tmp_path / "requirements-ai.txt"
    requirements_path.write_text("demo==1.0.0\n", "utf-8")
    runtime = {
        "schemaVersion": "vem-ai-runtime-descriptor/v1",
        "target": "windows-x86_64",
        "python": "3.11.9",
        "directRequirements": ["demo==1.0.0"],
        "requirementsAiSha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
        "requirementsAiLockSha256": hashlib.sha256((tmp_path / "lock.json").read_bytes()).hexdigest(),
        "workerLayout": {},
    }
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime, sort_keys=True, separators=(",", ":")), "utf-8")
    clean_venv = tmp_path / "clean-python"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(clean_venv)],
        check=True,
    )
    clean_python = clean_venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    destination = tmp_path / "downloaded"
    harness = textwrap.dedent(
        """
        import importlib.util
        import io
        import pathlib
        import sys

        module_path, descriptor_path, runtime_path, destination, url, payload_hex = sys.argv[1:]
        spec = importlib.util.spec_from_file_location("clean_materializer", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Response(io.BytesIO):
            def geturl(self):
                return url
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                self.close()

        module.materialize_ai_wheelhouse(
            pathlib.Path(descriptor_path),
            pathlib.Path(destination),
            opener=lambda _request, timeout: Response(bytes.fromhex(payload_hex)),
            runtime_descriptor_path=pathlib.Path(runtime_path),
        )
        """
    )

    completed = subprocess.run(
        [
            str(clean_python),
            "-c",
            harness,
            str(Path(__file__).parents[1] / "scripts" / "materialize_ai_wheelhouse.py"),
            str(tmp_path / "lock.json"),
            str(runtime_path),
            str(destination),
            url,
            payload.hex(),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (destination / descriptor["wheels"][0]["fileName"]).read_bytes() == payload


def test_stdlib_bootstrap_verifies_existing_ai_wheels_and_writes_hashed_requirements(tmp_path):
    from scripts.materialize_ai_wheelhouse import prepare_existing_ai_wheelhouse

    lock_path = tmp_path / "requirements-ai.lock.json"
    descriptor, payload, _url = _descriptor(lock_path)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = descriptor["wheels"][0]
    (wheelhouse / wheel["fileName"]).write_bytes(payload)
    requirements_path = tmp_path / "requirements-ai.txt"
    requirements_path.write_text("demo==1.0.0\n", "utf-8")
    runtime = {
        "schemaVersion": "vem-ai-runtime-descriptor/v1",
        "target": "windows-x86_64",
        "python": "3.11.9",
        "directRequirements": ["demo==1.0.0"],
        "requirementsAiSha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
        "requirementsAiLockSha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "workerLayout": {},
    }
    runtime_path = tmp_path / "ai-runtime-descriptor.json"
    runtime_path.write_text(json.dumps(runtime, sort_keys=True, separators=(",", ":")), "utf-8")
    output = tmp_path / "requirements-ai-release.txt"

    prepare_existing_ai_wheelhouse(lock_path, wheelhouse, runtime_path, output)

    assert output.read_text("utf-8") == (
        f"demo==1.0.0 --hash=sha256:{wheel['sha256']}\n"
    )


def test_existing_large_wheel_digest_is_streamed_without_read_bytes(tmp_path, monkeypatch):
    from scripts.materialize_ai_wheelhouse import prepare_existing_ai_wheelhouse

    lock_path = tmp_path / "requirements-ai.lock.json"
    descriptor, payload, _url = _descriptor(lock_path, payload=b"stream-this-wheel")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / descriptor["wheels"][0]["fileName"]
    wheel.write_bytes(payload)
    requirements_path = tmp_path / "requirements-ai.txt"
    requirements_path.write_text("demo==1.0.0\n", "utf-8")
    runtime = {
        "schemaVersion": "vem-ai-runtime-descriptor/v1",
        "target": "windows-x86_64",
        "python": "3.11.9",
        "directRequirements": ["demo==1.0.0"],
        "requirementsAiSha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
        "requirementsAiLockSha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "workerLayout": {},
    }
    runtime_path = tmp_path / "ai-runtime-descriptor.json"
    runtime_path.write_text(json.dumps(runtime, sort_keys=True, separators=(",", ":")), "utf-8")
    original = Path.read_bytes

    def guarded_read_bytes(path):
        if path.suffix == ".whl":
            raise AssertionError("wheel verification must stream")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    prepare_existing_ai_wheelhouse(
        lock_path, wheelhouse, runtime_path, tmp_path / "requirements-ai-release.txt"
    )


def test_materializer_aborts_expected_plus_one_before_writing_chunk(tmp_path, monkeypatch):
    _descriptor(tmp_path / "lock.json", payload=b"wheel")
    writes = []
    real_open = Path.open

    class Sink(io.BytesIO):
        def write(self, value):
            writes.append(len(value))
            return super().write(value)

    def guarded_open(path, mode="r", *args, **kwargs):
        if path.suffix == ".whl" and mode == "xb":
            return Sink()
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(MaterializeError, match="ai_wheelhouse_download_size"):
        materialize_ai_wheelhouse(
            tmp_path / "lock.json",
            tmp_path / "wheelhouse",
            opener=lambda request, timeout: _Response(b"wheel+", request.full_url),
        )

    assert writes == []
    assert not (tmp_path / "wheelhouse").exists()
