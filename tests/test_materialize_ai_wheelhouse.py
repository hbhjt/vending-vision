import hashlib
import io
import json

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
