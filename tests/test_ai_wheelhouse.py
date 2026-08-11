import json
import hashlib
import email.message
from pathlib import Path
import zipfile

import pytest

from vision.ai_runtime_descriptor import AiRuntimeDescriptorError, load_ai_runtime_descriptor

from scripts.verify_ai_wheelhouse import (
    WheelhouseError,
    build_ai_wheelhouse_descriptor,
    canonical_json,
    generate_hashed_requirements,
    verify_ai_wheelhouse,
)


def _write_wheel(path, *, requires=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    name, version, *_tags = path.name[:-4].split("-")
    metadata = email.message.Message()
    metadata["Metadata-Version"] = "2.1"
    metadata["Name"] = name.replace("_", "-")
    metadata["Version"] = version
    for requirement in requires:
        metadata["Requires-Dist"] = requirement
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(f"{name}-{version}.dist-info/METADATA", metadata.as_string())
        archive.writestr(f"{name}-{version}.dist-info/WHEEL", "Wheel-Version: 1.0\n")
    return path


def _wheel_entry(path, *, direct, transitive):
    parsed = path.name[:-4].split("-")
    return {
        "fileName": path.name,
        "name": parsed[0].replace("_", "-").lower(),
        "version": parsed[1],
        "tags": "-".join(parsed[2:]),
        "byteSize": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "direct": direct,
        "transitive": transitive,
        "url": f"https://files.pythonhosted.org/packages/test/{path.name}",
        "source": "pypi",
    }


def _runtime_descriptor(path, direct_requirements):
    requirements_path = path.parent / "requirements-ai.txt"
    lock_path = path.parent / "requirements-ai.lock.json"
    requirements_path.write_text("\n".join(direct_requirements) + "\n", "utf-8")
    lock_path.write_text(
        canonical_json(
            {
                "schemaVersion": "vem-ai-worker-wheelhouse-release/v1",
                "target": "windows-x86_64",
                "python": "3.11.9",
                "source": "pip-report-locked-wheelhouse",
                "directRequirements": direct_requirements,
                "wheels": [{"placeholder": True}],
            }
        ),
        "utf-8",
    )
    path.write_text(
        canonical_json(
            {
                "schemaVersion": "vem-ai-runtime-descriptor/v1",
                "target": "windows-x86_64",
                "python": "3.11.9",
                "directRequirements": direct_requirements,
                "requirementsAiSha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
                "requirementsAiLockSha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                "workerLayout": {"workerExecutable": "vending-vision-ai-worker/vending-vision-ai-worker.exe"},
            }
        ),
        "utf-8",
    )
    return path


def _bind_runtime_lock(runtime_descriptor, release_descriptor):
    raw = release_descriptor.read_text("utf-8")
    runtime_descriptor.with_name("requirements-ai.lock.json").write_text(raw, "utf-8")
    value = json.loads(runtime_descriptor.read_text("utf-8"))
    value["requirementsAiLockSha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    runtime_descriptor.write_text(canonical_json(value), "utf-8")


def test_runtime_descriptor_rejects_requirements_or_lock_tamper(tmp_path):
    descriptor_path = _runtime_descriptor(tmp_path / "ai-runtime-descriptor.json", ["torch==2.8.0+cpu"])

    load_ai_runtime_descriptor(descriptor_path)

    (tmp_path / "requirements-ai.txt").write_text("torch==9.9.0+cpu\n", "utf-8")
    with pytest.raises(AiRuntimeDescriptorError, match="ai_runtime_requirements_digest"):
        load_ai_runtime_descriptor(descriptor_path)

    descriptor_path = _runtime_descriptor(descriptor_path, ["torch==2.8.0+cpu"])
    lock = json.loads((tmp_path / "requirements-ai.lock.json").read_text("utf-8"))
    lock["directRequirements"] = ["torch==9.9.0+cpu"]
    (tmp_path / "requirements-ai.lock.json").write_text(canonical_json(lock), "utf-8")
    with pytest.raises(AiRuntimeDescriptorError, match="ai_runtime_lock_digest"):
        load_ai_runtime_descriptor(descriptor_path)

    descriptor = json.loads(descriptor_path.read_text("utf-8"))
    descriptor["requirementsAiLockSha256"] = hashlib.sha256(
        (tmp_path / "requirements-ai.lock.json").read_bytes()
    ).hexdigest()
    descriptor_path.write_text(canonical_json(descriptor), "utf-8")
    with pytest.raises(AiRuntimeDescriptorError, match="ai_runtime_lock_semantics"):
        load_ai_runtime_descriptor(descriptor_path)


def _release_descriptor(path, *, direct_requirements, wheels):
    descriptor = {
        "schemaVersion": "vem-ai-worker-wheelhouse-release/v1",
        "python": "3.11.9",
        "target": "windows-x86_64",
        "source": "pip-report-locked-wheelhouse",
        "directRequirements": direct_requirements,
        "wheels": wheels,
    }
    path.write_text(canonical_json(descriptor), "utf-8")
    return descriptor


def test_wheelhouse_descriptor_builder_records_exact_wheels_and_rejects_extra(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    first = wheelhouse / "torch-2.8.0-cp311-cp311-win_amd64.whl"
    second = wheelhouse / "diffusers-0.29.2-py3-none-any.whl"
    _write_wheel(first)
    _write_wheel(second)
    descriptor = build_ai_wheelhouse_descriptor(
        wheelhouse,
        requirements=["torch==2.8.0", "diffusers==0.29.2"],
        download_urls={
            first.name: f"https://download.pytorch.org/whl/cpu/{first.name}",
            second.name: f"https://files.pythonhosted.org/packages/test/{second.name}",
        },
    )
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(canonical_json(descriptor), "utf-8")
    runtime_descriptor = _runtime_descriptor(
        tmp_path / "runtime.json", ["torch==2.8.0", "diffusers==0.29.2"]
    )
    _bind_runtime_lock(runtime_descriptor, descriptor_path)

    verify_ai_wheelhouse(descriptor_path, wheelhouse, runtime_descriptor_path=runtime_descriptor)

    descriptor["wheels"][0]["url"] = f"https://download.pytorch.org/locked/{first.name}"
    descriptor_path.write_text(canonical_json(descriptor), "utf-8")
    with pytest.raises(WheelhouseError, match="ai_wheelhouse_runtime_lock_mismatch"):
        verify_ai_wheelhouse(descriptor_path, wheelhouse, runtime_descriptor_path=runtime_descriptor)
    descriptor_path.write_text(
        runtime_descriptor.with_name("requirements-ai.lock.json").read_text("utf-8"), "utf-8"
    )

    (wheelhouse / "extra-1.0.0-py3-none-any.whl").write_bytes(b"extra")
    with pytest.raises(WheelhouseError, match="ai_wheelhouse_extra_or_missing"):
        verify_ai_wheelhouse(descriptor_path, wheelhouse, runtime_descriptor_path=runtime_descriptor)


def test_tracked_ai_wheelhouse_descriptor_is_complete_and_nonempty():
    descriptor = json.loads(Path("requirements-ai.lock.json").read_text(encoding="utf-8"))

    assert len(descriptor["wheels"]) == 34
    assert all(wheel["url"].startswith("https://") for wheel in descriptor["wheels"])
    assert all(wheel["byteSize"] > 0 and len(wheel["sha256"]) == 64 for wheel in descriptor["wheels"])


def test_release_wheelhouse_generates_hashed_requirements_from_descriptor(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    torch = wheelhouse / "torch-2.8.0-cp311-cp311-win_amd64.whl"
    dep = wheelhouse / "filelock-3.19.1-py3-none-any.whl"
    _write_wheel(torch, requires=["filelock>=3.0"])
    _write_wheel(dep)
    runtime_descriptor = _runtime_descriptor(tmp_path / "runtime.json", ["torch==2.8.0"])
    descriptor = _release_descriptor(
        tmp_path / "wheelhouse-release.json",
        direct_requirements=["torch==2.8.0"],
        wheels=[_wheel_entry(torch, direct=True, transitive=False), _wheel_entry(dep, direct=False, transitive=True)],
    )
    descriptor_path = tmp_path / "wheelhouse-release.json"
    requirements_path = tmp_path / "requirements-ai-release.txt"
    _bind_runtime_lock(runtime_descriptor, descriptor_path)

    verify_ai_wheelhouse(
        descriptor_path,
        wheelhouse,
        runtime_descriptor_path=runtime_descriptor,
        requirements_output=requirements_path,
    )

    assert requirements_path.read_text("utf-8").splitlines() == [
        f"filelock==3.19.1 --hash=sha256:{descriptor['wheels'][1]['sha256']}",
        f"torch==2.8.0 --hash=sha256:{descriptor['wheels'][0]['sha256']}",
    ]
    assert generate_hashed_requirements(descriptor, wheelhouse) == requirements_path.read_text("utf-8")


def test_release_wheelhouse_rejects_direct_wheel_outside_pep440_specifier(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wrong = _write_wheel(wheelhouse / "torch-9.9.0+cpu-cp311-cp311-win_amd64.whl")
    runtime_descriptor = _runtime_descriptor(tmp_path / "runtime.json", ["torch==2.8.0+cpu"])
    _release_descriptor(
        tmp_path / "wheelhouse-release.json",
        direct_requirements=["torch==2.8.0+cpu"],
        wheels=[_wheel_entry(wrong, direct=True, transitive=False)],
    )
    _bind_runtime_lock(runtime_descriptor, tmp_path / "wheelhouse-release.json")

    with pytest.raises(WheelhouseError, match="ai_wheelhouse_direct_version_mismatch"):
        verify_ai_wheelhouse(
            tmp_path / "wheelhouse-release.json",
            wheelhouse,
            runtime_descriptor_path=runtime_descriptor,
        )


def test_release_descriptor_rejects_wrong_source_and_nonboolean_roles(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheel = _write_wheel(wheelhouse / "demo-1.0.0-py3-none-any.whl")
    runtime = _runtime_descriptor(tmp_path / "runtime.json", ["demo==1.0.0"])
    descriptor_path = tmp_path / "release.json"
    descriptor = _release_descriptor(
        descriptor_path,
        direct_requirements=["demo==1.0.0"],
        wheels=[_wheel_entry(wheel, direct=True, transitive=False)],
    )
    descriptor["source"] = "release-provided-wheelhouse"
    descriptor_path.write_text(canonical_json(descriptor), "utf-8")
    _bind_runtime_lock(runtime, descriptor_path)
    with pytest.raises(WheelhouseError, match="ai_wheelhouse_descriptor_source"):
        verify_ai_wheelhouse(descriptor_path, wheelhouse, runtime_descriptor_path=runtime)

    descriptor["source"] = "pip-report-locked-wheelhouse"
    descriptor["wheels"][0]["direct"] = 1
    descriptor_path.write_text(canonical_json(descriptor), "utf-8")
    _bind_runtime_lock(runtime, descriptor_path)
    with pytest.raises(WheelhouseError, match="ai_wheelhouse_entry_role"):
        verify_ai_wheelhouse(descriptor_path, wheelhouse, runtime_descriptor_path=runtime)


def test_release_wheelhouse_rejects_wrong_target_or_untracked_direct_requirement(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "torch-2.8.0-cp311-cp311-manylinux_x86_64.whl"
    _write_wheel(wheel)
    runtime_descriptor = _runtime_descriptor(tmp_path / "runtime.json", ["torch==2.8.0"])
    descriptor = {
        **_release_descriptor(
            tmp_path / "wheelhouse-release.json",
            direct_requirements=["torch==2.8.0"],
            wheels=[_wheel_entry(wheel, direct=True, transitive=False)],
        ),
        "target": "linux-x86_64",
    }
    descriptor_path = tmp_path / "wheelhouse-release.json"
    descriptor_path.write_text(canonical_json(descriptor), "utf-8")
    _bind_runtime_lock(runtime_descriptor, descriptor_path)

    with pytest.raises(WheelhouseError, match="ai_wheelhouse_target_mismatch"):
        verify_ai_wheelhouse(descriptor_path, wheelhouse, runtime_descriptor_path=runtime_descriptor)


def test_release_wheelhouse_requires_marker_evaluated_transitive_closure(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    root = _write_wheel(
        wheelhouse / "rootpkg-1.0.0-py3-none-any.whl",
        requires=[
            "neededdep>=2.0; python_version >= '3.11'",
            "linuxonly>=1.0; sys_platform == 'linux'",
        ],
    )
    linux_only = _write_wheel(wheelhouse / "linuxonly-1.0.0-py3-none-any.whl")
    runtime_descriptor = _runtime_descriptor(tmp_path / "runtime.json", ["rootpkg==1.0.0"])
    descriptor = _release_descriptor(
        tmp_path / "wheelhouse-release.json",
        direct_requirements=["rootpkg==1.0.0"],
        wheels=[_wheel_entry(root, direct=True, transitive=False), _wheel_entry(linux_only, direct=False, transitive=True)],
    )
    _bind_runtime_lock(runtime_descriptor, tmp_path / "wheelhouse-release.json")

    with pytest.raises(WheelhouseError, match="ai_wheelhouse_missing_dependency"):
        verify_ai_wheelhouse(tmp_path / "wheelhouse-release.json", wheelhouse, runtime_descriptor_path=runtime_descriptor)

    needed = _write_wheel(wheelhouse / "neededdep-2.0.0-py3-none-any.whl")
    descriptor["wheels"].append(_wheel_entry(needed, direct=False, transitive=True))
    (tmp_path / "wheelhouse-release.json").write_text(canonical_json(descriptor), "utf-8")
    _bind_runtime_lock(runtime_descriptor, tmp_path / "wheelhouse-release.json")
    with pytest.raises(WheelhouseError, match="ai_wheelhouse_extra_unrelated"):
        verify_ai_wheelhouse(tmp_path / "wheelhouse-release.json", wheelhouse, runtime_descriptor_path=runtime_descriptor)


def test_release_wheelhouse_rejects_sdist_wrong_abi_and_duplicate_distribution(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    root = _write_wheel(wheelhouse / "rootpkg-1.0.0-py3-none-any.whl")
    (wheelhouse / "rootpkg-1.0.0.tar.gz").write_bytes(b"sdist")
    runtime_descriptor = _runtime_descriptor(tmp_path / "runtime.json", ["rootpkg==1.0.0"])
    _release_descriptor(
        tmp_path / "wheelhouse-release.json",
        direct_requirements=["rootpkg==1.0.0"],
        wheels=[_wheel_entry(root, direct=True, transitive=False)],
    )
    _bind_runtime_lock(runtime_descriptor, tmp_path / "wheelhouse-release.json")

    with pytest.raises(WheelhouseError, match="ai_wheelhouse_extra_or_missing"):
        verify_ai_wheelhouse(tmp_path / "wheelhouse-release.json", wheelhouse, runtime_descriptor_path=runtime_descriptor)

    (wheelhouse / "rootpkg-1.0.0.tar.gz").unlink()
    root.unlink()
    root = _write_wheel(wheelhouse / "rootpkg-1.0.0-cp310-cp310-win_amd64.whl")
    _release_descriptor(
        tmp_path / "wheelhouse-release.json",
        direct_requirements=["rootpkg==1.0.0"],
        wheels=[_wheel_entry(root, direct=True, transitive=False)],
    )
    _bind_runtime_lock(runtime_descriptor, tmp_path / "wheelhouse-release.json")
    with pytest.raises(WheelhouseError, match="ai_wheelhouse_target_mismatch"):
        verify_ai_wheelhouse(tmp_path / "wheelhouse-release.json", wheelhouse, runtime_descriptor_path=runtime_descriptor)

    root.unlink()
    root = _write_wheel(wheelhouse / "rootpkg-1.0.0-py3-none-any.whl")
    duplicate = _write_wheel(wheelhouse / "rootpkg-1.0.0-cp311-cp311-win_amd64.whl")
    descriptor = _release_descriptor(
        tmp_path / "wheelhouse-release.json",
        direct_requirements=["rootpkg==1.0.0"],
        wheels=[_wheel_entry(root, direct=True, transitive=False), _wheel_entry(duplicate, direct=True, transitive=False)],
    )
    (tmp_path / "wheelhouse-release.json").write_text(canonical_json(descriptor), "utf-8")
    _bind_runtime_lock(runtime_descriptor, tmp_path / "wheelhouse-release.json")
    with pytest.raises(WheelhouseError, match="ai_wheelhouse_duplicate"):
        verify_ai_wheelhouse(tmp_path / "wheelhouse-release.json", wheelhouse, runtime_descriptor_path=runtime_descriptor)
