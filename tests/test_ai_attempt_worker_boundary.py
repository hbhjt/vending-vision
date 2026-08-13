import inspect
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image
import pytest

from vision.ai_model_pack import canonical_ai_model_manifest_json
from vision import ai_attempt_worker

ROOT = Path(__file__).parents[1]


def write_worker_pack(root: Path) -> None:
    files = {
        "CatVTON/SCHP/exp-schp-201908301523-atr.pth": b"atr",
        "CatVTON/SCHP/exp-schp-201908261155-lip.pth": b"lip",
        "CatVTON/mix-48k-1024/attention/model.safetensors": b"attention",
        "inpainting/scheduler/scheduler_config.json": b"{}",
        "inpainting/unet/config.json": b"{}",
        "inpainting/unet/diffusion_pytorch_model.bin": b"unet",
        "vae/config.json": b"{}",
        "vae/diffusion_pytorch_model.safetensors": b"vae",
    }
    manifest_files = []
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest_files.append(
            {
                "path": relative,
                "upstreamPath": relative.split("/", maxsplit=1)[1] if "/" in relative else relative,
                "upstream": "catvton" if relative.startswith("CatVTON/") else ("inpainting" if relative.startswith("inpainting/") else "vae"),
                "role": relative.replace("/", "_").replace(".", "_").replace("-", "_"),
                "format": relative.rsplit(".", maxsplit=1)[-1],
                "byteSize": path.stat().st_size,
                "sha256": __import__("hashlib").sha256(content).hexdigest(),
            }
        )
    manifest = {
        "schemaVersion": "vem-official-ai-model-pack-descriptor/v2",
        "catvtonSourceRevision": "test-source",
        "totalByteSize": sum(item["byteSize"] for item in manifest_files),
        "upstreams": [
            {"id": "catvton", "repository": "zhengchong/CatVTON", "revision": "test-catvton"},
            {"id": "inpainting", "repository": "booksforcharlie/stable-diffusion-inpainting", "revision": "test-inpainting"},
            {"id": "vae", "repository": "stabilityai/sd-vae-ft-mse", "revision": "test-vae"},
        ],
        "files": sorted(manifest_files, key=lambda item: item["path"]),
    }
    (root / "ai-model-manifest.json").write_text(
        canonical_ai_model_manifest_json(manifest),
        "utf-8",
    )


def write_miniature_catvton_modules(root: Path) -> None:
    (root / "sitecustomize.py").write_text(
        """
import json
import os
import sys
import types

descriptor = os.environ.get("CATVTON_MINIATURE_DESCRIPTOR")
if descriptor:
    import vision.ai_model_pack as pack
    loaded = json.loads(descriptor)
    pack.load_official_ai_model_pack_descriptor = lambda: loaded

if os.environ.get("CATVTON_MINIATURE_SKIP_TORCH"):
    sys.modules["torch"] = None
else:
    torch = types.ModuleType("torch")
    class inference_mode:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
    class Generator:
        def __init__(self, device="cpu"):
            self.device = device
        def manual_seed(self, seed):
            self.seed = seed
            return self
    torch.inference_mode = inference_mode
    torch.float32 = "float32"
    torch.set_num_threads = lambda _threads: None
    torch.Generator = Generator
    sys.modules["torch"] = torch

for name in ("torchvision", "accelerate", "diffusers", "transformers", "safetensors", "scipy", "tqdm"):
    sys.modules.setdefault(name, types.ModuleType(name))

import importlib.metadata
_real_version = importlib.metadata.version
def _mini_version(name):
    if name in {"torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors", "scipy", "tqdm", "opencv-python-headless"}:
        return {
            "torch": "2.8.0+cpu",
            "torchvision": "0.23.0+cpu",
            "diffusers": "0.29.2",
            "transformers": "4.53.3",
            "accelerate": "0.31.0",
            "safetensors": "0.5.3",
            "scipy": "1.16.1",
            "tqdm": "4.67.1",
            "opencv-python-headless": "4.12.0.88",
        }[name]
    return _real_version(name)
importlib.metadata.version = _mini_version

from PIL import Image
schp_module = types.ModuleType("vision.vendor.catvton.model.SCHP")
class SCHP:
    def __init__(self, checkpoint, device="cpu"):
        self.checkpoint = str(checkpoint)
    def __call__(self, image):
        width, height = image.size
        parse = Image.new("L", (width, height), 0)
        pixels = parse.load()
        for y in range(height // 4, min(height, height * 3 // 4)):
            for x in range(width // 4, min(width, width * 3 // 4)):
                pixels[x, y] = 5
        return parse
schp_module.SCHP = SCHP
sys.modules["vision.vendor.catvton.model.SCHP"] = schp_module

pipeline_module = types.ModuleType("vision.vendor.catvton.model.pipeline")
class CatVTONPipeline:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self, *, image, condition_image, mask, num_inference_steps, guidance_scale, height, width, generator):
        record = {
            "api": "CatVTONPipeline",
            "base_ckpt": self.kwargs.get("base_ckpt"),
            "vae_ckpt": self.kwargs.get("vae_ckpt"),
            "attn_ckpt": self.kwargs.get("attn_ckpt"),
            "attn_ckpt_version": self.kwargs.get("attn_ckpt_version"),
            "device": self.kwargs.get("device"),
            "weight_dtype": str(self.kwargs.get("weight_dtype")),
            "local_files_only": self.kwargs.get("local_files_only"),
            "skip_safety_check": self.kwargs.get("skip_safety_check"),
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
        }
        record_path = os.environ.get("CATVTON_MINIATURE_RECORD")
        if record_path:
            from pathlib import Path
            Path(record_path).write_text(json.dumps(record, sort_keys=True), "utf-8")
        generated = Image.blend(image.convert("RGB"), condition_image.convert("RGB"), 0.65)
        return [generated]
pipeline_module.CatVTONPipeline = CatVTONPipeline
sys.modules["vision.vendor.catvton.model.pipeline"] = pipeline_module

import vision.catvton_pose_masks as masks
import vision.source_provenance as provenance
import vision.regional_evaluator_provenance as regional_provenance
if os.environ.get("CATVTON_MINIATURE_SOURCE_TAMPER"):
    provenance.verify_official_source_provenance = lambda: False
if os.environ.get("CATVTON_MINIATURE_REGIONAL_TAMPER"):
    regional_provenance.verify_regional_evaluator_provenance = lambda: False
class L:
    def __init__(self, x, y, visibility=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility
landmarks = [L(0.5, 0.5, 0.0) for _ in range(33)]
for idx, xy in {
    0: (0.50, 0.14),
    2: (0.47, 0.13),
    5: (0.53, 0.13),
    7: (0.43, 0.16),
    8: (0.57, 0.16),
    11: (0.34, 0.30),
    12: (0.66, 0.30),
    13: (0.25, 0.48),
    14: (0.75, 0.48),
    15: (0.22, 0.68),
    16: (0.78, 0.68),
    23: (0.40, 0.70),
    24: (0.60, 0.70),
}.items():
    landmarks[idx] = L(*xy)
pose = types.SimpleNamespace(pose_landmarks=types.SimpleNamespace(landmark=landmarks))
masks._detect_pose = lambda _person_rgb: pose
""",
        "utf-8",
    )
    (root / "torch.py").write_text(
        """
class inference_mode:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


float32 = "float32"


def set_num_threads(_threads):
    return None


class Generator:
    def __init__(self, device="cpu"):
        self.device = device
    def manual_seed(self, seed):
        self.seed = seed
        return self
""",
        "utf-8",
    )


def write_png(path: Path, color: tuple[int, int, int, int], size: tuple[int, int] = (96, 128)) -> None:
    Image.new("RGBA", size, color).save(path, format="PNG")


def test_worker_probe_accepts_only_official_manifest_and_no_fake_arguments(tmp_path):
    pack = tmp_path / "pack"
    modules = tmp_path / "miniature-modules"
    pack.mkdir()
    modules.mkdir()
    write_worker_pack(pack)
    write_miniature_catvton_modules(modules)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{modules}{os.pathsep}{ROOT}"
    env["CATVTON_MINIATURE_DESCRIPTOR"] = (pack / "ai-model-manifest.json").read_text("utf-8")

    probe = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision.ai_attempt_worker",
            "--model-pack",
            str(pack),
            "--probe",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert probe.returncode == 0, f"{probe.stdout}{probe.stderr}"
    probe_payload = json.loads(probe.stdout)
    assert probe_payload["probe"] == "official-catvton-worker"
    assert probe_payload["torch"] == "2.8.0+cpu"

    fake = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision.ai_attempt_worker",
            "--model-pack",
            str(pack),
            "--probe",
            "--fake-worker",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert fake.returncode != 0
    assert "--fake-worker" in fake.stderr


def test_worker_runtime_probe_does_not_require_model_pack_and_reports_versions(tmp_path):
    modules = tmp_path / "miniature-modules"
    modules.mkdir()
    write_miniature_catvton_modules(modules)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{modules}{os.pathsep}{ROOT}"

    probe = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision.ai_attempt_worker",
            "--probe-runtime",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert probe.returncode == 0, f"{probe.stdout}{probe.stderr}"
    payload = json.loads(probe.stdout)
    assert payload["probe"] == "official-catvton-worker-runtime"
    assert payload["torch"] == "2.8.0+cpu"
    assert payload["catvtonSourceRevision"] == "3b795364a4d2f3b5adb365f39cdea376d20bc53c"

    legacy_probe = subprocess.run(
        [sys.executable, "run_vision_server.py", "--ai-attempt-worker", "--probe-runtime"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert legacy_probe.returncode == 0, f"{legacy_probe.stdout}{legacy_probe.stderr}"
    assert json.loads(legacy_probe.stdout)["probe"] == "official-catvton-worker-runtime"


def test_worker_probe_rejects_source_descriptor_tamper(tmp_path):
    pack = tmp_path / "pack"
    modules = tmp_path / "miniature-modules"
    pack.mkdir()
    modules.mkdir()
    write_worker_pack(pack)
    write_miniature_catvton_modules(modules)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{modules}{os.pathsep}{ROOT}"
    env["CATVTON_MINIATURE_DESCRIPTOR"] = (pack / "ai-model-manifest.json").read_text("utf-8")
    env["CATVTON_MINIATURE_SOURCE_TAMPER"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision.ai_attempt_worker",
            "--model-pack",
            str(pack),
            "--probe",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 2
    assert "official_catvton_probe_failed" in completed.stderr


def test_worker_runtime_probe_rejects_regional_evaluator_descriptor_tamper(tmp_path):
    modules = tmp_path / "miniature-modules"
    modules.mkdir()
    write_miniature_catvton_modules(modules)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{modules}{os.pathsep}{ROOT}"
    env["CATVTON_MINIATURE_REGIONAL_TAMPER"] = "1"

    completed = subprocess.run(
        [sys.executable, "-m", "vision.ai_attempt_worker", "--probe-runtime"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 2
    assert "official_catvton_regional_evaluator_provenance_mismatch" in completed.stderr


def test_worker_customer_attempt_runs_catvton_pipeline_and_writes_private_png(tmp_path):
    pack = tmp_path / "pack"
    modules = tmp_path / "miniature-modules"
    work = tmp_path / "work"
    pack.mkdir()
    modules.mkdir()
    work.mkdir()
    write_worker_pack(pack)
    write_miniature_catvton_modules(modules)
    person = work / "person.png"
    garment = work / "garment.png"
    output = work / "output.png"
    regional = work / "regional-evidence.json"
    record = work / "record.json"
    write_png(person, (20, 40, 90, 255))
    write_png(garment, (220, 40, 20, 255), size=(80, 110))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{modules}{os.pathsep}{ROOT}"
    env["CATVTON_MINIATURE_DESCRIPTOR"] = (pack / "ai-model-manifest.json").read_text("utf-8")
    env["CATVTON_MINIATURE_RECORD"] = str(record)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision.ai_attempt_worker",
            "--model-pack",
            str(pack),
            "--person",
            str(person),
            "--garment",
            str(garment),
            "--template",
            "tshirt_short_sleeve",
            "--output",
            str(output),
            "--regional-evidence-output",
            str(regional),
            "--captured-source",
            json.dumps(
                {
                    "adapter": "recorded_video",
                    "configSha256": "7" * 64,
                    "decodedFrameCount": 42,
                    "fixtureSha256": "8" * 64,
                    "frameIndex": 7,
                    "relabeled": False,
                    "role": "front",
                    "synthetic": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "--width",
            "64",
            "--height",
            "96",
            "--steps",
            "4",
            "--seed",
            "123",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert output.is_file()
    assert regional.is_file()
    with Image.open(output) as result:
        assert result.format == "PNG"
        assert result.size == (96, 128)
    pipeline_call = json.loads(record.read_text("utf-8"))
    assert pipeline_call["api"] == "CatVTONPipeline"
    assert pipeline_call["attn_ckpt_version"] == "mix"
    assert pipeline_call["device"] == "cpu"
    assert pipeline_call["local_files_only"] is True
    assert pipeline_call["skip_safety_check"] is True
    assert Path(pipeline_call["base_ckpt"]).name == "inpainting"
    assert Path(pipeline_call["vae_ckpt"]).name == "vae"
    assert "CatVTON" in pipeline_call["attn_ckpt"]

    raw_regional = regional.read_text("utf-8")
    sidecar = json.loads(raw_regional)
    assert raw_regional == json.dumps(
        sidecar, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert sidecar["schemaVersion"] == "vem-ai-regional-evidence/v1"
    assert sidecar["kind"] == "regional-evidence"
    assert sidecar["attempt"] == {
        "acquisitionSource": "direct_recorded_frame",
        "decodedHeight": 128,
        "decodedWidth": 96,
        "garmentSha256": hashlib.sha256(garment.read_bytes()).hexdigest(),
        "inputSha256": hashlib.sha256(person.read_bytes()).hexdigest(),
        "recordedFixtureSha256": "8" * 64,
        "resultSha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sourceCamera": "front",
    }
    assert sidecar["evaluator"] == {
        "algorithm": "rgb-absolute-delta-rle/v1",
        "atr": "schp-atr",
        "lip": "schp-lip",
        "pose": "mediapipe-pose",
        "sourceDescriptorSha256": hashlib.sha256(
            (ROOT / "regional-evaluator-descriptor.json").read_bytes()
        ).hexdigest(),
    }
    assert sidecar["masks"]["width"] == 96
    assert sidecar["masks"]["height"] == 128
    upper_runs = sidecar["masks"]["upperBody"]["runs"]
    protected_runs = sidecar["masks"]["protectedRegion"]["runs"]
    assert upper_runs
    assert protected_runs
    upper_pixels = {pixel for start, length in upper_runs for pixel in range(start, start + length)}
    protected_pixels = {
        pixel for start, length in protected_runs for pixel in range(start, start + length)
    }
    assert upper_pixels.isdisjoint(protected_pixels)

    with Image.open(person) as person_image, Image.open(output) as result_image:
        person_rgb = person_image.convert("RGB")
        result_rgb = result_image.convert("RGB")
        for name, pixels in (("upperBody", upper_pixels), ("protectedRegion", protected_pixels)):
            changed = 0
            delta = 0
            for pixel in pixels:
                x, y = pixel % person_rgb.width, pixel // person_rgb.width
                before, after = person_rgb.getpixel((x, y)), result_rgb.getpixel((x, y))
                differences = [abs(after[index] - before[index]) for index in range(3)]
                changed += int(any(differences))
                delta += sum(differences)
            measurement = sidecar["measurements"][name]
            assert measurement["sampledPixels"] == len(pixels)
            assert measurement["changedPixels"] == changed
            assert measurement["changedFractionBps"] == changed * 10_000 // len(pixels)
            assert measurement["meanDelta"] == delta // (len(pixels) * 3)
            assert measurement["verdict"] == (
                "changed"
                if name == "upperBody" and changed > 0
                else (
                    "insufficient_change"
                    if name == "upperBody"
                    else ("preserved" if changed == 0 else "changed")
                )
            )
    assert sidecar["policy"] == {
        "schemaVersion": "vem-ai-regional-evidence-policy/v1",
        "sha256": "7e1f34b2a96abecd245de6ba44c7b4b0fecd2ebeb459e10fa0cc6402955abe1d",
    }
    assert sidecar["verdict"] in {"passed", "regional_check_failed"}

    # The explicit integration job supplies the pinned VEM contract module.
    # A standalone Vision checkout must not depend on an incidental sibling
    # repository layout.
    contract_module = os.environ.get("VEM_REGIONAL_EVIDENCE_MODULE")
    if contract_module is None:
        return
    contract_module_path = Path(contract_module).resolve()
    assert contract_module_path.is_file()

    # This is the cross-repository contract: the real Vision worker sidecar
    # satisfies VEM identity validation before the intentionally pending
    # two-garment calibration gate rejects it.
    attempt_id = "0198f44e-21bd-7c62-8f52-b7c86cc2d099"
    artifact_root = work / "vem-artifacts"
    relative = f"regional/short/{attempt_id}.regional-evidence.json"
    artifact_sidecar = artifact_root / relative
    artifact_sidecar.parent.mkdir(parents=True)
    artifact_sidecar.write_bytes(regional.read_bytes())
    validation = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            """
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
const { validateAiRegionalEvidence } = await import(process.argv[1]);
const [artifactRoot, sidecarPath, attemptId, relative] = process.argv.slice(2);
const raw = readFileSync(sidecarPath);
const sidecar = JSON.parse(raw);
const sha256 = createHash("sha256").update(raw).digest("hex");
console.log(JSON.stringify(validateAiRegionalEvidence({
  attemptId,
  caseKey: "short",
  garment: { sha256: sidecar.attempt.garmentSha256 },
  input: { sha256: sidecar.attempt.inputSha256 },
  result: {
    decodedHeight: sidecar.attempt.decodedHeight,
    decodedWidth: sidecar.attempt.decodedWidth,
    sha256: sidecar.attempt.resultSha256,
  },
  regionalEvidence: {
    path: relative,
    schemaVersion: "vem-ai-regional-evidence-reference/v1",
    sha256,
    verdict: sidecar.verdict,
  },
}, artifactRoot, { files: [{
  byteLength: raw.byteLength,
  kind: "supportingEvidence",
  path: sidecarPath,
  sha256,
  track: "aiVirtualTryOn",
}] })));
""",
            contract_module_path.as_uri(),
            str(artifact_root),
            str(artifact_sidecar),
            attempt_id,
            relative,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert validation.returncode == 0, validation.stderr
    assert json.loads(validation.stdout) == {
        "ok": False,
        "reason": "AI regional evidence policy awaits Issue10 two-garment calibration",
    }


def test_ci_declares_pinned_vem_regional_evidence_contract_job():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")

    assert "regional-evidence-contract:" in workflow
    assert "repository: YKDZ/vem" in workflow
    assert "ref: 2dad5d56e752b2a0d19167bc0a6e32871ef3587e" in workflow
    assert "VEM_REGIONAL_EVIDENCE_MODULE:" in workflow
    assert "scripts/testbed/ai-regional-evidence-policy.json" in workflow
    assert workflow.count("fetch-depth: 0") == 5
    assert "tests/test_ai_attempt_worker_boundary.py::test_worker_customer_attempt_runs_catvton_pipeline_and_writes_private_png" in workflow


@pytest.mark.parametrize(
    "captured_source",
    [
        None,
        {"adapter": "recorded_video", "role": "front"},
        {
            "adapter": "recorded_video",
            "configSha256": "7" * 64,
            "decodedFrameCount": 42,
            "fixtureSha256": "8" * 64,
            "frameIndex": 42,
            "relabeled": False,
            "role": "front",
            "synthetic": False,
        },
    ],
)
def test_worker_regional_evidence_requires_exact_recorded_source_and_leaves_no_sidecar(
    tmp_path, captured_source
):
    pack = tmp_path / "pack"
    modules = tmp_path / "miniature-modules"
    work = tmp_path / "work"
    pack.mkdir()
    modules.mkdir()
    work.mkdir()
    write_worker_pack(pack)
    write_miniature_catvton_modules(modules)
    person = work / "person.png"
    garment = work / "garment.png"
    output = work / "output.png"
    regional = work / "regional-evidence.json"
    write_png(person, (20, 40, 90, 255))
    write_png(garment, (220, 40, 20, 255), size=(80, 110))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{modules}{os.pathsep}{ROOT}"
    env["CATVTON_MINIATURE_DESCRIPTOR"] = (pack / "ai-model-manifest.json").read_text(
        "utf-8"
    )
    command = [
        sys.executable,
        "-m",
        "vision.ai_attempt_worker",
        "--model-pack",
        str(pack),
        "--person",
        str(person),
        "--garment",
        str(garment),
        "--output",
        str(output),
        "--regional-evidence-output",
        str(regional),
        "--width",
        "64",
        "--height",
        "96",
        "--steps",
        "1",
    ]
    if captured_source is not None:
        command.extend(
            [
                "--captured-source",
                json.dumps(captured_source, sort_keys=True, separators=(",", ":")),
            ]
        )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode != 0
    assert not regional.exists()


@pytest.mark.parametrize(
    "case, expected",
    [
        ("missing_model", "ai_model_pack_digest"),
        ("missing_dependency", "official_catvton_dependency_missing"),
        ("invalid_input", "official_catvton_invalid_garment"),
    ],
)
def test_worker_customer_attempt_failures_are_typed_nonzero_and_leave_no_output(
    tmp_path,
    case,
    expected,
):
    pack = tmp_path / "pack"
    modules = tmp_path / "miniature-modules"
    work = tmp_path / "work"
    pack.mkdir()
    modules.mkdir()
    work.mkdir()
    write_worker_pack(pack)
    write_miniature_catvton_modules(modules)
    if case == "missing_model":
        (pack / "CatVTON/SCHP/exp-schp-201908261155-lip.pth").unlink()
    if case == "missing_dependency":
        (modules / "torch.py").unlink()
    person = work / "person.png"
    garment = work / "garment.png"
    output = work / "output.png"
    write_png(person, (20, 40, 90, 255))
    if case == "invalid_input":
        garment.write_bytes(b"not-a-png")
    else:
        write_png(garment, (220, 40, 20, 255), size=(80, 110))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{modules}{os.pathsep}{ROOT}"
    env["CATVTON_MINIATURE_DESCRIPTOR"] = (pack / "ai-model-manifest.json").read_text("utf-8")
    if case == "missing_dependency":
        env["CATVTON_MINIATURE_SKIP_TORCH"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision.ai_attempt_worker",
            "--model-pack",
            str(pack),
            "--person",
            str(person),
            "--garment",
            str(garment),
            "--template",
            "tshirt_short_sleeve",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode != 0
    assert expected in completed.stderr
    assert not output.exists()


def test_worker_source_hard_guards_downloads_and_probe_does_not_load_or_infer():
    source = inspect.getsource(ai_attempt_worker)

    assert "HF_HUB_OFFLINE" in source
    assert "TRANSFORMERS_OFFLINE" in source
    assert "HF_DATASETS_OFFLINE" in source
    assert "socket.socket = _blocked_socket" in source
    assert "snapshot_download" not in source
    assert "huggingface_hub" not in source
    assert "fake" not in source.lower()
    assert "exp-schp-201908261155-lip.pth" not in source
    assert "exp-schp-201908301523-atr.pth" not in source
    assert "protected_mask_to_original" not in source
    assert source.index("if args.probe:") < source.index(
        "metrics = run_regional_evaluator_attempt("
    )


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
