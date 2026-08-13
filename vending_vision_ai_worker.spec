# -*- mode: python ; coding: utf-8 -*-

import json
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata


ROOT = Path(SPECPATH)
OFFICIAL_AI_SOURCE_DESCRIPTOR_PATH = ROOT / "official-ai-source-descriptor.json"
REGIONAL_EVALUATOR_DESCRIPTOR_PATH = ROOT / "regional-evaluator-descriptor.json"
OFFICIAL_AI_SOURCE_DATA_FILES = []
for source in json.loads(OFFICIAL_AI_SOURCE_DESCRIPTOR_PATH.read_text("utf-8"))["sources"]:
    relative = source["path"]
    if relative.endswith(".py"):
        OFFICIAL_AI_SOURCE_DATA_FILES.append((str(ROOT / relative), str(Path(relative).parent)))
REGIONAL_EVALUATOR_SOURCE_DATA_FILES = []
official_source_paths = {
    source["path"]
    for source in json.loads(OFFICIAL_AI_SOURCE_DESCRIPTOR_PATH.read_text("utf-8"))["sources"]
}
for source in json.loads(REGIONAL_EVALUATOR_DESCRIPTOR_PATH.read_text("utf-8"))["sources"]:
    relative = source["path"]
    if relative.endswith(".py") and relative not in official_source_paths:
        REGIONAL_EVALUATOR_SOURCE_DATA_FILES.append((str(ROOT / relative), str(Path(relative).parent)))

datas = [
    (str(ROOT / "vision" / "_build_version.py"), "vision"),
    (str(ROOT / "official-ai-model-pack-descriptor.json"), "."),
    (str(ROOT / "ai-runtime-descriptor.json"), "."),
    (str(ROOT / "requirements-ai.txt"), "."),
    (str(ROOT / "requirements-ai.lock.json"), "."),
    (str(ROOT / "official-ai-source-descriptor.json"), "."),
    (str(ROOT / "regional-evaluator-descriptor.json"), "."),
    (str(ROOT / "vision" / "vendor" / "catvton" / "LICENSE"), "vision/vendor/catvton"),
    (str(ROOT / "vision" / "vendor" / "catvton" / "PROVENANCE.md"), "vision/vendor/catvton"),
    *OFFICIAL_AI_SOURCE_DATA_FILES,
    *REGIONAL_EVALUATOR_SOURCE_DATA_FILES,
]
hiddenimports = [
    "vision.ai_attempt_worker",
    "vision.regional_evaluator",
    "vision.regional_evaluator_provenance",
    "vision.ai_model_pack",
    "vision.ai_runtime_descriptor",
    "vision.ai_attempt_process",
    "vision.process_supervisor",
    "vision.catvton_pose_masks",
    "vision.catvton_preprocess",
]
hiddenimports += collect_submodules("vision.vendor.catvton")
hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("torchvision")
hiddenimports += collect_submodules("diffusers")
hiddenimports += collect_submodules("transformers")
hiddenimports += collect_submodules("accelerate")
hiddenimports += collect_submodules("safetensors")
hiddenimports += collect_submodules("cv2")

binaries = []
binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("torchvision")
binaries += collect_dynamic_libs("cv2")

for package in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors", "opencv-python-headless", "numpy", "Pillow"):
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

a = Analysis(
    ["run_ai_attempt_worker.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tests", "PIL.ImageQt", "PyQt5", "PySide2", "PySide6"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vending-vision-ai-worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="vending-vision-ai-worker",
)
