# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata


ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "official-ai-model-pack-descriptor.json"), "."),
    (str(ROOT / "ai-runtime-descriptor.json"), "."),
    (str(ROOT / "official-ai-source-descriptor.json"), "."),
    (str(ROOT / "vision" / "vendor" / "catvton" / "LICENSE"), "vision/vendor/catvton"),
    (str(ROOT / "vision" / "vendor" / "catvton" / "PROVENANCE.md"), "vision/vendor/catvton"),
]
hiddenimports = [
    "vision.ai_attempt_worker",
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
