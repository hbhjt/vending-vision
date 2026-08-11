# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata


ROOT = Path(SPECPATH)
hiddenimports = [
    "scripts.ai_model_pack_release",
    "scripts.candidate_artifact_manifest",
    "scripts.verify_trusted_candidate_inputs",
    "candidate_artifact_manifest",
    "vision.ai_model_pack",
    "vision.ai_runtime_descriptor",
    "vision.precutover_companion",
    "vision.process_supervisor",
    "PyInstaller.archive.readers",
]
hiddenimports += collect_submodules("packaging")
datas = []
for package in ("packaging",):
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

a = Analysis(
    [str(ROOT / "run_precutover_verifier.py")],
    pathex=[str(ROOT), str(ROOT / "scripts")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "cv2",
        "numpy",
        "scipy",
        "PIL",
        "tests",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vending-vision-precutover-verifier",
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
    name="vending-vision-precutover-verifier",
)
