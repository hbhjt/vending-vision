# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


ROOT = Path(SPECPATH)
CONTRACT_ROOT = ROOT / "contracts" / "vem_vision_v2"
CONTRACT_DATA_FILES = [
    (CONTRACT_ROOT / "manifest.json", "contracts/vem_vision_v2"),
    (CONTRACT_ROOT / "__init__.py", "contracts/vem_vision_v2"),
    (CONTRACT_ROOT / "python" / "__init__.py", "contracts/vem_vision_v2/python"),
    (CONTRACT_ROOT / "python" / "vision_v2_models.py", "contracts/vem_vision_v2/python"),
    (CONTRACT_ROOT / "vision-v2.schema.json", "contracts/vem_vision_v2"),
    (CONTRACT_ROOT / "fixtures" / "valid.json", "contracts/vem_vision_v2/fixtures"),
    (CONTRACT_ROOT / "fixtures" / "invalid.json", "contracts/vem_vision_v2/fixtures"),
]

datas = [
    (str(ROOT / "config.json"), "."),
    (str(ROOT / "config"), "config"),
    (str(ROOT / "dashboard"), "dashboard"),
    (str(ROOT / "models"), "models"),
    *[(str(source), destination) for source, destination in CONTRACT_DATA_FILES],
]
datas += collect_data_files("mediapipe")
datas += collect_data_files("cv2_enumerate_cameras")

binaries = []
binaries += collect_dynamic_libs("mediapipe")
binaries += collect_dynamic_libs("cv2_enumerate_cameras")

hiddenimports = [
    "vision.v2_contract_bundle",
    "contracts.vem_vision_v2.python.vision_v2_models",
    "uvicorn.lifespan.on",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
]
hiddenimports += collect_submodules("mediapipe")
hiddenimports += [
    "cv2_enumerate_cameras",
    "cv2_enumerate_cameras.windows_backend",
    "cv2_enumerate_cameras._windows_backend",
]

metadata_packages = [
    "fastapi",
    "uvicorn",
    "starlette",
    "pydantic",
    "mediapipe",
    "opencv-contrib-python",
    "openvino",
    "jsonschema",
    "cv2-enumerate-cameras",
]

for package in metadata_packages:
    try:
        datas += copy_metadata(package)
    except Exception:
        pass


a = Analysis(
    ["run_vision_server.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PIL.ImageQt", "PyQt5", "PySide2", "PySide6"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vending-vision",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="vending-vision",
)
