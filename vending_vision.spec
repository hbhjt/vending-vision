# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "config.json"), "."),
    (str(ROOT / "config"), "config"),
    (str(ROOT / "dashboard"), "dashboard"),
    (str(ROOT / "models"), "models"),
]
datas += collect_data_files("mediapipe")
datas += collect_data_files("cv2_enumerate_cameras")

binaries = []
binaries += collect_dynamic_libs("mediapipe")
binaries += collect_dynamic_libs("cv2_enumerate_cameras")
binaries += collect_dynamic_libs("cryptography")

hiddenimports = [
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
    "cryptography.hazmat.backends.openssl",
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
    "cryptography",
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
