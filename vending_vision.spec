# -*- mode: python ; coding: utf-8 -*-

import json
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


ROOT = Path(SPECPATH)
CONTRACT_ROOT = ROOT / "contracts" / "vem_vision_v2"
OFFICIAL_AI_SOURCE_DESCRIPTOR_PATH = ROOT / "official-ai-source-descriptor.json"
CONTRACT_DATA_FILES = [
    (CONTRACT_ROOT / "manifest.json", "contracts/vem_vision_v2"),
    (CONTRACT_ROOT / "__init__.py", "contracts/vem_vision_v2"),
    (CONTRACT_ROOT / "python" / "__init__.py", "contracts/vem_vision_v2/python"),
    (CONTRACT_ROOT / "python" / "vision_v2_models.py", "contracts/vem_vision_v2/python"),
    (CONTRACT_ROOT / "vision-v2.client.schema.json", "contracts/vem_vision_v2"),
    (CONTRACT_ROOT / "vision-v2.server.schema.json", "contracts/vem_vision_v2"),
    (CONTRACT_ROOT / "fixtures" / "client-valid.json", "contracts/vem_vision_v2/fixtures"),
    (CONTRACT_ROOT / "fixtures" / "client-invalid.json", "contracts/vem_vision_v2/fixtures"),
    (CONTRACT_ROOT / "fixtures" / "server-valid.json", "contracts/vem_vision_v2/fixtures"),
    (CONTRACT_ROOT / "fixtures" / "server-invalid.json", "contracts/vem_vision_v2/fixtures"),
]
OFFICIAL_AI_SOURCE_DATA_FILES = []
for source in json.loads(OFFICIAL_AI_SOURCE_DESCRIPTOR_PATH.read_text("utf-8"))["sources"]:
    relative = source["path"]
    if relative.endswith(".py"):
        OFFICIAL_AI_SOURCE_DATA_FILES.append((str(ROOT / relative), str(Path(relative).parent)))

datas = [
    (str(ROOT / "config.json"), "."),
    (str(ROOT / "config"), "config"),
    (str(ROOT / "dashboard"), "dashboard"),
    (str(ROOT / "models"), "models"),
    (str(ROOT / "official-ai-model-pack-descriptor.json"), "."),
    (str(ROOT / "ai-runtime-descriptor.json"), "."),
    (str(ROOT / "requirements-ai.txt"), "."),
    (str(ROOT / "requirements-ai.lock.json"), "."),
    (str(ROOT / "official-ai-source-descriptor.json"), "."),
    *[(str(source), destination) for source, destination in CONTRACT_DATA_FILES],
    *OFFICIAL_AI_SOURCE_DATA_FILES,
]
datas += collect_data_files("mediapipe")
datas += collect_data_files("cv2_enumerate_cameras")

binaries = []
binaries += collect_dynamic_libs("mediapipe")
binaries += collect_dynamic_libs("cv2_enumerate_cameras")

hiddenimports = [
    "vision.v2_contract_bundle",
    "vision.render_worker_target",
    "vision.acquisition_observer",
    "vision.worker_self_check",
    "vision.ai_model_pack",
    "vision.ai_runtime_descriptor",
    "vision.ai_attempt_worker",
    "vision.ai_attempt_process",
    "vision.ai_acceptance_evidence",
    "vision.process_supervisor",
    "vision.catvton_pose_masks",
    "vision.catvton_preprocess",
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
# Delivery layout contract: build `vending_vision_ai_worker.spec` alongside
# this main runtime and place the resulting `vending-vision-ai-worker` onedir
# next to `vending-vision`.  The main runtime supervises that artifact-relative
# worker executable per probe/attempt; model weights stay outside both archives.
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
