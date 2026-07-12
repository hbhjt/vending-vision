# Packaging

The recommended Windows build is PyInstaller `onedir`.

```powershell
cd /d D:\ai-cv\vending_vision
powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
```

The executable is created at:

```text
dist\vending-vision\vending-vision.exe
```

The build uses PyInstaller `onedir`; distribute the complete
`dist\vending-vision` directory, not the EXE by itself.

Run the packaged contract smoke test after every build:

```powershell
.\.venv-packaging\Scripts\python.exe scripts\verify_packaged_exe.py
```

The smoke test starts the EXE in mock mode and checks bundled resources,
`/health`, `/version`, `/dashboard`, `/metrics`, WebSocket handshake/ping, and
the strict eight-field `vision.profile_result` contract. Real-camera capture,
rotation, try-on MJPEG, and long-session owner renewal still require the local
hardware acceptance flow.

Run it from that folder. By default it starts the same FastAPI service as:

```text
python -m uvicorn app:app --host 127.0.0.1 --port 7892
```

Runtime notes:

- The build script uses Python 3.9 because the pinned MediaPipe version is not reliably installable on Python 3.12.
- `requirements.txt` and `requirements-packaging.txt` intentionally use the same runtime pins. Keep them aligned when changing dependency versions.
- `config.json`, `dashboard/`, and `models/` are bundled into the PyInstaller output.
- Put an editable `config.json` next to `vending-vision.exe` if you want to override the bundled config.
- Set `VISION_WORKDIR` if logs and relative external files should live somewhere other than the exe directory.
