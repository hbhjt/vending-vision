"""Attempt-scoped official AI child supervision with whole-tree termination."""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path


class AiAttemptProcess:
    def __init__(self, model_pack: Path):
        self._model_pack = model_pack
        self._process: subprocess.Popen | None = None

    async def probe(self, timeout: float = 10.0) -> None:
        if self._process is not None:
            raise RuntimeError("ai_attempt_child_already_running")
        kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP is paired with taskkill /T below.  The
            # Windows installed wrapper additionally places this group in its
            # Job Object; the worker is never a resident service.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self._process = subprocess.Popen([sys.executable, "-m", "vision.ai_attempt_worker", "--model-pack", str(self._model_pack), "--probe"], **kwargs)
        try:
            code = await asyncio.wait_for(asyncio.to_thread(self._process.wait), timeout)
            if code != 0:
                raise RuntimeError("official_ai_child_failed")
        finally:
            await self.close()

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            await asyncio.to_thread(subprocess.run, ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.to_thread(process.wait), 2.0)
        except asyncio.TimeoutError:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            await asyncio.to_thread(process.wait)
