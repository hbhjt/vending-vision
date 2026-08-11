"""Attempt-scoped official AI child supervision with whole-tree termination."""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path


def ai_attempt_worker_command(
    model_pack: Path,
    *,
    probe: bool = False,
    person_png: Path | None = None,
    garment_png: Path | None = None,
    output_png: Path | None = None,
) -> list[str]:
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--ai-attempt-worker"]
    else:
        command = [sys.executable, "-m", "vision.ai_attempt_worker"]
    command.extend(["--model-pack", str(model_pack)])
    if probe:
        command.append("--probe")
    else:
        if person_png is None or garment_png is None or output_png is None:
            raise RuntimeError("official_ai_child_attempt_paths_required")
        command.extend(
            [
                "--person",
                str(person_png),
                "--garment",
                str(garment_png),
                "--output",
                str(output_png),
            ]
        )
    return command


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
            # Job Object; the worker is never a resident service and runs
            # below the core Vision runtime priority.
            kwargs["creationflags"] = windows_ai_child_creation_flags(subprocess)
        else:
            kwargs["start_new_session"] = True
        self._process = subprocess.Popen(
            ai_attempt_worker_command(self._model_pack, probe=True),
            **kwargs,
        )
        try:
            code = await asyncio.wait_for(asyncio.to_thread(self._process.wait), timeout)
            if code != 0:
                raise RuntimeError("official_ai_child_failed")
        finally:
            await self.close()

    async def run(
        self,
        *,
        person_png: Path,
        garment_png: Path,
        output_png: Path,
        timeout: float,
    ) -> None:
        if self._process is not None:
            raise RuntimeError("ai_attempt_child_already_running")
        kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
        if os.name == "nt":
            kwargs["creationflags"] = windows_ai_child_creation_flags(subprocess)
        else:
            kwargs["start_new_session"] = True
        self._process = subprocess.Popen(
            ai_attempt_worker_command(
                self._model_pack,
                person_png=person_png,
                garment_png=garment_png,
                output_png=output_png,
            ),
            **kwargs,
        )
        try:
            code = await asyncio.wait_for(asyncio.to_thread(self._process.wait), timeout)
            if code != 0:
                raise RuntimeError("official_ai_child_failed")
            if not output_png.is_file():
                raise RuntimeError("official_ai_child_missing_output")
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


def windows_ai_child_creation_flags(subprocess_module) -> int:
    return (
        subprocess_module.CREATE_NEW_PROCESS_GROUP
        | subprocess_module.BELOW_NORMAL_PRIORITY_CLASS
    )
