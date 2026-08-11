"""Attempt-scoped official AI child supervision with whole-tree termination."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from vision.ai_runtime_descriptor import expected_dependency_versions
from vision.process_supervisor import ProcessSupervisorError, run_supervised


def ai_worker_executable_path() -> Path:
    """Return the artifact-relative frozen worker executable path."""
    executable = Path(sys.executable).resolve()
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [
        executable.with_name(f"vending-vision-ai-worker{suffix}"),
        executable.parent / "vending-vision-ai-worker" / f"vending-vision-ai-worker{suffix}",
        executable.parent.parent / "vending-vision-ai-worker" / f"vending-vision-ai-worker{suffix}",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("official_ai_worker_executable_missing")


def ai_attempt_worker_command(
    model_pack: Path,
    *,
    probe: bool = False,
    person_png: Path | None = None,
    garment_png: Path | None = None,
    output_png: Path | None = None,
    template: str = "tshirt_short_sleeve",
) -> list[str]:
    if getattr(sys, "frozen", False):
        command = [str(ai_worker_executable_path())]
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
                "--template",
                template,
            ]
        )
    return command


def probe_ai_attempt_worker(model_pack: Path, *, timeout: float = 30.0) -> None:
    result = asyncio.run(
        run_supervised(ai_attempt_worker_command(model_pack, probe=True), timeout=timeout)
    )
    if result.returncode != 0:
        raise RuntimeError("official_ai_child_probe_failed")
    try:
        payload = json.loads(result.stdout_tail.decode("utf-8").strip().splitlines()[-1])
    except (IndexError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("official_ai_child_probe_failed") from exc
    expected = expected_dependency_versions()
    for name in ("torch", "torchvision", "diffusers", "transformers"):
        if payload.get(name) != expected[name]:
            raise RuntimeError("official_ai_child_probe_failed")


class AiAttemptProcess:
    def __init__(self, model_pack: Path):
        self._model_pack = model_pack
        self._running = False

    async def probe(self, timeout: float = 10.0) -> None:
        if self._running:
            raise RuntimeError("ai_attempt_child_already_running")
        self._running = True
        try:
            result = await run_supervised(
                ai_attempt_worker_command(self._model_pack, probe=True),
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError("official_ai_child_failed")
        except ProcessSupervisorError as exc:
            raise RuntimeError("official_ai_child_failed") from exc
        finally:
            self._running = False

    async def run(
        self,
        *,
        person_png: Path,
        garment_png: Path,
        output_png: Path,
        timeout: float,
        template: str = "tshirt_short_sleeve",
    ) -> None:
        if self._running:
            raise RuntimeError("ai_attempt_child_already_running")
        self._running = True
        try:
            result = await run_supervised(
                ai_attempt_worker_command(
                    self._model_pack,
                    person_png=person_png,
                    garment_png=garment_png,
                    output_png=output_png,
                    template=template,
                ),
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError("official_ai_child_failed")
            if not output_png.is_file():
                raise RuntimeError("official_ai_child_missing_output")
        except ProcessSupervisorError as exc:
            raise RuntimeError("official_ai_child_failed") from exc
        finally:
            self._running = False

    async def close(self) -> None:
        self._running = False


def windows_ai_child_creation_flags(subprocess_module) -> int:
    return (
        subprocess_module.CREATE_NEW_PROCESS_GROUP
        | subprocess_module.BELOW_NORMAL_PRIORITY_CLASS
    )
