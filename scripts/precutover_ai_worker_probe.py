"""Run one packaged AI worker probe under the production tree supervisor."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from vision.process_supervisor import run_supervised


async def probe_worker(
    *, worker: Path, mode: str, model_pack: Path | None, timeout: float = 60.0
) -> dict[str, object]:
    if not worker.is_absolute() or worker.is_symlink() or not worker.is_file():
        raise RuntimeError("precutover_worker_identity")
    command = [str(worker)]
    if mode == "runtime":
        if model_pack is not None:
            raise RuntimeError("precutover_worker_arguments")
        command.append("--probe-runtime")
    elif mode == "model":
        if model_pack is None or not model_pack.is_absolute() or not model_pack.is_dir():
            raise RuntimeError("precutover_worker_arguments")
        command.extend(["--model-pack", str(model_pack), "--probe"])
    else:
        raise RuntimeError("precutover_worker_arguments")
    result = await run_supervised(command, timeout=timeout)
    if result.stdout_total > 64 * 1024 or result.stderr_total > 64 * 1024:
        raise RuntimeError("precutover_worker_output_limit")
    return {
        "returncode": result.returncode,
        "stderr": result.stderr_tail.decode("utf-8", errors="strict"),
        "stdout": result.stdout_tail.decode("utf-8", errors="strict"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--mode", choices=["runtime", "model"], required=True)
    parser.add_argument("--model-pack", type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    report = asyncio.run(
        probe_worker(
            worker=args.worker,
            mode=args.mode,
            model_pack=args.model_pack,
            timeout=args.timeout,
        )
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
