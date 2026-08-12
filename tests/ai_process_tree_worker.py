"""Test-only process tree used by public AI supervision integration tests."""
from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _wait_forever() -> None:
    while True:
        time.sleep(1)


def _terminate_and_wait(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _install_cleanup(process: subprocess.Popen | None = None) -> None:
    def stop(_signum, _frame):
        _terminate_and_wait(process)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def _child_command(
    role: str,
    mode: str,
    pid_file: Path,
    *,
    stress_mib: int,
    stress_seconds: float,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--role",
        role,
        "--mode",
        mode,
        "--pid-file",
        str(pid_file),
    ]
    if mode == "stress":
        command.extend(
            [
                "--stress-mib",
                str(stress_mib),
                "--stress-seconds",
                str(stress_seconds),
            ]
        )
    return command


def _run_bounded_pressure(*, stress_mib: int, stress_seconds: float) -> None:
    pressure = bytearray(stress_mib * 1024 * 1024)
    for offset in range(0, len(pressure), 4096):
        pressure[offset] = (offset // 4096) % 251
    deadline = time.monotonic() + stress_seconds
    accumulator = 1
    while time.monotonic() < deadline:
        for value in range(20_000):
            accumulator = (accumulator * 1_103_515_245 + value + 12_345) & 0xFFFFFFFF
        pressure[accumulator % len(pressure)] ^= accumulator & 0xFF


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role", choices=("leader", "child", "grandchild"), required=True
    )
    parser.add_argument(
        "--mode", choices=("crash", "block", "success", "stress"), required=True
    )
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--success-png", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--regional-evidence-output", type=Path)
    parser.add_argument("--person", type=Path)
    parser.add_argument("--garment", type=Path)
    parser.add_argument("--captured-source")
    parser.add_argument("--stress-mib", type=int, default=0)
    parser.add_argument("--stress-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.mode == "stress" and not (
        1 <= args.stress_mib <= 32 and 0.1 <= args.stress_seconds <= 30.0
    ):
        raise RuntimeError("stress mode requires bounded memory and duration")

    if args.role == "grandchild":
        _install_cleanup()
        if args.mode == "stress":
            _run_bounded_pressure(
                stress_mib=args.stress_mib,
                stress_seconds=args.stress_seconds,
            )
            return 0
        _wait_forever()

    if args.role == "child":
        grandchild = subprocess.Popen(
            _child_command(
                "grandchild",
                args.mode,
                args.pid_file,
                stress_mib=args.stress_mib,
                stress_seconds=args.stress_seconds,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _install_cleanup(grandchild)
        print(
            json.dumps(
                {"child": os.getpid(), "grandchild": grandchild.pid},
                sort_keys=True,
            ),
            flush=True,
        )
        if args.mode == "stress":
            _run_bounded_pressure(
                stress_mib=args.stress_mib,
                stress_seconds=args.stress_seconds,
            )
            _terminate_and_wait(grandchild)
            return 0
        _wait_forever()

    child = subprocess.Popen(
        _child_command(
            "child",
            args.mode,
            args.pid_file,
            stress_mib=args.stress_mib,
            stress_seconds=args.stress_seconds,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
    )
    _install_cleanup(child)
    assert child.stdout is not None
    descendants = json.loads(child.stdout.readline())
    tree = {"leader": os.getpid(), **descendants}
    temporary = args.pid_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(tree, sort_keys=True), "utf-8")
    os.replace(temporary, args.pid_file)
    if args.mode == "crash":
        return 17
    if args.mode == "success":
        if args.success_png is None or args.output is None:
            raise RuntimeError("success mode requires input and output paths")
        shutil.copyfile(args.success_png, args.output)
        if any(
            value is not None
            for value in (
                args.regional_evidence_output,
                args.person,
                args.garment,
                args.captured_source,
            )
        ):
            if not all(
                value is not None
                for value in (
                    args.regional_evidence_output,
                    args.person,
                    args.garment,
                    args.captured_source,
                )
            ):
                raise RuntimeError("regional evidence requires complete paths")
            with __import__("PIL.Image").Image.open(args.output) as image:
                width, height = image.size
            source = json.loads(args.captured_source)
            value = {
            "attempt": {
                "acquisitionSource": "direct_recorded_frame",
                "decodedHeight": height,
                "decodedWidth": width,
                "garmentSha256": __import__("hashlib").sha256(args.garment.read_bytes()).hexdigest(),
                "inputSha256": __import__("hashlib").sha256(args.person.read_bytes()).hexdigest(),
                "recordedFixtureSha256": source["fixtureSha256"],
                "resultSha256": __import__("hashlib").sha256(args.output.read_bytes()).hexdigest(),
                "sourceCamera": "front",
            },
            "evaluator": {},
            "kind": "regional-evidence",
            "masks": {},
            "measurements": {},
            "policy": {},
            "schemaVersion": "vem-ai-regional-evidence/v1",
            "verdict": "regional_check_failed",
            }
            args.regional_evidence_output.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", "utf-8"
            )
        time.sleep(0.1)
        _terminate_and_wait(child)
        return 0
    if args.mode == "stress":
        _run_bounded_pressure(
            stress_mib=args.stress_mib,
            stress_seconds=args.stress_seconds,
        )
        _terminate_and_wait(child)
        return 0
    _wait_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
