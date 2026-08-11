"""Test-only process tree used by public AI supervision integration tests."""
from __future__ import annotations

import argparse
import json
import os
import signal
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


def _child_command(role: str, mode: str, pid_file: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--role",
        role,
        "--mode",
        mode,
        "--pid-file",
        str(pid_file),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role", choices=("leader", "child", "grandchild"), required=True
    )
    parser.add_argument("--mode", choices=("crash", "block"), required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    args = parser.parse_args()

    if args.role == "grandchild":
        _install_cleanup()
        _wait_forever()

    if args.role == "child":
        grandchild = subprocess.Popen(
            _child_command("grandchild", args.mode, args.pid_file),
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
        _wait_forever()

    child = subprocess.Popen(
        _child_command("child", args.mode, args.pid_file),
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
    _wait_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
