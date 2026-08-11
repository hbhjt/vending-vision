"""Frozen official AI worker launcher.

This entrypoint is packaged as `vending-vision-ai-worker.exe`.  It is not a
service, camera owner, or resident process; the main Vision runtime supervises
it per attempt/probe.
"""
from __future__ import annotations

from vision.ai_attempt_worker import main


if __name__ == "__main__":
    raise SystemExit(main())
