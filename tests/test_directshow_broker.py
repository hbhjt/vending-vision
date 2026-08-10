import multiprocessing
import os
import time

import numpy as np
import pytest

from vision.directshow_broker import (
    DirectShowCameraBroker,
    directshow_broker_entry,
)


def _blocking_broker_target(connection, _config):
    try:
        command, _payload = connection.recv()
        if command == "read":
            while True:
                time.sleep(1)
    finally:
        connection.close()


def _happy_broker_target(connection, _config):
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            if command == "read":
                connection.send(
                    (
                        "ok",
                        {
                            "pid": os.getpid(),
                            "image": np.zeros((12, 16, 3), dtype=np.uint8),
                        },
                    )
                )
    finally:
        connection.close()


def _broker_config():
    return {
        "role": "profile_tryon",
        "index": 0,
        "backend": "dshow",
        "stableId": "test-front",
        "keep_open": True,
    }


def test_directshow_broker_target_is_spawn_importable_without_app_boundary():
    assert directshow_broker_entry.__module__ == "vision.directshow_broker"


def test_directshow_broker_deadline_kills_blocked_child_and_next_request_restarts():
    context = multiprocessing.get_context("spawn")
    broker = DirectShowCameraBroker(
        "front",
        _broker_config(),
        context=context,
        target=_blocking_broker_target,
    )

    with pytest.raises(TimeoutError):
        broker.read(warmup_frames=1, timeout=0.05)
    assert broker.assert_dead()

    restarted = DirectShowCameraBroker(
        "front",
        _broker_config(),
        context=context,
        target=_happy_broker_target,
    )
    try:
        image = restarted.read(warmup_frames=1, timeout=1.0)
        assert image.shape == (12, 16, 3)
        assert restarted.pid is not None
    finally:
        restarted.release()
    assert restarted.assert_dead()
