"""Spawn-safe broker children for v2 fast-attempt tests.

Keep this module free of application imports: Windows ``spawn`` imports the
target module afresh before it can enter a test-controlled synchronization
barrier.
"""

import base64
import os
import threading
import time


_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAACQAAAAwCAYAAAB5R9gVAAAAuUlEQVRYCc3BAQEAAAABID4Ya6wv3FCMVRxhrOIIYxVHGKs4wljFEcYqjjBWcYSxiiOMVRxhrOIIYxVHGKs4wljFEcYqjjBWcYSxiiOMVRxhrOIIYxVHGKs4wljFEcYqjjBWcYSxiiOMVRxhrOIIYxVHGKs4wljFEcYqjjBWcYSxiiOMVRxhrOIIYxVHGKs4wljFEcYqjjBWcYSxiiOMVRxhrOIIYxVHGKs4wljFEcYqjjBWcYSxiiMDIg5zgWq0meIAAAAASUVORK5CYII="
)


def block_first_render(connection, counter):
    connection.send(("ready", {"pid": os.getpid()}))
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            with counter.get_lock():
                counter.value += 1
                request_number = counter.value
            if request_number == 1:
                while True:
                    threading.Event().wait(1.0)
            connection.send(("ok", _PNG_BYTES))
    finally:
        connection.close()


def block_then_barrier_restart_render(
    connection, starts, requests, restart_entered, restart_release, restart_fails
):
    with starts.get_lock():
        starts.value += 1
        start_number = starts.value
    if start_number > 1:
        restart_entered.set()
        restart_release.wait()
        if restart_fails:
            connection.close()
            return
    connection.send(("ready", {"pid": os.getpid()}))
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            with requests.get_lock():
                requests.value += 1
                request_number = requests.value
            if request_number == 1:
                while True:
                    threading.Event().wait(1.0)
            connection.send(("ok", _PNG_BYTES))
    finally:
        connection.close()


def block_then_ready_barrier_directshow(connection, config):
    """Block the initial read, then pause replacement before its ready reply."""
    starts = config["starts"]
    blocked_read_entered = config["blockedReadEntered"]
    restart_ready_entered = config["restartReadyEntered"]
    restart_ready_release = config["restartReadyRelease"]
    with starts.get_lock():
        starts.value += 1
        start_number = starts.value
    if start_number == 2:
        restart_ready_entered.value = 1
        while not restart_ready_release.value:
            time.sleep(0.01)
    try:
        connection.send(("ready", {"pid": os.getpid()}))
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            if command == "read":
                if start_number == 1:
                    blocked_read_entered.value = 1
                    while True:
                        threading.Event().wait(1.0)
                import numpy as np

                connection.send(
                    (
                        "ok",
                        {
                            "pid": os.getpid(),
                            "image": np.full(
                                (80, 60, 3), (235, 220, 205), dtype=np.uint8
                            ),
                        },
                    )
                )
    finally:
        connection.close()
