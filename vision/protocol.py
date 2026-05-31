from datetime import datetime, timezone
from uuid import uuid4

from vision.config import settings


PROTOCOL = settings.PROTOCOL
APP_VERSION = settings.APP_VERSION


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def envelope(message_type: str, message_id: str | None = None, payload: dict | None = None):
    return {
        "protocol": PROTOCOL,
        "type": message_type,
        "messageId": message_id or str(uuid4()),
        "timestamp": now_iso(),
        "payload": payload or {}
    }


def error_envelope(
    code: str,
    message: str,
    session_id: str | None = None,
    retryable: bool = True,
    detail: dict | None = None,
    message_id: str | None = None
):
    payload = {
        "code": code,
        "message": message,
        "retryable": retryable,
    }

    if session_id is not None:
        payload["sessionId"] = session_id

    if detail:
        payload["detail"] = detail

    return envelope(
        message_type="vision.error",
        message_id=message_id,
        payload=payload
    )
