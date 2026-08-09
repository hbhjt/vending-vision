"""Static V2 contract boundary import retained in frozen Vision artifacts."""

from typing import Any

from contracts.vem_vision_v2.python.vision_v2_models import (
    VisionV2Envelope,
    parse_message,
)


def parse_v2_boundary_message(value: Any) -> VisionV2Envelope:
    """Parse one generated V2 boundary message through the vendored bundle."""
    return parse_message(value)
