"""Helpers for readable console logging of structured request and response data."""
from __future__ import annotations

import json
import logging
from typing import Any


def log_payload(logger: logging.Logger, label: str, payload: Any) -> None:
    formatted = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )
    logger.info("%s:\n%s", label, formatted)
