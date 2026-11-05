import asyncio
import logging
import os
import random
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional, TypeVar

import requests
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS = lambda status: status == 429 or 500 <= status <= 599
_CONNECTION_ERROR_NAMES = {"APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "ConnectTimeout"}


def _chain(exc: BaseException):
    """Yield exc and its __cause__/__context__ ancestors (wrappers hide the real error)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _response_of(exc: BaseException):
    response = getattr(exc, "response", None)
    return response if response is not None and hasattr(response, "headers") else None


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def classify(exc: BaseException):
    """Return (retryable, retry_after_seconds) for an exception, inspecting wrapped causes."""
    for e in _chain(exc):
        if isinstance(e, StarletteHTTPException):
            continue  # wrapper that hides the real upstream error; inspect its cause instead
        response = _response_of(e)
        status = getattr(e, "status_code", None) or (getattr(response, "status_code", None) if response is not None else None)
        if isinstance(status, int):
            retry_after = _parse_retry_after(response.headers.get("Retry-After")) if response is not None else None
            return RETRYABLE_STATUS(status), retry_after
        if isinstance(e, (requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError, asyncio.TimeoutError)):
            return True, None
        if type(e).__name__ in _CONNECTION_ERROR_NAMES:
            return True, None
    return False, None


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    description: str = "call",
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
) -> T:
    """
    Await func() with exponential backoff. Retries only on 429/5xx/connection errors and
    honors Retry-After. Env: RETRY_MAX_ATTEMPTS (4), RETRY_BASE_DELAY seconds (1.0), RETRY_MAX_DELAY seconds (30).
    """
    max_attempts = max_attempts or int(os.getenv("RETRY_MAX_ATTEMPTS", "4"))
    base_delay = base_delay if base_delay is not None else float(os.getenv("RETRY_BASE_DELAY", "1.0"))
    max_delay = max_delay if max_delay is not None else float(os.getenv("RETRY_MAX_DELAY", "30"))

    attempt = 1
    while True:
        try:
            return await func()
        except Exception as exc:
            retryable, retry_after = classify(exc)
            if not retryable or attempt >= max_attempts:
                raise
            delay = min(max_delay, retry_after if retry_after is not None else base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, 0.1 * max(delay, base_delay))
            logger.warning(f"{description} failed (attempt {attempt}/{max_attempts}): {exc!r}; retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
            attempt += 1
