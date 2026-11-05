import logging
import os
from contextlib import contextmanager
from typing import Iterable, Optional

import requests
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from retry_utils import _chain, _response_of

logger = logging.getLogger(__name__)

_CONNECTION_ERROR_NAMES = {"APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "ConnectTimeout",
                           "OperationalError", "InterfaceError", "PoolError"}


class ApiError(Exception):
    """Typed API error rendered as {error, stage, job_id?, detail}."""

    def __init__(self, status_code: int, error: str, detail: str, stage: Optional[str] = None, job_id: Optional[str] = None):
        super().__init__(f"{error}: {detail}")
        self.status_code = status_code
        self.error = error
        self.detail = detail
        self.stage = stage
        self.job_id = job_id

    def to_body(self) -> dict:
        body = {"error": self.error, "stage": self.stage, "detail": self.detail}
        if self.job_id:
            body["job_id"] = self.job_id
        return body

    def summary(self) -> str:
        return f"{self.error} (stage={self.stage}): {self.detail}"


def api_error_response(_request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_body())


def validation_error_response(_request, exc) -> JSONResponse:
    detail = "; ".join(
        f"{'.'.join(str(p) for p in err.get('loc', []))}: {err.get('msg')}" for err in exc.errors()
    )
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_request", "stage": "validating_request", "detail": detail},
    )


def require_env(names: Iterable[str], stage: str, job_id: Optional[str] = None):
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise ApiError(400, "missing_config", f"Missing required environment variable(s): {', '.join(missing)}", stage, job_id)


def classify_exception(exc: BaseException, stage: str, source: Optional[str] = None, job_id: Optional[str] = None) -> ApiError:
    """
    Map an arbitrary exception to an ApiError.
    source: 'omniparser' | 'openai' | 'storage' | 'database' | None - the dependency being called.
    """
    if isinstance(exc, ApiError):
        exc.stage = exc.stage or stage
        exc.job_id = exc.job_id or job_id
        return exc

    label = source or "internal"
    detail = str(exc) or type(exc).__name__
    for e in _chain(exc):
        if isinstance(e, StarletteHTTPException):
            continue  # generic wrapper; the real cause is further down the chain
        if isinstance(e, (requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError)) \
                or type(e).__name__ in _CONNECTION_ERROR_NAMES:
            return ApiError(503, f"{label}_unavailable", f"{label} dependency unreachable: {e}", stage, job_id)
        response = _response_of(e)
        status = getattr(e, "status_code", None) or (getattr(response, "status_code", None) if response is not None else None)
        if isinstance(status, int):
            return ApiError(502, f"upstream_{label}_error", f"{label} returned HTTP {status}: {e}", stage, job_id)

    if source == "database":
        return ApiError(503, "database_error", f"Database operation failed: {detail}", stage, job_id)
    if source in ("omniparser", "openai", "storage"):
        return ApiError(502, f"upstream_{label}_error", detail, stage, job_id)
    return ApiError(500, "internal_error", detail, stage, job_id)


@contextmanager
def api_errors(stage: str, source: Optional[str] = None, job_id: Optional[str] = None):
    """Convert any exception raised inside the block into an ApiError."""
    try:
        yield
    except ApiError as e:
        e.stage = e.stage or stage
        e.job_id = e.job_id or job_id
        raise
    except Exception as e:
        err = classify_exception(e, stage, source, job_id)
        logger.error(f"[{stage}] {err.summary()}")
        raise err from e
