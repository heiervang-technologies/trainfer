"""Structured error envelope + closed code taxonomy for `/v1/*`.

Every `/v1/*` response that carries an error uses this envelope shape:

    {"error": {"code": "<stable-string>",
               "message": "<human-readable>",
               "retryable": <bool>,
               "request_id": "req_<hex16>"}}

`code` is drawn from `ERROR_CODES` — adding is fine, removing/renaming is a
breaking change for clients. `retryable` is a hint to clients (hard-coded
backoff on `True`); the server itself does not retry.

See issue #12 for the full rationale.
"""

from __future__ import annotations

from typing import Any

# Closed-enum taxonomy. Read by clients, referenced by tests. Add new codes
# here and in the table below; never remove.
ERROR_CODES: frozenset[str] = frozenset(
    {
        "invalid_input",
        "unknown_objective",
        "unknown_response_id",
        "not_found",
        "queue_full",
        "rate_limited",
        "batch_too_large",
        "shutting_down",
        "shutdown_dropped",
        "timeout",
        "internal",
    }
)


class TrainferError(Exception):
    """Base class for all structured errors surfaced on the wire.

    Subclasses override ``code``, ``status_code``, and ``retryable``. The
    message comes from ``Exception.__str__`` (i.e. the constructor arg).
    """

    code: str = "internal"
    status_code: int = 500
    retryable: bool = False
    retry_after_ms: int | None = None


class InvalidInputError(TrainferError):
    code = "invalid_input"
    status_code = 400
    retryable = False


class UnknownObjectiveError(TrainferError):
    code = "unknown_objective"
    status_code = 400
    retryable = False


class UnknownResponseIdError(TrainferError):
    code = "unknown_response_id"
    status_code = 404
    retryable = False


class NotFoundError(TrainferError):
    code = "not_found"
    status_code = 404
    retryable = False


class QueueFullError(TrainferError):
    code = "queue_full"
    status_code = 503
    retryable = True
    retry_after_ms = 500


class BatchTooLargeError(TrainferError):
    code = "batch_too_large"
    status_code = 413
    retryable = False


class RateLimitedError(TrainferError):
    code = "rate_limited"
    status_code = 429
    retryable = True


class ShuttingDownError(TrainferError):
    code = "shutting_down"
    status_code = 503
    retryable = True


class ShutdownDroppedError(TrainferError):
    code = "shutdown_dropped"
    status_code = 503
    retryable = True


class TimeoutError(TrainferError):  # noqa: A001 — intentional shadow within this module
    code = "timeout"
    status_code = 504
    retryable = True


def envelope_payload(
    *,
    code: str,
    message: str,
    request_id: str,
    retryable: bool = False,
    retry_after_ms: int | None = None,
) -> dict[str, Any]:
    """Return the JSON body that goes on the wire.

    The top-level key is always ``"error"`` so a valid 2xx response (without
    an "error" key) is unambiguous.
    """
    err = {
        "code": code,
        "message": message,
        "retryable": bool(retryable),
        "request_id": request_id,
    }
    if retry_after_ms is not None:
        err["retry_after_ms"] = retry_after_ms

    return {"error": err}
