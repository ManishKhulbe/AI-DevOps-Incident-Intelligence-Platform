import uuid
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.observability.logger import get_logger

log = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Inject a unique request ID into every request and response.

    Why is this important in production?
    When 100 requests are in-flight simultaneously, logs from different
    requests are interleaved. Without a request ID, you cannot tell which
    log line belongs to which request.

    With a request ID, you can filter your log aggregator:
        request_id = "abc-123"  →  see every log line for that one request,
        across all 6 agents, in order.

    The ID is stored on request.state so route handlers and agents can
    attach it to their own log lines.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)

        log.info(
            "request_handled",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        return response
