import uuid
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Inject a unique request ID into every request and response.

    Why is this important in production?
    When 100 requests are in-flight simultaneously, logs from different
    requests are interleaved. Without a request ID, you cannot tell which
    log line belongs to which request.

    With a request ID, you can grep your log aggregator:
        {"request_id": "abc-123", "agent": "critic", "message": "..."}
    and see the full trace of ONE request across all agents.

    The ID is:
    - Stored on request.state so route handlers can access it
    - Added to the response header so clients can log it
    - Short enough to be scannable (first 8 chars of UUID)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)

        return response
