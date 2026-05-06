import logging
import sys

import structlog

# ---------------------------------------------------------------------------
# Why structlog instead of the stdlib logging module?
#
# stdlib logging produces lines like:
#   2024-01-15 14:28:05 ERROR root Connection timeout
#
# structlog produces:
#   {"timestamp":"2024-01-15T14:28:05Z","level":"error","event":"Connection timeout",
#    "request_id":"abc-123","agent":"critic","service":"payment-service"}
#
# The JSON format is directly queryable in every log aggregator (Datadog,
# CloudWatch, Loki, Grafana). You can filter by field, not grep by regex.
# In high-traffic production, this is the difference between finding a bug
# in 2 minutes vs 2 hours.
# ---------------------------------------------------------------------------


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for JSON output to stdout.

    Call this ONCE at app startup (in create_app or main.py).
    After this call, every module that does:
        log = structlog.get_logger(__name__)
        log.info("event", key=value, ...)
    will emit a structured JSON line to stdout.

    Why stdout?
    In containers (Docker, Kubernetes), stdout is captured by the container
    runtime and forwarded to your log aggregator automatically. Writing to
    files inside a container creates stale logs that nobody reads.
    """
    # Configure the stdlib root logger to forward to structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            # Add log level as a field
            structlog.stdlib.add_log_level,
            # Add ISO timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # If an exception is attached, format its traceback as a string
            structlog.processors.format_exc_info,
            # Render everything as a JSON string
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """
    Return a logger bound to a component name.

    Usage in any module:
        from src.observability.logger import get_logger
        log = get_logger(__name__)

        log.info("chunk_stored", chunk_id="abc", service="payment-service")
        log.warning("pool_near_limit", used=98, total=100)
        log.error("query_failed", error=str(exc), request_id=request_id)

    Each call emits one JSON line:
        {"level":"info","timestamp":"...","event":"chunk_stored",
         "chunk_id":"abc","service":"payment-service","logger":"src.ingestion.pipeline"}
    """
    return structlog.get_logger(name)
