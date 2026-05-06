import uuid

from src.ingestion.models import LogChunk, LogSeverity, RawLog
from src.ingestion.metadata import extract_metadata

# How many log lines per chunk and how many lines overlap between chunks.
# Overlap ensures events that span a chunk boundary are not split and lose context.
_WINDOW = 10
_OVERLAP = 3
_STEP = _WINDOW - _OVERLAP   # advance by 7 lines each iteration


def chunk_log(raw_log: RawLog, cleaned_content: str) -> list[LogChunk]:
    """
    Split cleaned log content into overlapping sliding windows.

    Each chunk is _WINDOW lines long. Consecutive chunks share _OVERLAP lines
    so that multi-line error sequences are never cut in half.

    Why sliding window for logs?
    A single "ERROR: connection refused" line has no meaning without the
    preceding "INFO: attempting DB connection to payments-db:5432" line.
    The overlap preserves that cause-and-effect context across chunk boundaries.
    """
    lines = [l for l in cleaned_content.split("\n") if l.strip()]
    if not lines:
        return []

    # One source_log_id groups all chunks that came from the same raw log.
    # This lets us later fetch "all chunks from this ingestion" by ID.
    source_log_id = str(uuid.uuid4())
    chunks: list[LogChunk] = []

    for start in range(0, len(lines), _STEP):
        window_lines = lines[start : start + _WINDOW]
        if not window_lines:
            break

        chunk_text = "\n".join(window_lines)
        meta = extract_metadata(chunk_text)

        chunk = LogChunk(
            chunk_id=str(uuid.uuid4()),
            source_log_id=source_log_id,
            service_name=raw_log.service_name,
            environment=raw_log.environment,
            severity=meta.get("severity", LogSeverity.INFO),
            timestamp=meta.get("timestamp") or raw_log.ingested_at,
            content=chunk_text,
            metadata={
                "trace_id":      meta.get("trace_id"),
                "deployment_id": meta.get("deployment_id"),
                "host":          meta.get("host"),
                "source":        raw_log.source,
                "line_start":    start,
                "line_end":      min(start + _WINDOW, len(lines)),
            },
        )
        chunks.append(chunk)

        # Last window — stop before going out of bounds
        if start + _WINDOW >= len(lines):
            break

    return chunks
