import re
from datetime import datetime

from src.ingestion.models import LogSeverity

# --- Timestamp patterns ---
# Ordered most-specific to least-specific so the first match wins.
# Each tuple is (regex_pattern, strptime_format).
_TIMESTAMP_PATTERNS: list[tuple[str, str]] = [
    (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%S"),   # ISO 8601
    (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S"),    # common log format
    (r"\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}", "%d/%b/%Y:%H:%M:%S"),    # nginx/apache
]

# --- Severity ---
_SEVERITY_RE = re.compile(
    r"\b(DEBUG|INFO|NOTICE|WARN(?:ING)?|ERROR|CRITICAL|FATAL|EMERGENCY)\b",
    re.IGNORECASE,
)
_SEVERITY_MAP: dict[str, LogSeverity] = {
    "DEBUG":     LogSeverity.DEBUG,
    "INFO":      LogSeverity.INFO,
    "NOTICE":    LogSeverity.INFO,
    "WARN":      LogSeverity.WARNING,
    "WARNING":   LogSeverity.WARNING,
    "ERROR":     LogSeverity.ERROR,
    "CRITICAL":  LogSeverity.CRITICAL,
    "FATAL":     LogSeverity.CRITICAL,
    "EMERGENCY": LogSeverity.CRITICAL,
}

# --- Other structured fields ---
# Using non-capturing groups around the key aliases keeps each regex short.
_TRACE_RE = re.compile(
    r"(?:trace[_-]?id|request[_-]?id|x[_-]request[_-]id|correlation[_-]?id)"
    r"[=:\s]+([a-zA-Z0-9\-_]+)",
    re.IGNORECASE,
)
_DEPLOY_RE = re.compile(
    r"(?:deploy[_-]?id|deployment|git[_-]?sha|version|release)"
    r"[=:\s]+([a-zA-Z0-9\-_.]+)",
    re.IGNORECASE,
)
_HOST_RE = re.compile(
    r"(?:host|hostname|node|pod)[=:\s]+([a-zA-Z0-9\-_.]+)",
    re.IGNORECASE,
)


def extract_metadata(text: str) -> dict:
    """
    Extract structured metadata from a chunk of log text.

    Returns a dict with keys:
        timestamp      datetime | None
        severity       LogSeverity
        trace_id       str | None
        deployment_id  str | None
        host           str | None
    """
    return {
        "timestamp":     _extract_timestamp(text),
        "severity":      _extract_severity(text),
        "trace_id":      _first_group(_TRACE_RE, text),
        "deployment_id": _first_group(_DEPLOY_RE, text),
        "host":          _first_group(_HOST_RE, text),
    }


def _extract_timestamp(text: str) -> datetime | None:
    for pattern, fmt in _TIMESTAMP_PATTERNS:
        match = re.search(pattern, text)
        if not match:
            continue
        # Strip sub-seconds and timezone suffix before parsing
        raw = match.group(0).split(".")[0].rstrip("Z")
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _extract_severity(text: str) -> LogSeverity:
    match = _SEVERITY_RE.search(text)
    if match:
        return _SEVERITY_MAP.get(match.group(0).upper(), LogSeverity.INFO)
    return LogSeverity.INFO


def _first_group(pattern: re.Pattern, text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None
