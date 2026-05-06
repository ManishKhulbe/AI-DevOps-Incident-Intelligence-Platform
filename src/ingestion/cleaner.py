import re
import json

# Matches all ANSI terminal escape sequences (colors, cursor moves, etc.)
# Without this, log files from Docker/K8s will contain garbled characters.
_ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def clean_log(content: str, source: str) -> str:
    """
    Entry point. Dispatches to the right cleaner based on source format.

    Why separate cleaners per source?
    JSON logs and plaintext logs have different noise. JSON needs to be
    flattened to text first; plaintext just needs sanitization.
    """
    if source == "json":
        return _clean_json_log(content)
    return _clean_text_log(content)


def _clean_text_log(content: str) -> str:
    content = _ANSI_ESCAPE.sub("", content)        # strip terminal colors
    content = content.replace("\x00", "")           # strip null bytes
    content = content.replace("\r\n", "\n").replace("\r", "\n")  # normalize line endings

    lines = [line.rstrip() for line in content.split("\n")]
    lines = [line for line in lines if line.strip()]   # drop blank lines
    return "\n".join(lines)


def _clean_json_log(content: str) -> str:
    """
    Convert a JSON log (single object or array) into readable plaintext lines.
    Falls back to text cleaning if the JSON cannot be parsed.
    """
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return "\n".join(_entry_to_line(e) for e in data if isinstance(e, dict))
        if isinstance(data, dict):
            return _entry_to_line(data)
    except json.JSONDecodeError:
        pass
    return _clean_text_log(content)


def _entry_to_line(entry: dict) -> str:
    """
    Flatten one JSON log entry into a single human-readable line.

    Different logging libraries use different field names for the same concepts,
    so we check the most common aliases in order.
    """
    timestamp = entry.get("timestamp") or entry.get("time") or entry.get("@timestamp", "")
    level     = entry.get("level") or entry.get("severity") or entry.get("loglevel", "")
    service   = entry.get("service") or entry.get("app") or entry.get("container", "")
    message   = entry.get("message") or entry.get("msg") or entry.get("log", "")

    parts = [p for p in [timestamp, level.upper() if level else "", service, message] if p]
    return " ".join(parts)
