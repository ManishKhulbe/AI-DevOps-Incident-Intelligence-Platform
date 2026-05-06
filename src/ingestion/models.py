from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class LogSeverity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class RawLog(BaseModel):
    source: str           # "kubernetes", "json", "text"
    service_name: str
    environment: str      # "prod", "staging"
    content: str          # raw log string or JSON stringified
    ingested_at: datetime = datetime.utcnow()

class LogChunk(BaseModel):
    chunk_id: str         # UUID
    source_log_id: str
    service_name: str
    environment: str
    severity: LogSeverity
    timestamp: datetime
    content: str          # the actual chunk text
    metadata: dict        # trace_id, deployment_id, host, etc.