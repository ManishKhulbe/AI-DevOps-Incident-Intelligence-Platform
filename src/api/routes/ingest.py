from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.ingestion.models import RawLog
from src.ingestion.pipeline import ingest

router = APIRouter(tags=["ingestion"])


# ── Request / Response models ──────────────────────────────────────────────────
# These are separate from the internal RawLog model intentionally.
# The API contract (what the caller sends) and the internal model (how we
# process it) should be decoupled — a future API change won't break internals.

class IngestRequest(BaseModel):
    source:       str = Field(..., description="Log format: 'text', 'json', or 'kubernetes'")
    service_name: str = Field(..., description="Name of the service that produced these logs")
    environment:  str = Field("prod", description="Environment: 'prod', 'staging', 'dev'")
    content:      str = Field(..., description="Raw log content as a string")

    model_config = {
        "json_schema_extra": {
            "example": {
                "source": "text",
                "service_name": "payment-service",
                "environment": "prod",
                "content": "2024-01-15T14:28:05 ERROR payment-service Connection timeout trace_id=abc-001",
            }
        }
    }


class IngestResponse(BaseModel):
    log_id:        str
    chunks_stored: int
    request_id:    str
    message:       str


# ── Route ──────────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest logs",
    description="Clean, chunk, embed, and store log content in Qdrant + Elasticsearch.",
)
async def ingest_logs(body: IngestRequest, request: Request) -> IngestResponse:
    """
    FR-1: Log Upload — receive logs and run the full write-path pipeline.

    Flow inside this handler:
        IngestRequest → RawLog → pipeline.ingest() → dual-write → IngestResponse

    Why async?
    pipeline.ingest() calls qdrant_store and es_store which both do network I/O.
    Async lets FastAPI handle other incoming requests while waiting for those
    writes to complete, instead of blocking the entire server.

    Error handling:
    We let pipeline.ingest() exceptions bubble up here and convert them to
    HTTP 500 with a clear message. The caller can retry — both stores use
    chunk_id as the document key, so re-ingesting the same log is idempotent.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        raw_log = RawLog(
            source=body.source,
            service_name=body.service_name,
            environment=body.environment,
            content=body.content,
        )
        result = await ingest(raw_log)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    return IngestResponse(
        log_id=result["log_id"],
        chunks_stored=result["chunks_stored"],
        request_id=request_id,
        message=f"Successfully stored {result['chunks_stored']} chunks.",
    )
