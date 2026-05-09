from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from src.ingestion.models import RawLog
from src.ingestion.pipeline import ingest
from src.observability.logger import get_logger

router = APIRouter(tags=["ingestion"])
log = get_logger(__name__)


# ── Request / Response models ──────────────────────────────────────────────────

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
    """
    request_id = getattr(request.state, "request_id", "unknown")

    log.info(
        "ingest_started",
        request_id=request_id,
        service_name=body.service_name,
        environment=body.environment,
        source=body.source,
        content_length=len(body.content),
    )

    try:
        raw_log = RawLog(
            source=body.source,
            service_name=body.service_name,
            environment=body.environment,
            content=body.content,
        )
        result = await ingest(raw_log)
    except Exception as exc:
        log.error(
            "ingest_failed",
            request_id=request_id,
            service_name=body.service_name,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    log.info(
        "ingest_completed",
        request_id=request_id,
        log_id=result["log_id"],
        chunks_stored=result["chunks_stored"],
        service_name=body.service_name,
    )

    return IngestResponse(
        log_id=result["log_id"],
        chunks_stored=result["chunks_stored"],
        request_id=request_id,
        message=f"Successfully stored {result['chunks_stored']} chunks.",
    )


# ── File Upload Route ──────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".log", ".txt"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest log file",
    description="Upload a .log or .txt file directly. Reads the file content and runs the full ingest pipeline.",
)
async def ingest_file(
    request: Request,
    file: UploadFile = File(..., description="A .log or .txt file"),
    source: str = Form("text", description="Log format: 'text', 'json', or 'kubernetes'"),
    service_name: str = Form(..., description="Name of the service that produced these logs"),
    environment: str = Form("prod", description="Environment: 'prod', 'staging', 'dev'"),
) -> IngestResponse:
    request_id = getattr(request.state, "request_id", "unknown")

    # Validate file extension
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported. Got: '{ext or 'no extension'}'",
        )

    # Read file with size guard
    content_bytes = await file.read(MAX_FILE_SIZE + 1)
    if len(content_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the 500 MB limit.",
        )

    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File is not valid UTF-8 text.",
        )

    log.info(
        "ingest_file_started",
        request_id=request_id,
        filename=filename,
        service_name=service_name,
        environment=environment,
        source=source,
        file_size_bytes=len(content_bytes),
    )

    try:
        raw_log = RawLog(
            source=source,
            service_name=service_name,
            environment=environment,
            content=content,
        )
        result = await ingest(raw_log)
    except Exception as exc:
        log.error(
            "ingest_file_failed",
            request_id=request_id,
            filename=filename,
            service_name=service_name,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    log.info(
        "ingest_file_completed",
        request_id=request_id,
        filename=filename,
        log_id=result["log_id"],
        chunks_stored=result["chunks_stored"],
    )

    return IngestResponse(
        log_id=result["log_id"],
        chunks_stored=result["chunks_stored"],
        request_id=request_id,
        message=f"Successfully stored {result['chunks_stored']} chunks from '{filename}'.",
    )
