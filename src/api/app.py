import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import RequestIDMiddleware
from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router
from src.observability.logger import get_logger, setup_logging

log = get_logger(__name__)


def _configure_langsmith() -> None:
    """
    Enable LangSmith tracing by setting the required environment variables.

    LangSmith traces every LangChain/LangGraph call automatically — no code
    changes needed in agents. When enabled, every graph.invoke() produces a
    full trace in the LangSmith dashboard showing:
      - Which agent ran, in what order
      - Exact prompt sent to the LLM
      - Exact response received
      - Token count and latency per call
      - The retry loop iterations

    How to enable:
      Set in your .env file:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=ls__...
        LANGCHAIN_PROJECT=devops-incident-agent

    Why configure here instead of in .env directly?
    pydantic-settings reads the values, but LangSmith reads them via os.environ.
    We bridge the two by setting os.environ from our validated settings object.
    This ensures the values are validated (not silently missing) before we
    attempt to activate tracing.
    """
    from src.config import settings

    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"]  = "true"
        os.environ["LANGCHAIN_API_KEY"]     = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"]     = settings.langchain_project
        log.info("langsmith_enabled", project=settings.langchain_project)
    else:
        log.info("langsmith_disabled", reason="LANGCHAIN_TRACING_V2 or LANGCHAIN_API_KEY not set")


def create_app() -> FastAPI:
    """
    FastAPI app factory.

    Startup order:
    1. Setup structured logging (must be first — all subsequent code logs)
    2. Configure LangSmith tracing
    3. Register middleware (outermost = last registered = RequestID)
    4. Include routers
    """
    setup_logging()
    _configure_langsmith()

    log.info("app_starting", version="0.1.0")

    app = FastAPI(
        title="DevOps Incident Intelligence Platform",
        description=(
            "AI-powered log analysis. "
            "Ingest: POST /api/v1/ingest. "
            "Query: POST /api/v1/query or WS /api/v1/query/stream."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware — applied in reverse registration order.
    # RequestIDMiddleware must be last registered so it runs outermost,
    # meaning every handler and middleware underneath it can access request_id.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    app.include_router(ingest_router, prefix="/api/v1")
    app.include_router(query_router,  prefix="/api/v1")

    @app.get("/health", tags=["system"])
    async def health():
        """Liveness probe — confirms the process is alive."""
        return {"status": "ok"}

    log.info("app_ready", docs="/docs")
    return app
