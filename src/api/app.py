from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import RequestIDMiddleware
from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router


def create_app() -> FastAPI:
    """
    FastAPI app factory.

    Why a factory function instead of a module-level app = FastAPI()?
    A factory lets you create separate app instances in tests with different
    config, without sharing global state between test runs. It also makes
    it explicit that creating the app is a deliberate action, not a side
    effect of importing the module.
    """
    app = FastAPI(
        title="DevOps Incident Intelligence Platform",
        description=(
            "AI-powered log analysis platform. "
            "Ingest logs via POST /api/v1/ingest. "
            "Query incidents via POST /api/v1/query or WS /api/v1/query/stream."
        ),
        version="0.1.0",
        docs_url="/docs",       # Swagger UI at /docs
        redoc_url="/redoc",     # ReDoc UI at /redoc
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    # Middleware is applied in REVERSE registration order.
    # RequestID must be outermost so every subsequent middleware and handler
    # can access request.state.request_id.

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],      # Tighten this in production
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    # ── Routes ────────────────────────────────────────────────────────────────
    # All routes are prefixed with /api/v1.
    # Versioning in the URL path means we can add /api/v2 routes later
    # without breaking existing clients that still call /api/v1.

    app.include_router(ingest_router, prefix="/api/v1")
    app.include_router(query_router,  prefix="/api/v1")

    # ── Health check ──────────────────────────────────────────────────────────

    @app.get("/health", tags=["system"])
    async def health():
        """
        Liveness probe — confirms the process is running.
        Does not check Elasticsearch/Qdrant connectivity.
        Add a /ready endpoint for a full dependency check before production use.
        """
        return {"status": "ok"}

    return app
