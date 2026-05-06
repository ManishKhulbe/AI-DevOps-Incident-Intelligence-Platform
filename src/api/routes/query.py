import asyncio
import json

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from src.agents.graph import agent_graph
from src.agents.state import AgentState
from src.observability.logger import get_logger

router = APIRouter(tags=["query"])
log = get_logger(__name__)


# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., description="Plain-English incident question")

    model_config = {
        "json_schema_extra": {
            "example": {"question": "Why did the payment service fail at 2:28 PM?"}
        }
    }


class CitationOut(BaseModel):
    claim:      str
    evidence:   str
    chunk_id:   str
    confidence: float


class QueryResponse(BaseModel):
    question:       str
    final_response: str
    citations:      list[CitationOut]
    request_id:     str


# ── Helper ─────────────────────────────────────────────────────────────────────

def _build_initial_state(question: str) -> AgentState:
    return AgentState(
        user_query=question,
        query_plan={},
        retrieved_chunks=[],
        reasoning_output="",
        critic_feedback={},
        retry_count=0,
        additional_queries=[],
        citations=[],
        final_response="",
    )


# ── POST /query — full blocking response ──────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Incident query",
    description=(
        "Run the full 6-agent pipeline and return the complete RCA with citations."
    ),
)
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    """
    FR-2 + FR-3: Natural language incident query with AI-generated RCA.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    log.info(
        "query_started",
        request_id=request_id,
        question=body.question,
    )

    try:
        initial_state = _build_initial_state(body.question)
        result: AgentState = await asyncio.to_thread(agent_graph.invoke, initial_state)
    except Exception as exc:
        log.error(
            "query_failed",
            request_id=request_id,
            question=body.question,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent pipeline failed: {exc}",
        )

    citations = result.get("citations", [])
    retry_count = result.get("retry_count", 0)
    confidence = result.get("critic_feedback", {}).get("confidence", 1.0)

    log.info(
        "query_completed",
        request_id=request_id,
        citations_count=len(citations),
        retry_count=retry_count,
        confidence=confidence,
    )

    return QueryResponse(
        question=body.question,
        final_response=result["final_response"],
        citations=[CitationOut(**c) for c in citations],
        request_id=request_id,
    )


# ── WebSocket /query/stream — agent progress + final response ─────────────────

@router.websocket("/query/stream")
async def query_stream(websocket: WebSocket) -> None:
    """
    Stream agent progress events to the client node-by-node, then send the
    final response when the graph completes.

    Protocol:
        Client → {"question": "Why did payment fail?"}
        Server → {"event": "agent_progress", "agent": "planner", "message": "..."}
        Server → {"event": "agent_progress", "agent": "retriever", "message": "..."}
        ...
        Server → {"event": "done", "response": "...", "citations": [...]}
    """
    await websocket.accept()

    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        question = data.get("question", "").strip()

        if not question:
            await websocket.send_text(
                json.dumps({"event": "error", "message": "question is required"})
            )
            await websocket.close()
            return

        log.info("ws_query_started", question=question)

        initial_state = _build_initial_state(question)

        agent_labels = {
            "planner":    "Analysing your question...",
            "retriever":  "Searching log database...",
            "reasoning":  "Building incident timeline...",
            "critic":     "Validating evidence...",
            "reflection": "Checking for gaps...",
            "citation":   "Generating cited report...",
        }

        def run_stream():
            states = []
            for node_name, state_update in agent_graph.stream(initial_state):
                states.append((node_name, state_update))
            return states

        step_results = await asyncio.to_thread(run_stream)

        final_state: AgentState | None = None
        for node_name, state_update in step_results:
            label = agent_labels.get(node_name, node_name)
            await websocket.send_text(json.dumps({
                "event":   "agent_progress",
                "agent":   node_name,
                "message": label,
            }))
            log.info("ws_agent_step", agent=node_name, question=question)
            final_state = state_update

        if final_state:
            await websocket.send_text(json.dumps({
                "event":     "done",
                "response":  final_state.get("final_response", ""),
                "citations": final_state.get("citations", []),
            }))
            log.info(
                "ws_query_completed",
                question=question,
                citations_count=len(final_state.get("citations", [])),
            )
        else:
            await websocket.send_text(
                json.dumps({"event": "error", "message": "Pipeline produced no output."})
            )

    except WebSocketDisconnect:
        log.info("ws_client_disconnected")
    except Exception as exc:
        log.error("ws_query_failed", error=str(exc))
        await websocket.send_text(json.dumps({"event": "error", "message": str(exc)}))
    finally:
        await websocket.close()
