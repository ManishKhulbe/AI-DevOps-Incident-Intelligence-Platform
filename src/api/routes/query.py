import asyncio
import json

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from src.agents.graph import agent_graph
from src.agents.state import AgentState

router = APIRouter(tags=["query"])


# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., description="Plain-English incident question")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Why did the payment service fail at 2:28 PM?",
            }
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
    """
    Build the starting state dict for graph.invoke().

    Every key in AgentState must be present — LangGraph does not fill
    missing keys with defaults. Providing empty values here means every
    node can safely read any key without a KeyError.
    """
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
        "Run the full 6-agent pipeline (Planner → Retriever → Reasoning → "
        "Critic → Reflection → Citation) and return the complete RCA with citations."
    ),
)
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    """
    FR-2 + FR-3: Natural language incident query with AI-generated RCA.

    Why asyncio.to_thread for graph.invoke()?
    LangGraph's .invoke() is synchronous — it blocks until all nodes complete.
    Calling it directly in an async handler would freeze the FastAPI event loop
    for the full duration of 6 LLM calls (~5-15 seconds).
    asyncio.to_thread() runs it in a thread pool so the event loop stays free
    to handle other requests while this one processes.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        initial_state = _build_initial_state(body.question)
        result: AgentState = await asyncio.to_thread(agent_graph.invoke, initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent pipeline failed: {exc}",
        )

    return QueryResponse(
        question=body.question,
        final_response=result["final_response"],
        citations=[CitationOut(**c) for c in result.get("citations", [])],
        request_id=request_id,
    )


# ── WebSocket /query/stream — token-by-token streaming ────────────────────────

@router.websocket("/query/stream")
async def query_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint that streams agent progress events to the client.

    Why streaming instead of just waiting for the full response?
    The 6-agent pipeline takes 5-15 seconds. A blank screen for 15 seconds
    feels broken. Streaming lets the UI show:
      "Planner: Identified payment-service incident at 14:28..."
      "Retriever: Found 5 relevant log chunks..."
      "Reasoning: Building timeline..."
      (final response appears token by token)

    Protocol:
    1. Client connects and sends: {"question": "Why did payment fail?"}
    2. Server sends progress events: {"event": "agent_start", "agent": "planner"}
    3. Server sends final result:   {"event": "done", "response": "...", "citations": [...]}
    4. Server closes connection

    Each message is a JSON string. The client parses and renders progressively.
    """
    await websocket.accept()

    try:
        # 1. Receive the question from the client
        raw = await websocket.receive_text()
        data = json.loads(raw)
        question = data.get("question", "").strip()

        if not question:
            await websocket.send_text(json.dumps({"event": "error", "message": "question is required"}))
            await websocket.close()
            return

        initial_state = _build_initial_state(question)

        # 2. Stream agent progress by running each node step and sending updates.
        #    LangGraph's .stream() yields (node_name, output_state) tuples after
        #    each node completes — this is different from token streaming (which
        #    would need LLM streaming callbacks), but gives the user visible
        #    progress between each of the 6 agents.
        agent_labels = {
            "planner":    "Analysing your question...",
            "retriever":  "Searching log database...",
            "reasoning":  "Building incident timeline...",
            "critic":     "Validating evidence...",
            "reflection": "Checking for gaps...",
            "citation":   "Generating cited report...",
        }

        final_state: AgentState | None = None

        def run_stream():
            """Run the graph stream in a thread (graph.stream is synchronous)."""
            states = []
            for node_name, state_update in agent_graph.stream(initial_state):
                states.append((node_name, state_update))
            return states

        # Run the full stream in a thread, collect (node, state) pairs
        step_results = await asyncio.to_thread(run_stream)

        for node_name, state_update in step_results:
            label = agent_labels.get(node_name, node_name)
            await websocket.send_text(json.dumps({
                "event": "agent_progress",
                "agent": node_name,
                "message": label,
            }))
            # Keep the last state as the final result
            final_state = state_update

        # 3. Send the final completed response
        if final_state:
            await websocket.send_text(json.dumps({
                "event": "done",
                "response": final_state.get("final_response", ""),
                "citations": final_state.get("citations", []),
            }))
        else:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": "Pipeline produced no output.",
            }))

    except WebSocketDisconnect:
        pass  # Client disconnected mid-stream — nothing to clean up
    except Exception as exc:
        await websocket.send_text(json.dumps({"event": "error", "message": str(exc)}))
    finally:
        await websocket.close()
