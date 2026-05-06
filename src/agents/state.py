from typing import TypedDict


class AgentState(TypedDict):
    """
    Shared state that flows through every node in the LangGraph graph.

    Why TypedDict?
    LangGraph requires a TypedDict so it knows the exact shape of state at
    compile time. This lets it validate that every node only reads/writes
    keys that exist, catching bugs before runtime.

    Node contract: every node receives the full state and returns a DICT
    containing only the keys it changed. LangGraph merges the returned dict
    into the existing state — it does not replace the whole state.

    Example: reasoning node only touches reasoning_output, so it returns:
        {"reasoning_output": "Timeline: ..."}
    It does not return user_query, retrieved_chunks, etc.
    """

    # ── Input ─────────────────────────────────────────────────────────────
    user_query: str              # The raw question from the user

    # ── Planner output ────────────────────────────────────────────────────
    query_plan: dict             # Structured plan: intent, services, time range, sub-questions

    # ── Retriever output ──────────────────────────────────────────────────
    retrieved_chunks: list[dict] # Top-k log chunks from hybrid retrieval + reranking

    # ── Reasoning output ──────────────────────────────────────────────────
    reasoning_output: str        # Draft RCA: timeline + root cause (evidence-only)

    # ── Critic output ─────────────────────────────────────────────────────
    critic_feedback: dict        # {"valid": bool, "issues": [{"claim": ..., "reason": ...}]}

    # ── Reflection tracking ───────────────────────────────────────────────
    retry_count: int             # How many retrieval retries have been attempted (max 3)
    additional_queries: list[str]# Queries generated from unsupported claims for retry

    # ── Final output ──────────────────────────────────────────────────────
    citations: list[dict]        # [{"claim": ..., "evidence": ..., "chunk_id": ..., "confidence": ...}]
    final_response: str          # Human-readable answer with inline citation markers
