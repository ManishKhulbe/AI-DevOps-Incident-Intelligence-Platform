import asyncio

from src.agents.state import AgentState
from src.retrieval.retrieval_engine import retrieve
from src.observability.logger import get_logger

log = get_logger(__name__)


def retriever_node(state: AgentState) -> dict:
    """
    Retriever Agent — calls the Phase 3 retrieval engine once per sub_question,
    deduplicates by chunk_id, and returns the merged evidence set.
    """
    plan          = state["query_plan"]
    services      = plan.get("services", [])
    time_range    = plan.get("time_range", {}) or {}
    severity      = plan.get("severity_filter", [])
    sub_questions = plan.get("sub_questions", [])
    additional    = state.get("additional_queries", [])
    retry_count   = state.get("retry_count", 0)

    all_queries  = sub_questions + additional
    service_name = services[0] if services else None
    start_time   = time_range.get("start")
    end_time     = time_range.get("end")

    log.info(
        "agent_start",
        agent="retriever",
        queries_count=len(all_queries),
        service=service_name,
        retry=retry_count,
    )

    seen_ids: set[str] = set()
    merged: list[dict] = []

    for query in all_queries:
        results = asyncio.run(
            retrieve(
                query=query,
                service_name=service_name,
                severity=severity if severity else None,
                start_time=start_time,
                end_time=end_time,
            )
        )
        for chunk in results:
            if chunk["chunk_id"] not in seen_ids:
                seen_ids.add(chunk["chunk_id"])
                merged.append(chunk)

    merged.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)

    log.info(
        "agent_done",
        agent="retriever",
        chunks_retrieved=len(merged),
        service=service_name,
    )

    return {"retrieved_chunks": merged}
