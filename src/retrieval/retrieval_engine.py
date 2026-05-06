from src.config import settings
from src.retrieval.hybrid_retriever import retrieve_hybrid
from src.retrieval.reranker import rerank


async def retrieve(
    query: str,
    service_name: str | None = None,
    environment: str | None = None,
    severity: list[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> list[dict]:
    """
    Full read-path retrieval pipeline.

    Step 1 — Hybrid search (BM25 + vector, merged with RRF)
        Fetches top_k_retrieve candidates. Running both searches concurrently
        via asyncio.gather keeps latency close to max(bm25_time, vector_time)
        rather than sum(bm25_time + vector_time).

    Step 2 — Cross-encoder reranking
        Narrows candidates to top_k_rerank using a model that sees the full
        (query, document) pair. More accurate than bi-encoder similarity but
        only called on the small candidate set, so it stays within the 2s SLA.

    This is the function LangGraph's Retriever Agent will call as a Tool.
    All filter arguments are optional — the Planner Agent populates them from
    the structured query plan it produces.
    """
    candidates = await retrieve_hybrid(
        query=query,
        service_name=service_name,
        environment=environment,
        severity=severity,
        start_time=start_time,
        end_time=end_time,
        top_k=settings.top_k_retrieve,
    )

    return await rerank(query, candidates, top_k=settings.top_k_rerank)
