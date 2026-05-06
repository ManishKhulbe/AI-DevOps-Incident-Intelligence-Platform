from src.ingestion.es_store import bm25_search


async def retrieve_bm25(
    query: str,
    service_name: str | None = None,
    environment: str | None = None,
    severity: list[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    top_k: int = 20,
) -> list[dict]:
    """
    Thin wrapper around es_store.bm25_search.

    Why a separate retrieval module instead of calling es_store directly?
    The retrieval layer is consumed by the hybrid retriever and later by
    LangGraph agent tools. Keeping it as a named, importable function makes
    it easy to swap the underlying search engine without touching agent code.

    Returns a ranked list of chunk dicts ordered by BM25 score descending.
    Each dict has: chunk_id, content, service_name, timestamp, severity,
    score (BM25), source ("bm25").
    """
    return await bm25_search(
        query=query,
        service_name=service_name,
        environment=environment,
        severity=severity,
        start_time=start_time,
        end_time=end_time,
        top_k=top_k,
    )
