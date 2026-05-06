import asyncio

from src.ingestion.embedder import embed_chunks
from src.ingestion.qdrant_store import vector_search


async def retrieve_vector(
    query: str,
    service_name: str | None = None,
    environment: str | None = None,
    severity: list[str] | None = None,
    top_k: int = 20,
) -> list[dict]:
    """
    Embed the query then run semantic nearest-neighbour search in Qdrant.

    Why asyncio.to_thread for embedding?
    embed_chunks runs a PyTorch model — CPU-bound and blocking.
    Calling it directly inside an async function would block the entire
    FastAPI event loop, freezing all other in-flight requests.
    asyncio.to_thread runs it in a thread pool so the event loop stays free.

    Returns a ranked list of chunk dicts ordered by cosine similarity descending.
    Each dict has: chunk_id, content, service_name, timestamp, severity,
    score (cosine), source ("vector").
    """
    # Embed in a thread so we don't block the event loop
    query_vector = await asyncio.to_thread(embed_chunks, [query])
    query_vector = query_vector[0]

    return await vector_search(
        query_vector=query_vector,
        service_name=service_name,
        environment=environment,
        severity=severity,
        top_k=top_k,
    )
