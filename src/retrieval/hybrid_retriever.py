import asyncio

from src.retrieval.bm25_retriever import retrieve_bm25
from src.retrieval.vector_retriever import retrieve_vector


def _reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Merge multiple ranked lists into one using Reciprocal Rank Fusion (RRF).

    Formula: score(doc) = sum of  1 / (k + rank)  across every list the doc appears in.

    Why k=60?
    The original RRF paper (Cormack et al. 2009) showed k=60 is empirically
    optimal across many retrieval benchmarks. It balances giving weight to
    high-ranked documents without completely ignoring low-ranked ones.

    Why not just average the scores?
    BM25 scores and cosine similarity scores are on completely different scales
    (BM25 can be 0-25, cosine is 0-1). Rank position is scale-independent.

    A document that appears at rank #2 in BM25 AND rank #4 in vector search
    will score much higher than one that only appears in one list, even if it
    scored highest in that list.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in docs:
                docs[cid] = doc

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    return [
        {**docs[cid], "rrf_score": scores[cid], "source": "hybrid"}
        for cid in ranked_ids
    ]


async def retrieve_hybrid(
    query: str,
    service_name: str | None = None,
    environment: str | None = None,
    severity: list[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    top_k: int = 20,
) -> list[dict]:
    """
    Run BM25 and vector search concurrently then merge with RRF.

    asyncio.gather runs both searches at the same time. Since each search is
    a network call (ES and Qdrant), parallelising them cuts latency roughly
    in half compared to running them sequentially.
    """
    bm25_results, vector_results = await asyncio.gather(
        retrieve_bm25(query, service_name, environment, severity, start_time, end_time, top_k),
        retrieve_vector(query, service_name, environment, severity, top_k),
    )

    return _reciprocal_rank_fusion([bm25_results, vector_results])
