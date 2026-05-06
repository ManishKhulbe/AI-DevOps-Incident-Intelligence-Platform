import asyncio

from sentence_transformers import CrossEncoder

from src.config import settings

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """
    Lazy singleton — load the cross-encoder model only on first call.

    Why lazy? The model is ~1 GB. Loading it at import time would slow down
    every startup (including test runs that never call rerank). Loading on
    first use means the cost is paid once, when the first query arrives.
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


async def rerank(query: str, documents: list[dict], top_k: int | None = None) -> list[dict]:
    """
    Rerank candidate documents using a cross-encoder model.

    Bi-encoder (embedder) vs cross-encoder (reranker):
    - Bi-encoder encodes query and document SEPARATELY → fast but less accurate.
      Used in Phase 2 to generate embeddings for storage and retrieval.
    - Cross-encoder sees BOTH query and document at the same time → much more
      accurate because it models their interaction, but O(n) in doc count.

    The pattern here is: retrieve top-20 cheaply (bi-encoder + BM25),
    then rerank to top-5 accurately (cross-encoder). This gives near-optimal
    precision without paying the cross-encoder cost on every document.

    Why asyncio.to_thread?
    CrossEncoder.predict is CPU-bound PyTorch — same reason as embed_chunks.
    Running it in a thread keeps the FastAPI event loop unblocked.
    """
    if not documents:
        return []

    reranker = _get_reranker()
    pairs = [(query, doc["content"]) for doc in documents]

    scores: list[float] = await asyncio.to_thread(reranker.predict, pairs)

    scored = [
        {**doc, "rerank_score": float(score)}
        for doc, score in zip(documents, scores)
    ]

    ranked = sorted(scored, key=lambda d: d["rerank_score"], reverse=True)
    return ranked[:top_k] if top_k is not None else ranked
