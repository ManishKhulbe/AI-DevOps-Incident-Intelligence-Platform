from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import settings
from src.ingestion.models import LogChunk

COLLECTION = "incident_logs"
VECTOR_SIZE = 1024  # output dim of BAAI/bge-large-en-v1.5

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    """
    Lazy singleton. We create the client once and reuse it for every request.
    Creating a new HTTP connection per request would be very slow.
    """
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url)
    return _client


async def _ensure_collection() -> None:
    """
    Create the Qdrant collection with Cosine distance if it doesn't exist yet.

    Why Cosine distance?
    BGE embeddings are L2-normalised (normalize_embeddings=True in embedder.py),
    so cosine similarity == dot product. Qdrant optimises dot-product search
    very efficiently when vectors are normalised.
    """
    client = get_client()
    existing = await client.get_collections()
    names = {c.name for c in existing.collections}
    if COLLECTION not in names:
        await client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


async def upsert_chunks(chunks: list[LogChunk], vectors: list[list[float]]) -> None:
    """
    Write vectors + metadata payload to Qdrant.

    Each point has:
    - id      : the chunk_id UUID string (Qdrant accepts UUID-format strings)
    - vector  : 1024-float embedding from BGE
    - payload : all metadata we want to filter on during retrieval

    Why store content in the payload?
    Qdrant is a vector store, not a document store. We store the raw text in
    the payload so retrieval returns self-contained results without a second
    round-trip to Elasticsearch.
    """
    await _ensure_collection()
    client = get_client()

    points = [
        PointStruct(
            id=chunk.chunk_id,
            vector=vector,
            payload={
                "chunk_id":      chunk.chunk_id,
                "source_log_id": chunk.source_log_id,
                "service_name":  chunk.service_name,
                "environment":   chunk.environment,
                "severity":      chunk.severity.value,
                "timestamp":     chunk.timestamp.isoformat(),
                "content":       chunk.content,
                "trace_id":      chunk.metadata.get("trace_id"),
                "deployment_id": chunk.metadata.get("deployment_id"),
                "host":          chunk.metadata.get("host"),
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    await client.upsert(collection_name=COLLECTION, points=points)


async def vector_search(
    query_vector: list[float],
    service_name: str | None = None,
    environment: str | None = None,
    severity: list[str] | None = None,
    top_k: int = 20,
) -> list[dict]:
    """
    Semantic nearest-neighbour search with optional payload filters.

    Filters narrow the candidate set BEFORE the ANN search, which is both
    faster and more accurate than post-filtering. Qdrant calls this
    'filtered HNSW' — it maintains per-filter index structures internally.
    """
    await _ensure_collection()
    client = get_client()

    conditions = []
    if service_name:
        conditions.append(FieldCondition(key="service_name", match=MatchValue(value=service_name)))
    if environment:
        conditions.append(FieldCondition(key="environment", match=MatchValue(value=environment)))

    qdrant_filter = Filter(must=conditions) if conditions else None

    hits = await client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "chunk_id":    hit.payload["chunk_id"],
            "content":     hit.payload["content"],
            "service_name": hit.payload["service_name"],
            "timestamp":   hit.payload["timestamp"],
            "severity":    hit.payload["severity"],
            "score":       hit.score,
            "source":      "vector",
        }
        for hit in hits
    ]
