from elasticsearch import AsyncElasticsearch

from src.config import settings
from src.ingestion.models import LogChunk

INDEX = "incident_logs"

# Explicit mapping tells Elasticsearch exactly how to index each field.
# Without it, ES would auto-detect types and almost always get it wrong
# (e.g. treating service_name as "text" instead of "keyword", breaking filters).
_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "chunk_id":      {"type": "keyword"},
            "source_log_id": {"type": "keyword"},
            "service_name":  {"type": "keyword"},   # exact-match filter
            "environment":   {"type": "keyword"},   # exact-match filter
            "severity":      {"type": "keyword"},   # exact-match filter
            "timestamp":     {"type": "date"},      # enables range queries
            "content":       {"type": "text", "analyzer": "standard"},  # BM25 full-text
            "trace_id":      {"type": "keyword"},
            "deployment_id": {"type": "keyword"},
            "host":          {"type": "keyword"},
        }
    },
    "settings": {
        "number_of_shards":   1,   # single node in dev — no need for sharding
        "number_of_replicas": 0,   # no replica in dev (saves memory)
    },
}

_client: AsyncElasticsearch | None = None


def get_client() -> AsyncElasticsearch:
    global _client
    if _client is None:
        _client = AsyncElasticsearch(hosts=[settings.elasticsearch_url])
    return _client


async def _ensure_index() -> None:
    client = get_client()
    if not await client.indices.exists(index=INDEX):
        await client.indices.create(index=INDEX, body=_INDEX_MAPPING)


async def index_chunks(chunks: list[LogChunk]) -> None:
    """
    Bulk-index chunks into Elasticsearch for BM25 full-text search.

    Why bulk instead of one request per chunk?
    Bulk sends all documents in a single HTTP request. For 42 chunks, bulk is
    ~40x faster than 42 individual index calls because it avoids 41 round-trips.
    Elasticsearch's bulk API also batches the internal Lucene writes.

    The operations list alternates between:
        {"index": {"_index": ..., "_id": ...}}   <- action header
        { ...document... }                        <- document body
    """
    await _ensure_index()
    client = get_client()

    operations = []
    for chunk in chunks:
        operations.append({"index": {"_index": INDEX, "_id": chunk.chunk_id}})
        operations.append({
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
        })

    await client.bulk(operations=operations)


async def bm25_search(
    query: str,
    service_name: str | None = None,
    environment: str | None = None,
    severity: list[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    top_k: int = 20,
) -> list[dict]:
    """
    BM25 keyword search with structured metadata filters.

    The ES bool query separates two concerns:
    - must  : scored by BM25 TF-IDF (higher score = more relevant)
    - filter: cached exact-match (does not affect score, just narrows candidates)

    Why put filters in 'filter' and not 'must'?
    'filter' results are cached by Elasticsearch. If 100 queries all filter
    by service_name="payment", ES caches that bitset and reuses it — much
    faster than rescoring the filter every time.
    """
    await _ensure_index()
    client = get_client()

    filters = []
    if service_name:
        filters.append({"term": {"service_name": service_name}})
    if environment:
        filters.append({"term": {"environment": environment}})
    if severity:
        filters.append({"terms": {"severity": severity}})
    if start_time or end_time:
        date_range: dict = {}
        if start_time:
            date_range["gte"] = start_time
        if end_time:
            date_range["lte"] = end_time
        filters.append({"range": {"timestamp": date_range}})

    es_query = {
        "query": {
            "bool": {
                "must":   {"multi_match": {"query": query, "fields": ["content"]}},
                "filter": filters,
            }
        },
        "size": top_k,
    }

    response = await client.search(index=INDEX, body=es_query)

    return [
        {
            "chunk_id":    hit["_source"]["chunk_id"],
            "content":     hit["_source"]["content"],
            "service_name": hit["_source"]["service_name"],
            "timestamp":   hit["_source"]["timestamp"],
            "severity":    hit["_source"]["severity"],
            "score":       hit["_score"],
            "source":      "bm25",
        }
        for hit in response["hits"]["hits"]
    ]
