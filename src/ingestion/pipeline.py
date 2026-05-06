import uuid

from src.ingestion.cleaner import clean_log
from src.ingestion.chunker import chunk_log
from src.ingestion.embedder import embed_chunks
from src.ingestion.models import RawLog
from src.ingestion.qdrant_store import upsert_chunks
from src.ingestion.es_store import index_chunks


async def ingest(raw_log: RawLog) -> dict:
    """
    Full write-path pipeline: clean → chunk → embed → dual-write.

    Flow:
        1. clean_log    — strip ANSI codes, normalize JSON to text
        2. chunk_log    — sliding window into LogChunk objects
        3. embed_chunks — BGE vectors for every chunk's text
        4. upsert_chunks / index_chunks — write to Qdrant AND Elasticsearch

    Why async?
    Steps 4a and 4b are network I/O. Async lets FastAPI serve other requests
    while we wait for Qdrant/ES to acknowledge the writes.

    Why dual-write in the same function?
    We want both stores to have the same data. If Qdrant succeeds but ES fails
    (or vice versa), we raise — the caller gets an error and can retry.
    Both stores use chunk_id as their document key, so a retry is idempotent.
    """
    log_id = str(uuid.uuid4())

    cleaned = clean_log(raw_log.content, raw_log.source)
    chunks = chunk_log(raw_log, cleaned)

    if not chunks:
        return {"log_id": log_id, "chunks_stored": 0}

    vectors = embed_chunks([c.content for c in chunks])

    # Both raises on failure — intentionally not catching here so the
    # FastAPI endpoint can return a 500 with a clear message.
    await upsert_chunks(chunks, vectors)
    await index_chunks(chunks)

    return {"log_id": log_id, "chunks_stored": len(chunks)}
