# Implementation Guide — DevOps Incident Intelligence Platform

## Mental Model First

Before any code, understand the two pipelines in this system:

```
WRITE PATH (Ingestion)          READ PATH (Query)
──────────────────────          ─────────────────
Logs → Clean → Chunk            User Question
     → Embed                         ↓
     → Store in Qdrant           Hybrid Retrieval (ES + Qdrant)
     → Index in Elasticsearch        ↓
                                  Reranker
                                      ↓
                                  Agent System
                                      ↓
                                  Response + Citations
```

Every phase below builds one piece of either the write path or read path.
Never build both at the same time.

---

## Build Order

```
Week 1   Phase 0 (infra) + Phase 1 (ingestion)
Week 2   Phase 2 (embedding + storage)
Week 3   Phase 3 (retrieval) ← most important, take your time here
Week 4   Phase 4 (agents)   ← build one agent per day
Week 5   Phase 5 (API) + Phase 6 (observability)
Week 6   Phase 7 (evaluation) + Phase 8 (frontend)
```

> Test at every phase boundary before moving forward.
> Agents built on bad retrieval are useless.

---

## Phase 0 — Project Skeleton + Infrastructure

**What you learn:** Python project structure, Docker Compose, environment config

**Why this comes first:** Every later phase depends on Elasticsearch and Qdrant running.
Get the infrastructure up once; never touch it again.

### Folder Structure

```
devops-incident-agent/
├── docker-compose.yml
├── .env
├── .env.example
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion/
│   ├── retrieval/
│   ├── agents/
│   ├── api/
│   ├── evaluation/
│   └── observability/
├── scripts/
│   └── seed_sample_logs.py
└── tests/
```

### docker-compose.yml — What to Put In It

Three services:

| Service | Purpose | Port |
|---|---|---|
| Elasticsearch 8.x | Full-text BM25 search | 9200 |
| Qdrant | Vector database | 6333 |
| Redis | Rate limiting + caching | 6379 |

**Why Docker Compose and not local install?**
Production always runs in containers. Defining infrastructure as code from day 1 is the right habit.

### `src/config.py` — Pattern to Understand

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    elasticsearch_url: str
    qdrant_url: str
    openai_api_key: str
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-large"
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
```

**Why `pydantic_settings`?**
It validates types at startup. If `ELASTICSEARCH_URL` is missing, the app crashes immediately
with a clear error instead of failing mysteriously 10 minutes into a query.
Production-grade config is validated config.

---

## Phase 1 — Log Ingestion Service

**What you learn:** FastAPI, Pydantic models, async Python, log parsing, chunking strategy

**Why chunking matters:**
A Kubernetes log file can be 100,000 lines. You cannot embed a 100k-line file as one vector —
it loses all meaning. Chunk it into semantically meaningful units, embed each chunk,
and retrieve only the relevant ones.

### What is a "Chunk"?

For logs, the right chunk is **one log event + its context window** (2-3 lines before/after).
Why? A single `ERROR: connection refused` line is useless without seeing the preceding
`INFO: attempting DB connection` line.

### Files to Create in `src/ingestion/`

```
src/ingestion/
├── __init__.py
├── models.py       ← Pydantic schemas for incoming log payloads
├── cleaner.py      ← removes noise (ANSI codes, duplicate timestamps)
├── chunker.py      ← splits logs into overlapping windows
├── metadata.py     ← extracts service name, severity, timestamp, trace ID
└── pipeline.py     ← orchestrates: clean → chunk → extract metadata
```

### `models.py` — What to Define

```python
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class LogSeverity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class RawLog(BaseModel):
    source: str           # "kubernetes", "json", "text"
    service_name: str
    environment: str      # "prod", "staging"
    content: str          # raw log string or JSON stringified
    ingested_at: datetime = datetime.utcnow()

class LogChunk(BaseModel):
    chunk_id: str         # UUID
    source_log_id: str
    service_name: str
    environment: str
    severity: LogSeverity
    timestamp: datetime
    content: str          # the actual chunk text
    metadata: dict        # trace_id, deployment_id, host, etc.
```

**Why enums for severity?**
Downstream agents need to filter `severity == ERROR`. String comparison breaks when one system
writes `"error"` and another writes `"ERROR"`. Enums enforce consistency at the boundary.

### `chunker.py` — The Key Algorithm

**Sliding window with overlap:**
- Window size: 10 log lines
- Overlap: 3 lines (last 3 lines of chunk N become first 3 lines of chunk N+1)

Why overlap? So a sequence of events that spans a chunk boundary doesn't get split and lose context.

### `metadata.py` — What to Extract

Use regex patterns to extract:

| Field | Look for |
|---|---|
| Timestamp | ISO 8601, epoch, custom formats |
| Severity | `ERROR`, `WARN`, `INFO`, etc. |
| Trace ID | `trace_id=`, `request_id=`, `X-Request-ID` |
| Service name | Header or log prefix |
| Deployment ID | `deploy_id=`, `version=`, `git_sha=` |

**Why extract metadata?**
Phase 3 retrieval uses metadata filters. "Show me logs from the payment service from yesterday"
requires filtering by `service_name` and `timestamp` — pure vector search cannot do that.

### API Endpoint to Build (FR-1)

```
POST /api/v1/ingest
Body:     { "source": "kubernetes", "service_name": "payment", "content": "..." }
Response: { "chunks_stored": 42, "log_id": "uuid" }
```

---

## Phase 2 — Embedding + Dual Storage

**What you learn:** How embeddings work, vector databases, Elasticsearch indexing, dual-write pattern

### What is an Embedding?

An embedding converts text → a list of numbers (e.g. 1024 floats for `bge-large-en-v1.5`).
Two semantically similar sentences will have vectors that are geometrically close.
This lets you find relevant logs by meaning, not just keyword match.

**Why `BAAI/bge-large-en-v1.5` and not OpenAI embeddings?**
- Runs locally — no API cost per embedding
- Top-performing open model on MTEB benchmark
- At millions of logs, paying per embedding gets expensive fast

### Files to Create (extending `src/ingestion/`)

```
src/ingestion/
├── embedder.py       ← loads BGE model, generates embeddings in batches
├── qdrant_store.py   ← upserts vectors into Qdrant collection
└── es_store.py       ← indexes text into Elasticsearch
```

### `embedder.py` — Key Concepts

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    # BGE requires a specific instruction prefix for retrieval tasks
    prefixed = [f"Represent this incident log for retrieval: {c}" for c in chunks]
    return model.encode(prefixed, batch_size=32, normalize_embeddings=True).tolist()
```

**Why the instruction prefix?**
BGE is instruction-tuned. Adding a task-specific prefix dramatically improves retrieval quality.
Without it, you are using the model incorrectly.

**Why `normalize_embeddings=True`?**
Normalizing makes cosine similarity equivalent to dot product, which is faster and what Qdrant uses.

### `qdrant_store.py` — Key Concepts

Create a collection with:
- `vector_size`: 1024 (BGE-large output dimension)
- `distance`: Cosine
- Payload fields: `service_name`, `timestamp`, `severity`, `environment`

Qdrant payload = the metadata stored alongside the vector. You need this for filtering in Phase 3.

### `es_store.py` — Key Concepts

Create an Elasticsearch index mapping:

| Field | ES Type | Why |
|---|---|---|
| `content` | `text` | Enables BM25 full-text search (analyzed, tokenized) |
| `service_name` | `keyword` | Enables exact-match filter (stored as-is) |
| `severity` | `keyword` | Exact filter |
| `environment` | `keyword` | Exact filter |
| `timestamp` | `date` | Enables range queries |

**Why `keyword` vs `text`?**
`text` fields are analyzed (tokenized, lowercased, stemmed) — good for full-text search.
`keyword` fields are stored as-is — good for filtering (`service_name == "payment"`).
Mixing them up is one of the most common Elasticsearch mistakes.

### Dual-Write Pattern

Write to both stores in the same flow:

```
embed(chunk) → qdrant.upsert(vector + payload)
             + es.index(text + metadata)
```

If one fails, roll back the other using the `chunk_id`. Store the same `chunk_id` in both.

---

## Phase 3 — Hybrid Retrieval Engine

**What you learn:** BM25 vs vector search, RRF fusion, cross-encoder reranking

**This is the most technically important phase.**
Poor retrieval = poor answers, no matter how good your LLM is.

### Why Hybrid Retrieval?

| Scenario | BM25 wins | Vector wins |
|---|---|---|
| Query: `"OOMKilled"` (exact error code) | Yes — finds exact string | No — may miss it |
| Query: `"memory ran out"` (semantic) | No — no keyword match | Yes — finds OOMKilled logs |
| Query: `"payment timeout last Tuesday"` | Partial | Partial |

Neither alone is sufficient. Hybrid retrieval combines both.

### Files to Create in `src/retrieval/`

```
src/retrieval/
├── __init__.py
├── bm25_retriever.py       ← Elasticsearch query
├── vector_retriever.py     ← Qdrant similarity search
├── hybrid_retriever.py     ← fuses both result sets using RRF
├── reranker.py             ← cross-encoder reranking
└── retrieval_engine.py     ← orchestrates the full pipeline
```

### `bm25_retriever.py` — What to Build

```python
query = {
    "query": {
        "bool": {
            "must": {
                "multi_match": {"query": user_query, "fields": ["content"]}
            },
            "filter": [
                {"term": {"service_name": service_filter}},
                {"range": {"timestamp": {"gte": start_time, "lte": end_time}}}
            ]
        }
    }
}
```

**Why `bool` query?**
It separates scoring (`must`) from filtering (`filter`). Filters are cached by Elasticsearch
and do not affect relevance score. The `must` clause uses BM25 TF-IDF to score.

### `vector_retriever.py` — What to Build

```python
results = qdrant_client.search(
    collection_name="incident_logs",
    query_vector=embed(user_query),
    query_filter=Filter(
        must=[FieldCondition(key="service_name", match=MatchValue(value=service_filter))]
    ),
    limit=top_k
)
```

### `hybrid_retriever.py` — RRF Fusion

**Reciprocal Rank Fusion (RRF)** is the standard algorithm for combining ranked lists.

```
score(doc) = Σ  1 / (k + rank_in_list)    where k = 60
```

Steps:
1. Fetch top-20 results from BM25
2. Fetch top-20 results from vector search
3. For each document, sum its RRF scores from both lists
4. Sort by combined score

A document that ranks #3 in BM25 AND #5 in vector search scores much higher
than one that appears in only one list. `k=60` is empirically the best constant.

### `reranker.py` — Cross-Encoder Reranking

The embedder (bi-encoder) generates vectors independently for query and document — fast but less accurate.
The cross-encoder sees both query and document together — slow but much more accurate.

**Pattern: retrieve top-20, rerank to top-5.**

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-large")

def rerank(query: str, documents: list[str], top_k: int = 5) -> list[int]:
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]
```

**Why rerank only top-20 and not all results?**
Cross-encoder is O(n) — running it on 1000 docs would take 10+ seconds.
The bi-encoder narrows the candidate set cheaply; the cross-encoder makes the final precise ranking.

---

## Phase 4 — LangGraph Multi-Agent System

**What you learn:** LangGraph state machines, agent design patterns, tool use, role-specific prompting

**Build one agent at a time. Test each one before adding the next.**

### Why 6 Agents Instead of One?

One monolithic agent with a giant prompt is brittle.

| Agent | Single responsibility |
|---|---|
| Planner | Understands query intent |
| Retriever | Fetches relevant evidence |
| Reasoning | Builds timeline and RCA |
| Critic | Validates claims against evidence |
| Reflection | Decides to retry or proceed |
| Citation | Attaches source references |

You can swap, retrain, or tune each independently.
LangGraph lets you see exactly which agent caused a failure.

### `AgentState` — Design This First

```python
from typing import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    user_query: str
    query_plan: dict              # Planner's structured plan
    filters: dict                 # service, time range, severity
    retrieved_chunks: list[dict]  # from Retrieval Engine (Phase 3)
    reasoning_output: str         # Reasoning Agent's RCA draft
    critic_feedback: dict         # {"valid": bool, "issues": [...]}
    citations: list[dict]         # chunk_id + relevant excerpt
    final_response: str
    retry_count: int
    session_id: str
```

### Agent 1 — Planner Agent

**Input:** raw user query
**Output:** structured JSON plan

```json
{
  "intent": "root_cause_analysis",
  "services": ["payment-service"],
  "time_range": {
    "start": "2024-01-15T14:00:00",
    "end": "2024-01-15T16:00:00"
  },
  "severity_filter": ["ERROR", "CRITICAL"],
  "sub_questions": [
    "What errors occurred in payment-service between 2-4pm?",
    "Were there any deployment events in that window?"
  ]
}
```

**Why structured JSON output?**
The Retriever Agent reads `filters` from this plan to build ES/Qdrant queries.
Unstructured text output would need fragile parsing.
Use `with_structured_output(PlannerOutput)` in LangChain to force valid JSON.

### Agent 2 — Retriever Agent

**Input:** `query_plan` from state
**Output:** fills `retrieved_chunks` in state

This agent calls your Phase 3 retrieval engine as a LangChain Tool.
It runs one retrieval query per `sub_question` in the plan and merges results.

### Agent 3 — Reasoning Agent

**Input:** user query + retrieved chunks
**Output:** draft RCA narrative

Structure the system prompt as:
1. "Here are the relevant logs: {chunks}"
2. "Build a timeline of events"
3. "Identify the root cause based ONLY on evidence in the logs above"
4. "List your reasoning steps"

**Why "based only on evidence"?**
Prevents hallucination. The Critic Agent checks every claim against source chunks.
If you allow the model to reason freely, it invents plausible-sounding causes that
are not in the logs.

### Agent 4 — Critic Agent

**Input:** reasoning output + source chunks
**Output:** `{"valid": bool, "issues": [{"claim": "...", "supported_by": "none"}]}`

The Critic compares every factual claim in the Reasoning Agent's output against source chunks.
If a claim has no supporting evidence, it flags it as unsupported.

**Key prompt pattern:**
```
For each factual claim in the RCA below, find the log chunk that supports it.
If no chunk supports a claim, mark it UNSUPPORTED.

RCA: {reasoning_output}
Available evidence: {chunks}
```

### Agent 5 — Reflection Agent

**Input:** critic feedback
**Logic (this is the retry loop):**

```
if critic_feedback.valid == True:
    → go to Citation Agent

if critic_feedback.valid == False AND retry_count < 3:
    → extract unsupported claims as new search queries
    → go back to Retriever Agent with those queries
    → increment retry_count

if critic_feedback.valid == False AND retry_count >= 3:
    → go to Citation Agent with a low-confidence caveat
```

The unsupported claims become new search queries — the agent asks
"what evidence do I need to support or refute this claim?"

### Agent 6 — Citation Agent

**Input:** final reasoning + validated chunks
**Output:** response with inline citations

Format each citation:
```json
{
  "claim": "Payment service timed out at 14:32:15",
  "evidence": "14:32:15 ERROR payment-service Connection timeout after 30s",
  "chunk_id": "abc-123",
  "confidence": 0.94
}
```

### Graph Wiring

```python
graph.add_node("planner", planner_agent)
graph.add_node("retriever", retriever_agent)
graph.add_node("reasoning", reasoning_agent)
graph.add_node("critic", critic_agent)
graph.add_node("reflection", reflection_agent)
graph.add_node("citation", citation_agent)

graph.set_entry_point("planner")
graph.add_edge("planner", "retriever")
graph.add_edge("retriever", "reasoning")
graph.add_edge("reasoning", "critic")
graph.add_edge("critic", "reflection")

graph.add_conditional_edges("reflection", route_after_reflection, {
    "retriever": "retriever",   # retry with new search queries
    "citation": "citation",     # proceed to final response
})

graph.add_edge("citation", END)
```

### Full Agent Flow

```
User Query
    ↓
Planner Agent  →  structured plan (intent, services, time range, sub-questions)
    ↓
Retriever Agent  →  hybrid retrieval for each sub-question → top-5 chunks
    ↓
Reasoning Agent  →  timeline + RCA draft (only from retrieved evidence)
    ↓
Critic Agent  →  validates every claim against source chunks
    ↓
Reflection Agent
    ├── valid? → Citation Agent
    └── invalid + retry < 3? → back to Retriever (with unsupported claims as new queries)
    └── invalid + retry >= 3? → Citation Agent (with low confidence caveat)
    ↓
Citation Agent  →  final response + inline citations + confidence scores
    ↓
Final Response
```

---

## Phase 5 — FastAPI Layer

**What you learn:** Async FastAPI, streaming responses, middleware, JWT auth

### Endpoints to Build

```
POST /api/v1/ingest          ← FR-1: upload logs
POST /api/v1/query           ← FR-2: incident query (full response)
WS   /api/v1/query/stream    ← FR-2: streaming (token by token)
GET  /api/v1/health          ← infrastructure health check
POST /api/v1/auth/token      ← JWT login
```

### Production Patterns to Apply

**1. Request ID middleware**
Every request gets a UUID injected into headers and logs.
Without it, debugging which request caused an error in high traffic is impossible.

```python
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
```

**2. Async everywhere**
All I/O (Qdrant, Elasticsearch, LLM calls) must use `async/await`.
Blocking calls in FastAPI handlers freeze the entire event loop.

**3. Streaming for LLM responses**
LLM generation is slow. Stream tokens using `StreamingResponse` with `astream()`.
Users see output immediately instead of waiting 10 seconds for the full response.

**4. Pydantic request/response models**
Every endpoint has typed request and response models.
This auto-generates OpenAPI docs at `/docs` and validates inputs at the boundary.

---

## Phase 6 — Observability

**What you learn:** Distributed tracing, structured logging, LangSmith

### LangSmith Integration

Traces every LLM call: which agent, which prompt, how many tokens, latency, input/output.
Enable with two environment variables — no code changes needed:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=devops-incident-agent
```

Every `graph.invoke()` call automatically traces all 6 agents.

### Structured Logging

Replace `print()` with a structured JSON logger.
Every log line should carry:

```json
{
  "level": "ERROR",
  "request_id": "abc-123",
  "agent": "critic",
  "message": "Hallucination detected",
  "claim": "Payment service restarted at 14:35",
  "timestamp": "2024-01-15T14:32:00Z"
}
```

**Why structured logs?**
In production, logs go to a log aggregator (Datadog, CloudWatch, Loki).
Structured logs are queryable: "show all hallucination detections in the last 24h".
Unstructured strings are not.

Use `structlog` or `python-json-logger` — both integrate cleanly with FastAPI.

---

## Phase 7 — Evaluation Pipeline

**What you learn:** Ragas metrics, ground truth dataset creation, automated quality testing

### Why You Need This

Before you ship, you need to know: does the system hallucinate? How often?
How good is retrieval? Without metrics, you are guessing.

### Ragas Measures Four Things

| Metric | Question it answers |
|---|---|
| Faithfulness | Is every claim in the answer supported by retrieved context? |
| Answer Relevancy | Does the answer actually address the question? |
| Context Precision | Are the retrieved chunks relevant to the question? |
| Context Recall | Did we miss any relevant chunks? |

### How to Build a Ground Truth Dataset

1. Create 20-30 sample log scenarios (synthetic is fine for learning)
2. Write the expected answer for each by hand
3. Run your system on each scenario
4. Ragas compares system output vs expected answer and scores each metric

Run this as a script before every major change to the agent system or prompts.
If faithfulness drops from 0.92 to 0.78 after a prompt change, you know immediately.

```
scripts/
└── evaluate.py    ← loads ground truth, runs all scenarios, prints Ragas scores
```

---

## Phase 8 — Frontend (Next.js)

**What you learn:** Next.js App Router, streaming UI, file upload, Server vs Client Components

### Pages to Build

```
/                    ← Dashboard: recent incidents, MTTR metrics
/query               ← Chat interface for incident queries
/ingest              ← Log upload (drag-and-drop file)
/incidents/:id       ← Incident detail: full RCA + citations + timeline
/observability       ← Agent trace viewer
```

### Key Frontend Concepts

**Streaming UI**
Use `fetch()` with `ReadableStream` to display tokens as they arrive from the backend.
This makes the app feel fast even when the LLM takes 8-10 seconds to complete.

**Citation UI**
Render citations as inline footnote markers `[1]` in the response text.
Clicking a citation scrolls to the source log chunk.
This directly addresses the BRD objective: "explainable AI responses with citations."

**Component split: Server vs Client**
- Use Server Components for data fetching (incident list, metrics dashboard)
- Use Client Components only where you need interactivity (chat input, streaming display)
This keeps the bundle small and pages fast.

---

## Production Patterns — Apply Throughout All Phases

| Pattern | Where to Apply | Why |
|---|---|---|
| Pydantic models at every boundary | All phases | Catches bad data early, auto-generates docs |
| `async/await` for all I/O | Phases 1-5 | Prevents event loop blocking |
| Retry with exponential backoff | LLM calls, ES, Qdrant | Handles transient failures gracefully |
| Structured JSON logging | All phases | Queryable in production log systems |
| Request ID propagation | Phase 5 | Debug specific requests in high traffic |
| Environment-based config | Phase 0 | Never hardcode secrets |
| Batch embedding (not one-by-one) | Phase 2 | 10x faster ingestion |
| Elasticsearch index aliases | Phase 2 | Zero-downtime re-indexing |
| `temperature=0` for structured outputs | Phase 4 | Deterministic JSON from LLM |
| Validate SQL / query before execution | Phase 3 | Block destructive operations |

---

## What to Do Right Now

1. Create the folder structure
2. Write `docker-compose.yml` with Elasticsearch, Qdrant, Redis
3. Run `docker compose up -d` and verify all three services respond on their ports
4. Write `src/config.py` with `pydantic_settings`
5. Write `src/ingestion/models.py` with `RawLog` and `LogChunk`

That is your Phase 0. When those five things work, move to Phase 1.
