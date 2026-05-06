# DevOps Incident Intelligence Platform — Project Overview

---

## What This Project Is

A system that lets a DevOps engineer ask a plain-English question like:

> "Why did the payment service fail at 2:28 PM yesterday?"

and receive a structured, evidence-backed answer with root cause analysis and
citations pointing to the exact log lines that support each claim.

No more manually grepping through 50,000 log lines across five services.
The system reads the logs, finds the relevant pieces, reasons over them, and
explains what happened — with proof.

---

## The Problem It Solves

When a production incident happens, engineers face four problems simultaneously:

| Problem | What it means in practice |
|---|---|
| Logs are scattered | Payment logs in Datadog, K8s logs in CloudWatch, DB logs on the server |
| Volume is overwhelming | A 1-hour incident window can produce millions of log lines |
| Context is lost | A single error line means nothing without the 3 lines before it |
| Cause is unclear | The error you see is rarely the root cause — it's the symptom |

This system solves all four: it ingests logs from any source, chunks them
with context preserved, retrieves only the relevant pieces per question,
and uses a chain of AI agents to reason from evidence to root cause.

---

## How It Works — The Two Paths

Every operation in this system follows one of two paths.
Understanding both is the key to understanding every file in the codebase.

```
╔══════════════════════════════════════════════════════════════════╗
║  WRITE PATH  (happens when logs are uploaded)                    ║
║                                                                  ║
║  Raw Logs                                                        ║
║      │                                                           ║
║      ▼                                                           ║
║  [cleaner.py]   Strip ANSI codes, normalize JSON → text         ║
║      │                                                           ║
║      ▼                                                           ║
║  [chunker.py]   Sliding window: 10 lines, 3-line overlap        ║
║      │                                                           ║
║      ▼                                                           ║
║  [metadata.py]  Extract: timestamp, severity, trace_id, host    ║
║      │                                                           ║
║      ▼                                                           ║
║  [embedder.py]  BGE model → 1024-float vector per chunk         ║
║      │                                                           ║
║      ├──────────────────────┐                                    ║
║      ▼                      ▼                                    ║
║  [qdrant_store.py]     [es_store.py]                            ║
║  Vector + payload      Text + metadata                           ║
║  (semantic search)     (keyword search)                          ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║  READ PATH  (happens when a user asks a question)                ║
║                                                                  ║
║  User Question: "Why did checkout fail after deployment?"        ║
║      │                                                           ║
║      ▼                                                           ║
║  [Planner Agent]                                                 ║
║  → intent: root_cause_analysis                                   ║
║  → services: [checkout-service]                                  ║
║  → time_range: {start: "16:00", end: "16:10"}                   ║
║  → sub_questions: ["What errors at 16:00?", "Any deployment?"]  ║
║      │                                                           ║
║      ▼                                                           ║
║  [Retriever Agent]  calls retrieval_engine.retrieve()           ║
║      │                                                           ║
║      ├── [bm25_retriever.py]   Elasticsearch keyword search     ║
║      ├── [vector_retriever.py] Qdrant semantic search           ║
║      ├── [hybrid_retriever.py] RRF merge → top-20 candidates    ║
║      └── [reranker.py]         Cross-encoder → top-5 results    ║
║      │                                                           ║
║      ▼                                                           ║
║  [Reasoning Agent]  Timeline + draft RCA (evidence-only)        ║
║      │                                                           ║
║      ▼                                                           ║
║  [Critic Agent]  Validates every claim against source chunks    ║
║      │                                                           ║
║      ├── All claims supported? ──► [Citation Agent]             ║
║      └── Unsupported claims?   ──► [Reflection Agent]           ║
║                                         │                        ║
║                                         ├── retry < 3? ──► Retriever (new queries)
║                                         └── retry >= 3? ─► Citation Agent
║      │                                                           ║
║      ▼                                                           ║
║  [Citation Agent]                                                ║
║  → Final answer with inline citations                            ║
║  → Each citation: claim + supporting log line + confidence score ║
║      │                                                           ║
║      ▼                                                           ║
║  Every LLM call traced automatically via LangSmith              ║
║  Every request logged as structured JSON via structlog           ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Data Flow — Step by Step

### Write Path (Log Ingestion)

#### Step 1 — Raw log arrives

An engineer uploads a Kubernetes log file via `POST /api/v1/ingest`.
It looks like this (raw, messy, unstructured):

```
\x1b[31m2024-01-15T14:28:05 ERROR payment-service Connection timeout\x1b[0m
2024-01-15T14:28:05 ERROR payment-service pool exhausted trace_id=abc-001
```

**File:** `src/ingestion/models.py`
The `RawLog` model validates the incoming request — source type, service name,
environment, and raw content.

---

#### Step 2 — Cleaning

**File:** `src/ingestion/cleaner.py`

`clean_log()` strips:
- ANSI terminal escape codes (`\x1b[31m` → nothing)
- Null bytes
- Carriage returns
- Blank lines

JSON logs are flattened to plaintext lines first.

After cleaning:
```
2024-01-15T14:28:05 ERROR payment-service Connection timeout
2024-01-15T14:28:05 ERROR payment-service pool exhausted trace_id=abc-001
```

---

#### Step 3 — Chunking

**File:** `src/ingestion/chunker.py`

`chunk_log()` splits the cleaned content using a sliding window:
- Window = 10 log lines per chunk
- Overlap = 3 lines shared between consecutive chunks

Why overlap? Without it, a root-cause sequence like:

```
Line 7:  INFO  Acquiring DB connection (pool 99/100)
Line 8:  WARN  Pool at 99% capacity
Line 9:  ERROR Connection timeout     ← chunk boundary
Line 10: ERROR Pool exhausted
```

would be split across two chunks. Line 9 lands in chunk 1, line 10 lands in chunk 2.
Neither chunk tells the full story. With 3-line overlap, both chunks contain
lines 7-9 AND 8-10 respectively — the context is always preserved.

Each chunk becomes a `LogChunk` object with a unique `chunk_id` (UUID).

---

#### Step 4 — Metadata extraction

**File:** `src/ingestion/metadata.py`

`extract_metadata()` scans each chunk with regex patterns to pull out:

| Field | Extracted from | Example |
|---|---|---|
| `timestamp` | ISO 8601, nginx, epoch patterns | `2024-01-15T14:28:05` |
| `severity` | `ERROR`, `WARN`, `INFO`, etc. | `LogSeverity.ERROR` |
| `trace_id` | `trace_id=`, `request_id=`, `X-Request-ID` | `abc-001` |
| `deployment_id` | `deploy_id=`, `git_sha=`, `version=` | `deploy-789` |
| `host` | `host=`, `hostname=`, `pod=` | `payment-pod-1` |

These fields go into the chunk's `metadata` dict and are stored as filterable
fields in both Qdrant and Elasticsearch. Without them, a question like
"show me errors from payment-pod-1 between 2pm and 3pm" would require scanning
every document.

---

#### Step 5 — Embedding

**File:** `src/ingestion/embedder.py`

`embed_chunks()` runs each chunk's text through the BGE model:

```
"2024-01-15T14:28:05 ERROR payment-service Connection timeout..."
                      ↓
   BAAI/bge-large-en-v1.5  (1024-dimensional embedding)
                      ↓
  [0.023, -0.441, 0.112, ..., 0.087]   ← 1024 floats
```

Two important implementation details:
1. **Instruction prefix** — BGE is instruction-tuned. Every chunk is prefixed
   with `"Represent this incident log for retrieval: "` before embedding.
   Without this, retrieval quality drops significantly.
2. **Batch processing** — all chunks from a log file are embedded in one
   `model.encode()` call with `batch_size=32`. Calling encode one-by-one
   would be ~30x slower.

---

#### Step 6 — Dual write

**Files:** `src/ingestion/qdrant_store.py`, `src/ingestion/es_store.py`

The same chunk is written to two stores simultaneously:

**Qdrant** stores:
- The 1024-float vector (for semantic search)
- A payload with all metadata (for filtering)
- Point ID = `chunk_id` (UUID string)

**Elasticsearch** stores:
- The raw text content (for BM25 keyword search)
- Structured metadata fields with explicit types (`keyword`, `date`, `text`)
- Document ID = `chunk_id` (same UUID)

Using the same `chunk_id` in both is critical. When the hybrid retriever
merges results, it uses `chunk_id` to deduplicate — a document that appears
in both BM25 and vector results is one entry, not two.

**File:** `src/ingestion/pipeline.py`
The `ingest()` function orchestrates steps 1-6 in sequence:
```
clean → chunk → embed → qdrant.upsert() + es.index()
```

---

### Read Path (User Query)

#### Step 1 — Query planning

**Agent:** Planner Agent (`src/agents/planner.py`)

The raw user question:
> "Why did checkout fail after the deployment at 4pm?"

Is structured into a typed query plan:
```json
{
  "intent": "root_cause_analysis",
  "services": ["checkout-service"],
  "time_range": {"start": "2024-01-15T16:00:00", "end": "2024-01-15T16:15:00"},
  "severity_filter": ["ERROR", "CRITICAL"],
  "sub_questions": [
    "What errors occurred in checkout-service between 16:00 and 16:10?",
    "Was there a deployment in checkout-service around 16:00?",
    "Did the error rate spike after the deployment?"
  ]
}
```

Why structured output? Because the next agent (Retriever) needs to translate
this into precise Elasticsearch and Qdrant filter parameters.
Unstructured text cannot be reliably parsed.

---

#### Step 2 — Hybrid retrieval

**Files:** `src/retrieval/bm25_retriever.py`, `src/retrieval/vector_retriever.py`,
`src/retrieval/hybrid_retriever.py`

For each `sub_question`, two searches run **concurrently** via `asyncio.gather`:

**BM25 search (Elasticsearch)**
Finds chunks that contain the exact keywords in the question.
Good for: error codes (`OOMKilled`, `NullPointerException`), service names,
trace IDs, exact error messages.

```
Query: "checkout-service errors after deployment 16:00"
Filter: service_name="checkout-service", timestamp >= "16:00"
Result: top-20 chunks ranked by TF-IDF keyword relevance
```

**Vector search (Qdrant)**
Embeds the query into a 1024-float vector, then finds the nearest chunk
vectors by cosine similarity.
Good for: paraphrased questions, conceptual matches
("service went down" matches "503 Service Unavailable").

```
Query vector: embed("checkout-service errors after deployment 16:00")
Filter: service_name="checkout-service"  (applied before ANN search)
Result: top-20 chunks by cosine similarity
```

**RRF fusion** (`hybrid_retriever.py`)
Both result lists are merged with Reciprocal Rank Fusion:

```
score(chunk) = 1/(60 + rank_in_bm25) + 1/(60 + rank_in_vector)
```

A chunk at rank #3 in BM25 and rank #5 in vector search scores higher
than one that appears only in one list, even if it ranked #1 there.
This handles the case where neither search alone finds the best evidence.

---

#### Step 3 — Reranking

**File:** `src/retrieval/reranker.py`

The top-20 RRF candidates are passed to the cross-encoder model
`BAAI/bge-reranker-large`, which scores each `(query, chunk)` pair
by reading both texts together:

```
Input:  ("Why did checkout fail?", "16:00:26 CRITICAL checkout-service error rate 78%...")
Output: 0.97   ← high relevance score

Input:  ("Why did checkout fail?", "16:00:00 INFO deployment started version=v2.3.1...")
Output: 0.71   ← moderate relevance
```

The top-5 scored chunks become the evidence set for the agents.

Why not run the cross-encoder on all chunks?
The cross-encoder is O(n) — scoring 10,000 chunks would take ~100 seconds.
The bi-encoder (step 2) narrows candidates to 20 cheaply. The cross-encoder
then makes a precise final selection from that small set.

---

#### Step 4 — Reasoning

**Agent:** Reasoning Agent (`src/agents/reasoning.py`)

The agent receives the top-5 chunks and the original question.
It builds a timeline and writes a draft RCA constrained to the evidence:

```
Timeline:
  16:00:00 — Deployment of v2.3.1 started (deploy_id=deploy-789)
  16:00:20 — v2.3.1 pod began serving traffic
  16:00:22 — First NullPointerException in CartSerializer.toJson()
  16:00:25 — Error rate reached 45% (threshold: 5%)
  16:00:26 — Error rate reached 78% → rollback triggered
  16:00:40 — Traffic restored to v2.3.0
  16:00:45 — Error rate returned to 0.3%

Root Cause:
  Deployment v2.3.1 introduced a NullPointerException in CartSerializer.toJson()
  triggered when processing cart items with null discount fields.
  The error only surfaced under real traffic (not caught in staging).
```

The system prompt enforces evidence-only reasoning:
`"Identify root cause based ONLY on the log chunks provided."`

---

#### Step 5 — Critic validation

**Agent:** Critic Agent (`src/agents/critic.py`)

Every factual claim in the RCA is checked against the source chunks.
If the Reasoning Agent says "error rate reached 78%" but no chunk contains that
number, the Critic flags it as `UNSUPPORTED`.

Supported claims proceed to the Citation Agent.
Unsupported claims trigger the Reflection Agent.

---

#### Step 6 — Reflection and retry

**Agent:** Reflection Agent (`src/agents/reflection.py`)

Unsupported claims become new retrieval queries:

```
Unsupported claim: "v2.3.1 had null discount fields"
New query: "checkout-service v2.3.1 null discount NullPointerException"
→ retry Retriever with this query
```

Up to 3 retry attempts. After 3 failures, the Citation Agent marks
those claims with a low-confidence flag.

---

#### Step 7 — Citation

**Agent:** Citation Agent (`src/agents/citation.py`)

The final response is formatted with inline evidence:

```
The root cause of the 16:00 outage was deployment v2.3.1 [1], which introduced
a NullPointerException in CartSerializer.toJson() [2]. The error rate reached
78% within 6 seconds of the new version receiving traffic [3], triggering an
automatic rollback to v2.3.0 [4]. Recovery completed by 16:00:45 [5].

Citations:
[1] 16:00:00 INFO checkout-service Deployment started version=v2.3.1 deploy_id=deploy-789
[2] 16:00:22 ERROR checkout-service NullPointerException in CartSerializer.toJson()
[3] 16:00:26 CRITICAL checkout-service Error rate 78% — triggering automatic rollback
[4] 16:00:27 INFO checkout-service Rollback initiated v2.3.1 → v2.3.0
[5] 16:00:45 INFO checkout-service Error rate 0.3% — back to normal
```

---

### Observability

**File:** `src/observability/logger.py`

Every request through the system emits structured JSON logs via structlog:

```json
{"event": "ingest_complete", "chunks": 47, "service": "payment-service", "level": "info", "timestamp": "2024-01-15T14:28:10Z"}
{"event": "retrieval_done", "bm25_count": 20, "vector_count": 20, "reranked_count": 5, "level": "info"}
{"event": "eval_run_done", "answer_length": 412, "contexts_count": 5, "retry_count": 1, "level": "info"}
```

`get_logger(__name__)` returns a bound logger with the module name attached —
logs from each layer are identifiable without adding boilerplate to every call.

LangSmith traces every LLM call automatically when `LANGCHAIN_TRACING_V2=true`
is set in `.env`. No code changes are needed — the LangChain integration
instruments all ChatOpenAI calls transparently.

---

### Evaluation

**Files:** `src/evaluation/ground_truth.py`, `src/evaluation/pipeline.py`,
`src/evaluation/metrics.py`

**Script:** `scripts/evaluate.py`

The evaluation pipeline measures four RAG quality dimensions using Ragas:

| Metric | What it measures | Threshold |
|---|---|---|
| `faithfulness` | Are all claims in the answer supported by retrieved chunks? | ≥ 0.80 |
| `answer_relevancy` | Does the answer address the actual question? | ≥ 0.75 |
| `context_precision` | Are retrieved chunks relevant (low noise)? | ≥ 0.70 |
| `context_recall` | Did retrieval find all the needed evidence? | ≥ 0.70 |

Five hand-written ground truth scenarios cover: DB connection pool exhaustion,
OOMKill restart loops, deployment-triggered error spikes, cross-scenario
correlation, and an out-of-scope negative test.

Run before every change to agent prompts, retrieval parameters, or chunking strategy:

```bash
python scripts/evaluate.py              # run all 5 scenarios
python scripts/evaluate.py --question 1 # run only scenario 1 (1-indexed)
python scripts/evaluate.py --dry-run    # print ground truth without running agents
```

---

## File Map — Where Everything Lives

```
src/
├── config.py                   Settings loaded from .env (pydantic_settings)
│
├── ingestion/
│   ├── models.py               RawLog, LogChunk, LogSeverity data models
│   ├── cleaner.py              Strip noise from raw logs
│   ├── chunker.py              Sliding window chunking with overlap
│   ├── metadata.py             Regex extraction of structured fields
│   ├── embedder.py             BGE embedding model (batch, normalised)
│   ├── pipeline.py             Orchestrates the full write path
│   ├── qdrant_store.py         Qdrant vector upsert + filtered search
│   └── es_store.py             Elasticsearch bulk index + BM25 search
│
├── retrieval/
│   ├── bm25_retriever.py       Keyword search wrapper
│   ├── vector_retriever.py     Semantic search (embeds query → Qdrant)
│   ├── hybrid_retriever.py     Parallel BM25 + vector → RRF merge
│   ├── reranker.py             Cross-encoder final ranking
│   └── retrieval_engine.py     Single entry point: hybrid → rerank
│
├── agents/
│   ├── state.py                AgentState TypedDict shared across all nodes
│   ├── planner.py              Structures query into typed plan
│   ├── retriever.py            Calls retrieval_engine as a Tool
│   ├── reasoning.py            Builds timeline and draft RCA
│   ├── critic.py               Validates claims against evidence
│   ├── reflection.py           Retry logic and route decisions
│   ├── citation.py             Formats final answer with citations
│   └── graph.py                Wires all agents into LangGraph state machine
│
├── api/
│   ├── app.py                  App factory, middleware registration, LangSmith wiring
│   ├── routes/
│   │   ├── ingest.py           POST /api/v1/ingest
│   │   └── query.py            POST /api/v1/query, WS /api/v1/query/stream
│   └── middleware.py           Request ID injection
│
├── observability/
│   └── logger.py               structlog JSON logger setup + get_logger()
│
└── evaluation/
    ├── ground_truth.py         5 hand-written (question, answer, expected_contexts) entries
    ├── pipeline.py             Runs agent_graph per entry, collects Ragas inputs
    └── metrics.py              Ragas scoring + threshold checks + print_report()

scripts/
├── seed_sample_logs.py         3 realistic incident scenarios for testing
└── evaluate.py                 CLI: --question N, --dry-run flags

main.py                         Uvicorn entry point — imports create_app()

docs/
├── BRD.md                      Business requirements (what and why)
├── FRD.md                      Functional requirements (how it behaves)
├── ARCHITECTURE.md             High-level system diagram
├── AGENTFLOW.md                Agent-to-agent flow
├── DATAFLOW.md                 Original data flow notes
├── TECHSTACK.md                Technology choices
├── IMPLEMENTATION_GUIDE.md     Phase-by-phase build guide with code patterns
└── PROJECT_OVERVIEW.md         ← this file
```

---

## Technology Choices — Why Each Tool

| Technology | Role | Why this one |
|---|---|---|
| **FastAPI** | API layer | Async-native, auto OpenAPI docs, Pydantic integration |
| **Pydantic** | Data validation | Catches bad data at system boundaries, not deep in logic |
| **Elasticsearch** | BM25 keyword search | Industry standard, exact-match filters are fast and cached |
| **Qdrant** | Vector storage | Supports filtered HNSW (filter before ANN, not after) |
| **BAAI/bge-large-en-v1.5** | Embeddings | Top open model on MTEB benchmark; runs locally, no API cost |
| **BAAI/bge-reranker-large** | Cross-encoder reranking | Sees query+document together; far more accurate than bi-encoder |
| **LangGraph** | Agent orchestration | State machine with typed state, retry loops, conditional edges |
| **structlog** | Structured logging | JSON output per module; `get_logger()` returns bound loggers |
| **LangSmith** | LLM call tracing | Zero-code instrumentation; traces every ChatOpenAI call automatically |
| **Ragas** | RAG evaluation | Faithfulness/hallucination + retrieval quality metrics against ground truth |

---

## Key Concepts to Understand

### Why two databases?

BM25 (Elasticsearch) and vector search (Qdrant) complement each other:

- Ask `"OOMKilled"` → BM25 wins (exact string match)
- Ask `"the container ran out of memory"` → vector search wins (semantic match)
- Neither alone handles both cases well

Running both and fusing results with RRF gives you the best of both worlds.

### Why chunk with overlap?

Log events are not independent. An `ERROR` on line 50 was caused by something
on line 47. Splitting logs into non-overlapping windows breaks causality.
3-line overlap guarantees the cause-and-effect sequence always appears together
in at least one chunk.

### Why a critic agent?

LLMs hallucinate. Given context about a payment timeout, a model might
confidently state "the database was overloaded" even if no log line says that.
The Critic Agent treats the LLM's output as a claim to be verified, not a fact
to be trusted. Every claim must be traceable to a specific chunk.

### Why rerank instead of just returning the top vector results?

The bi-encoder (embedding model) generates vectors independently for query and
document. It never sees them together, so it measures approximate semantic
similarity. The cross-encoder sees `(query, document)` as a single input and
models their exact relationship — much more accurate, but too slow to run on
every stored document. The two-stage approach (retrieve 20 cheaply, rerank 5
accurately) hits the right cost-quality tradeoff for a < 2s retrieval SLA.

### Why evaluate with Ragas instead of just testing manually?

Manual testing finds bugs you thought of in advance. Ragas measures the four
dimensions that matter for RAG systems — hallucination rate, answer relevancy,
retrieval precision, and retrieval recall — objectively and reproducibly.
A score drop after changing a prompt or retrieval parameter tells you exactly
what regressed and by how much, before it reaches production.
