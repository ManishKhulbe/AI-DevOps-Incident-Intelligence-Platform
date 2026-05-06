# Data Flow

> This is extremely important.

## End-to-End Flow

### Step 1 — Log Ingestion

```
Logs
  ↓
Cleaner
  ↓
Chunker
  ↓
Metadata Extractor
  ↓
Embedding Generator
  ↓
Store in Qdrant + Elasticsearch
```

### Step 2 — User Query

Example:
> "Why did checkout fail after deployment?"

### Step 3 — Query Understanding

**Planner Agent** identifies:
- Services involved
- Timeline
- Deployment relevance

### Step 4 — Hybrid Retrieval

Retrieve:
- Deployment logs
- Incident tickets
- Monitoring alerts

### Step 5 — Reranking

Cross-encoder reranks top results.

### Step 6 — Reasoning Agent

Builds:
- Timeline
- Service dependency map
- Probable root cause analysis (RCA)

### Step 7 — Critic Agent

Checks:
- Unsupported claims
- Low-confidence conclusions

### Step 8 — Citation Agent

Attaches:
- Log references
- Timestamps
- Deployment IDs

### Step 9 — Final Response

Returns:
- Summary
- RCA
- Confidence score
- Citations
