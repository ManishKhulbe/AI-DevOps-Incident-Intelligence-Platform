# System Design

## High-Level Architecture

```
Frontend (Next.js)
        ↓
API Gateway (FastAPI)
        ↓
Agent Orchestrator (LangGraph)
        ↓
    Hybrid Retrieval Layer
    ↓               ↓
ElasticSearch    Qdrant
(BM25)           (Vectors)
        ↓
     Reranker
        ↓
     LLM Layer
        ↓
Critic + Reflection Agents
        ↓
  Response + Citations
```
