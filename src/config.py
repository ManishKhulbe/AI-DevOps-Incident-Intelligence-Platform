from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""

    # Storage
    elasticsearch_url: str = "http://localhost:9200"
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379"

    # Models
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-large"

    # Retrieval tuning
    top_k_retrieve: int = 20   # candidates fetched by hybrid search
    top_k_rerank: int = 5      # final results after cross-encoder reranking

    # LangSmith (optional — leave blank to disable tracing)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "devops-incident-agent"

    class Config:
        env_file = ".env"


settings = Settings()
