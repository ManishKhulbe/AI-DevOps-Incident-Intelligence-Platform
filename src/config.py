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