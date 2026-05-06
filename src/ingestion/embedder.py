from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    # BGE requires a specific instruction prefix for retrieval tasks
    prefixed = [f"Represent this incident log for retrieval: {c}" for c in chunks]
    return model.encode(prefixed, batch_size=32, normalize_embeddings=True).tolist()