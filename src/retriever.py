from src.embedder import Embedder
from src.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        vec = self.embedder.embed_single(query)
        return self.vector_store.search(vec, top_k)
