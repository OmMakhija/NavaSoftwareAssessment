import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[str] = []

    def add(self, vectors: np.ndarray, texts: list[str]):
        if len(vectors) == 0:
            return
        self.index.add(vectors.astype(np.float32))
        self.chunks.extend(texts)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[str]:
        if self.index.ntotal == 0:
            return []
        q = query_vec.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.index.ntotal)
        _, indices = self.index.search(q, k)
        return [self.chunks[i] for i in indices[0] if i != -1]

    def reset(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks.clear()

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0
