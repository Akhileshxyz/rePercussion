from __future__ import annotations

from typing import Iterable, List, Tuple


class FaissVectorStore:
    """A minimal FAISS wrapper for local vector search."""

    def __init__(self, dimension: int):
        import faiss  # lazy import

        self._faiss = faiss
        self._index = faiss.IndexFlatL2(dimension)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(self, vectors: List[List[float]]):
        import numpy as np

        arr = np.array(vectors, dtype="float32")
        if arr.ndim != 2 or arr.shape[1] != self._dimension:
            raise ValueError("Vectors must be 2D and match the index dimension")
        self._index.add(arr)

    def search(self, query_vectors: Iterable[List[float]], k: int = 5) -> Tuple[List[List[float]], List[List[int]]]:
        import numpy as np

        q = np.array(list(query_vectors), dtype="float32")
        distances, indices = self._index.search(q, k)
        return distances.tolist(), indices.tolist()


