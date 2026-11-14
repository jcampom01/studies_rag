"""Contract for vector stores: where embedded chunks are kept and searched.

A store holds named embedding sets. Each set keeps, per chunk, its vector,
text and metadata, plus the manifest describing how the vectors were made.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from studies_rag.chunkers import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity: 1 = same direction, higher = closer


class VectorStore(ABC):
    @abstractmethod
    def replace(self, name: str, chunks: list[Chunk], vectors: np.ndarray, manifest: dict) -> None:
        """Store an embedding set under `name`, replacing its previous content.

        Row i of `vectors` belongs to chunks[i]. Either the whole set is
        replaced or nothing changes.
        """

    @abstractmethod
    def manifest(self, name: str) -> dict | None:
        """Manifest stored with the set, or None if the set doesn't exist."""

    @abstractmethod
    def search(self, name: str, query_vector: np.ndarray, k: int = 5,
               filters: dict | None = None) -> list[SearchResult]:
        """The k chunks closest to `query_vector`, best first.

        `filters` maps metadata keys to the values they must have.
        """

    def close(self) -> None:
        """Release connections or other resources. Nothing to do by default."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
