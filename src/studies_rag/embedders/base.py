"""Contract between the pipeline and the embedding models."""
from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Turns texts into vectors. Documents and queries may be encoded differently
    (e.g. an instruction only on the query side), so they have separate methods."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """One normalized float32 vector per text: shape (len(texts), dimension)."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """One normalized float32 vector: shape (dimension,)."""

    @abstractmethod
    def describe(self) -> dict:
        """Every setting that changes the vectors (model, version, prompts...).

        It is stored next to the vectors, so a query can only be compared with
        vectors made by an embedder with the same description.
        """
