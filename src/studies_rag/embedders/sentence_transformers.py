"""Embedder for any Hugging Face model supported by sentence-transformers.

Options (from the config):
  model            Hugging Face repository, e.g. "org/name"
  revision         commit hash of that repository: pins the exact model version,
                   so an update on the Hub can't silently change the vectors
  query_prompt     text prepended to queries (instruction), or None
  document_prompt  text prepended to documents, or None
  batch_size       texts encoded at once
  device           "cuda", "cpu"... or None to pick the GPU when there is one
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from studies_rag.embedders.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model: str,
        revision: str | None = None,
        query_prompt: str | None = None,
        document_prompt: str | None = None,
        batch_size: int = 16,
        device: str | None = None,
    ):
        self.model_name = model
        self.revision = revision
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self.batch_size = batch_size
        self.device = device
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Loaded on first use: describe() works without loading 1 GB of weights."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, revision=self.revision, device=self.device)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.document_prompt, show_progress_bar=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], self.query_prompt)[0]

    def _encode(self, texts: list[str], prompt: str | None, show_progress_bar: bool = False) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return vectors.astype(np.float32)

    def describe(self) -> dict:
        return {
            "embedder": "sentence_transformers",
            "model": self.model_name,
            "revision": self.revision,
            "query_prompt": self.query_prompt,
            "document_prompt": self.document_prompt,
            "normalized": True,
        }
