from studies_rag.embedders.base import Embedder
from studies_rag.embedders.sentence_transformers import SentenceTransformerEmbedder

# Name used in the config -> embedder class
EMBEDDERS: dict[str, type[Embedder]] = {
    "sentence_transformers": SentenceTransformerEmbedder,
}

__all__ = ["EMBEDDERS", "Embedder"]
