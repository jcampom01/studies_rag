from studies_rag.vector_stores.base import SearchResult, VectorStore
from studies_rag.vector_stores.pgvector import PgVectorStore

# Name used in the config -> class
VECTOR_STORES: dict[str, type[VectorStore]] = {
    "pgvector": PgVectorStore,
}

__all__ = ["VECTOR_STORES", "PgVectorStore", "SearchResult", "VectorStore"]
