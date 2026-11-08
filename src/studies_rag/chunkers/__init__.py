from studies_rag.chunkers.base import Chunk, Chunker
from studies_rag.chunkers.langchain import LangChainChunker
from studies_rag.chunkers.structured import StructuredChunker

# Name used in the config -> chunker class
CHUNKERS: dict[str, type[Chunker]] = {
    "structured": StructuredChunker,
    "langchain": LangChainChunker,
}

__all__ = ["CHUNKERS", "Chunk", "Chunker"]
