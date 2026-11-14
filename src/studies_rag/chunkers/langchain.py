"""Chunking with LangChain's RecursiveCharacterTextSplitter, used the usual way.

Each page becomes a LangChain Document and is split on its own: chunks never
cross a page boundary, and pages are not grouped by section.

Options (from the config):
  chunk_size     maximum characters per chunk
  chunk_overlap  characters repeated between consecutive chunks of a page
  separators     split points, tried in order (paragraph, line, sentence, word)
"""
from collections.abc import Iterable, Iterator

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from studies_rag.adapters.base import PageRecord
from studies_rag.chunkers.base import Chunk, Chunker, make_chunk_id

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class LangChainChunker(Chunker):
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200, separators: list[str] | None = None):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or DEFAULT_SEPARATORS,
        )

    def split(self, pages: Iterable[PageRecord]) -> Iterator[Chunk]:
        for page in pages:
            document = Document(
                page_content=page.text,
                metadata={**page.metadata, "source": page.source, "pages": [page.page]},
            )
            # split_documents copies the document's metadata into every piece
            for piece in self.splitter.split_documents([document]):
                yield Chunk(
                    chunk_id=make_chunk_id(page.source, piece.page_content),
                    text=piece.page_content,
                    metadata=piece.metadata,
                )
