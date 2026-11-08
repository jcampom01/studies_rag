"""Contract between the generic chunking step and the chunking strategies."""
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from studies_rag.adapters.base import PageRecord


@dataclass
class Chunk:
    """The unit that gets embedded, stored and retrieved."""
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)  # page metadata + "source" and "pages"


def make_chunk_id(source: str, text: str) -> str:
    """Stable id: the same text from the same file always gets the same id.

    Re-running the pipeline then updates chunks in the vector store instead of
    duplicating them.
    """
    return hashlib.sha256(f"{source}\n{text}".encode("utf-8")).hexdigest()[:16]


class Chunker(ABC):
    """Turns the page records of a corpus into chunks."""

    @abstractmethod
    def split(self, pages: Iterable[PageRecord]) -> Iterator[Chunk]:
        """Yield the chunks of the corpus. Pages come in file order, then page order."""
