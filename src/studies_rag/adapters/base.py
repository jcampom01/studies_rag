"""Contract between the generic ingestion and the document-specific adapters.

An adapter knows how one kind of document is laid out (where its title is,
what is noise, which pages matter). The ingestion only knows this contract.
"""
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


def normalize(text: str) -> str:
    """Unicode NFKC normalization.

    Makes visually identical strings byte-identical: ligatures ('ﬁ' -> 'fi'),
    decomposed accents ('a' + U+0301 -> 'á'), full-width characters, etc.
    """
    return unicodedata.normalize("NFKC", text)


@dataclass
class PageRecord:
    """One page of a document, ready for the next pipeline step (chunking)."""
    source: str                                    # path of the original file
    page: int                                      # 1-based page number
    text: str
    metadata: dict = field(default_factory=dict)   # adapter-specific fields (title, section...)


class Adapter(ABC):
    """Extracts the useful pages of one kind of document."""

    file_pattern: str = "*"  # glob pattern of the files this adapter can read

    @abstractmethod
    def extract(self, path: Path) -> Iterator[PageRecord]:
        """Yield the pages of a document worth indexing, with their text and metadata."""
