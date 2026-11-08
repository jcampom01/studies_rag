"""Structure-aware chunking: sections -> paragraphs -> sentences.

1. Consecutive pages of the same file and group (e.g. the same section) are
   joined, so a chunk can span pages but never mixes two groups.
2. A paragraph cut by a page break is glued back together.
3. Whole paragraphs are packed into chunks of up to target_chars. A paragraph
   longer than max_chars is split by lines, then sentences, then words.
4. Each chunk starts with the last overlap_paragraphs paragraphs of the
   previous one, so an idea at the boundary is not lost.
5. A chunk never ends with a heading (a single-line paragraph starting with
   "# ", "## "..., Markdown style): trailing headings move to the start of the next chunk, replacing
   the overlap, since a heading opens a new topic.

Every chunk keeps the pages it comes from. Metadata fields with the same value
on all those pages stay as a single value; fields that differ become a list.

Options (from the config):
  group_by            metadata fields that define a group, besides the file
  target_chars        chunks are filled up to this size
  max_chars           paragraphs longer than this are split
  overlap_paragraphs  paragraphs repeated at the start of the next chunk
"""
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import groupby

from studies_rag.adapters.base import PageRecord
from studies_rag.chunkers.base import Chunk, Chunker, make_chunk_id

PARAGRAPH_SEP = "\n\n"
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
# A paragraph ending without one of these and followed by one starting in
# lowercase was cut by a page break
CLOSING_PUNCTUATION = (".", "!", "?", ":", ";")
# Markdown heading: "#" to "######", a space, and a single line (code comments span several lines)
HEADING_RE = re.compile(r"^#{1,6} [^\n]*$")


@dataclass
class Paragraph:
    text: str
    pages: list[int] = field(default_factory=list)

    @property
    def is_heading(self) -> bool:
        return HEADING_RE.match(self.text) is not None


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split a text longer than max_chars by lines, then sentences, then words."""
    if len(text) <= max_chars:
        return [text]
    for splitter, joiner in (("\n", "\n"), (SENTENCE_END_RE, " "), (" ", " ")):
        parts = splitter.split(text) if isinstance(splitter, re.Pattern) else text.split(splitter)
        if len(parts) > 1:
            return pack(parts, max_chars, joiner)
    # A single "word" longer than max_chars: cut it blindly
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def pack(parts: list[str], max_chars: int, joiner: str) -> list[str]:
    """Greedily join consecutive parts into pieces of up to max_chars."""
    pieces, current = [], ""
    for part in parts:
        for sub in split_long_text(part, max_chars):  # a part may itself be too long
            candidate = f"{current}{joiner}{sub}" if current else sub
            if len(candidate) <= max_chars:
                current = candidate
            else:
                pieces.append(current)
                current = sub
    if current:
        pieces.append(current)
    return pieces


def merge_metadata(records: list[PageRecord]) -> dict:
    """One value per field when all pages agree, a list of distinct values otherwise."""
    merged = {}
    for key in records[0].metadata:
        values = []
        for record in records:
            value = record.metadata.get(key)
            if value not in values:
                values.append(value)
        merged[key] = values[0] if len(values) == 1 else values
    return merged


class StructuredChunker(Chunker):
    def __init__(
        self,
        group_by: list[str] | None = None,
        target_chars: int = 1200,
        max_chars: int = 2000,
        overlap_paragraphs: int = 1,
    ):
        self.group_by = group_by or []
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.overlap_paragraphs = overlap_paragraphs

    def split(self, pages: Iterable[PageRecord]) -> Iterator[Chunk]:
        # groupby only joins *consecutive* records with the same key: pages
        # come in order, so a group is a run of pages of the same section
        for _, group in groupby(pages, key=self._group_key):
            records = list(group)
            paragraphs = self._paragraphs(records)
            for chunk_paragraphs in self._pack_paragraphs(paragraphs):
                yield self._make_chunk(chunk_paragraphs, records)

    def _group_key(self, page: PageRecord) -> tuple:
        return (page.source, *(str(page.metadata.get(key)) for key in self.group_by))

    def _paragraphs(self, records: list[PageRecord]) -> list[Paragraph]:
        """Paragraphs of a group, each with its page; glues paragraphs cut by a page break."""
        paragraphs: list[Paragraph] = []
        for record in records:
            for i, text in enumerate(t for t in record.text.split(PARAGRAPH_SEP) if t.strip()):
                previous = paragraphs[-1] if paragraphs else None
                cut_by_page_break = (
                    i == 0
                    and previous is not None
                    and not previous.text.rstrip().endswith(CLOSING_PUNCTUATION)
                    and text[:1].islower()
                )
                if cut_by_page_break:
                    joiner = "\n" if "\n" in previous.text else " "  # keep line breaks of code
                    previous.text = f"{previous.text}{joiner}{text}"
                    previous.pages.append(record.page)
                else:
                    paragraphs.append(Paragraph(text=text, pages=[record.page]))
        return paragraphs

    def _pack_paragraphs(self, paragraphs: list[Paragraph]) -> Iterator[list[Paragraph]]:
        """Group whole paragraphs into chunks of up to target_chars, with paragraph overlap."""
        current: list[Paragraph] = []
        n_overlap = 0  # how many leading paragraphs of current are repeated from the previous chunk
        for paragraph in self._split_long_paragraphs(paragraphs):
            has_new_content = len(current) > n_overlap
            if has_new_content and self._size(current + [paragraph]) > self.target_chars:
                # A chunk must not end with a heading: take trailing headings out
                headings = []
                while current and current[-1].is_heading:
                    headings.insert(0, current.pop())
                if len(current) > n_overlap:
                    yield current
                    # The headings start a new topic, so they replace the overlap
                    overlap = [] if headings else self._overlap(current, paragraph)
                    current, n_overlap = overlap + headings, len(overlap)
                else:
                    # Only the overlap was left before the headings: drop it
                    current, n_overlap = headings, 0
            current.append(paragraph)
        if len(current) > n_overlap:
            yield current

    def _overlap(self, chunk: list[Paragraph], next_paragraph: Paragraph) -> list[Paragraph]:
        """Last paragraphs of a chunk to repeat in the next one, unless they make it too big."""
        overlap = chunk[-self.overlap_paragraphs:] if self.overlap_paragraphs else []
        return overlap if self._size(overlap + [next_paragraph]) <= self.target_chars else []

    def _split_long_paragraphs(self, paragraphs: list[Paragraph]) -> Iterator[Paragraph]:
        for paragraph in paragraphs:
            if len(paragraph.text) <= self.max_chars:
                yield paragraph
            else:
                for piece in split_long_text(paragraph.text, self.target_chars):
                    yield Paragraph(text=piece, pages=list(paragraph.pages))

    @staticmethod
    def _size(paragraphs: list[Paragraph]) -> int:
        return sum(len(p.text) for p in paragraphs) + len(PARAGRAPH_SEP) * (len(paragraphs) - 1)

    @staticmethod
    def _make_chunk(paragraphs: list[Paragraph], records: list[PageRecord]) -> Chunk:
        pages = sorted({page for p in paragraphs for page in p.pages})
        source = records[0].source
        text = PARAGRAPH_SEP.join(p.text for p in paragraphs)
        metadata = merge_metadata([r for r in records if r.page in pages])
        return Chunk(
            chunk_id=make_chunk_id(source, text),
            text=text,
            metadata={**metadata, "source": source, "pages": pages},
        )
