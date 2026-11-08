"""Adapter for course notes exported as PDF with a fixed template.

Template:
  - page 1 is a cover with the document number and the collection name,
    each one identified by its font size
  - the PDF has an internal TOC whose first entry is "<number>. <title>"
  - every content page has a header naming the page type and a footer

Options (from the config):
  keep_page_types  page types to keep, e.g. ["Ideas clave"]; None keeps every
                   page that has a page-type header
"""
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from studies_rag.adapters.base import Adapter, PageRecord, normalize

# Cover page: font sizes of its fields
COVER_NUMBER_SIZE = 18.0
COVER_COLLECTION_SIZE = 19.8
SIZE_TOLERANCE = 0.5

# Content pages: vertical zones (in points, y grows downwards)
HEADER_MAX_Y = 75     # blocks starting above this are the page header
FOOTER_MIN_Y = 790    # blocks starting below this are the page footer
PAGE_TYPE_SIZE = 24.0  # font size of the header block naming the page type

# Content pages: typography used to rebuild paragraphs
HEADING_MIN_SIZE = 12.0   # section and subsection titles
CODE_SIZE = 9.0           # code listings: their line breaks are kept
PARAGRAPH_GAP = 15.0      # vertical gap (pt) between lines that starts a new paragraph
BULLET_MARKERS = {"▸", "•"}

COVER_NUMBER_RE = re.compile(r"^Tema\s+(\d+)$")
TOC_TITLE_RE = re.compile(r"^Tema\s+\d+\.\s*(.+)$")
FILENAME_TITLE_RE = re.compile(r"Tema_\d+\._(.+?)_esl-ES$")


@dataclass
class Block:
    """A text block of a page: its position, its largest font size and its lines."""
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in points, y grows downwards
    size: float
    lines: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.lines)


def text_blocks(page: pymupdf.Page) -> list[Block]:
    """Text blocks of a page in reading order, skipping images and empty blocks."""
    blocks = []
    for b in page.get_text("dict", sort=True)["blocks"]:
        if b["type"] != 0:  # 1 = image
            continue
        # The spans of a line are pieces of it with different styles: concatenate them
        lines = [normalize("".join(s["text"] for s in line["spans"])).strip() for line in b["lines"]]
        lines = [line for line in lines if line]
        if not lines:
            continue
        size = max(s["size"] for line in b["lines"] for s in line["spans"])
        blocks.append(Block(bbox=tuple(b["bbox"]), size=size, lines=lines))
    return blocks


@dataclass
class Paragraph:
    kind: str     # "heading", "code" or "text"
    size: float   # font size of its first block
    lines: list[str]
    bottom: float  # y1 of its last block, to measure the gap to the next one

    def render(self) -> str:
        return "\n".join(self.lines) if self.kind == "code" else " ".join(self.lines)


def block_kind(block: Block) -> str:
    if block.size >= HEADING_MIN_SIZE - SIZE_TOLERANCE:
        return "heading"
    if abs(block.size - CODE_SIZE) < SIZE_TOLERANCE:
        return "code"
    return "text"


def build_paragraphs(blocks: list[Block]) -> list[str]:
    """Merge the blocks of a page (often one per visual line) into paragraphs.

    A block starts a new paragraph when its kind or font size changes, when it
    is a bullet marker, or when the vertical gap to the previous block is larger
    than the normal line spacing. Consecutive code blocks are always merged.
    """
    paragraphs: list[Paragraph] = []
    for block in blocks:
        kind, y0, y1 = block_kind(block), block.bbox[1], block.bbox[3]
        if block.text in BULLET_MARKERS:
            # The marker is its own block; the item text comes next, on the same line
            paragraphs.append(Paragraph(kind="text", size=block.size, lines=["-"], bottom=y1))
            continue
        current = paragraphs[-1] if paragraphs else None
        starts_new = (
            current is None
            or kind != current.kind
            or abs(block.size - current.size) > SIZE_TOLERANCE
            or (kind != "code" and y0 - current.bottom > PARAGRAPH_GAP)
        )
        if starts_new:
            paragraphs.append(Paragraph(kind=kind, size=block.size, lines=list(block.lines), bottom=y1))
        else:
            current.lines.extend(block.lines)
            current.bottom = y1
    return [p.render() for p in paragraphs]


def section_at(toc: list, page_number: int) -> tuple[str | None, str | None]:
    """Section (TOC level 1) and subsection (TOC level 2) a page belongs to.

    It's the last entry of each level starting on or before the page.
    Assumes the TOC is in page order.
    """
    section, subsection = None, None
    for level, title, start_page in toc:
        if start_page > page_number:
            break
        if level == 1:
            section, subsection = normalize(title), None  # a new section resets the subsection
        elif level == 2:
            subsection = normalize(title)
    return section, subsection


class MasterNotesAdapter(Adapter):
    file_pattern = "*.pdf"

    def __init__(self, keep_page_types: list[str] | None = None):
        self.keep_page_types = keep_page_types

    def extract(self, path: Path) -> Iterator[PageRecord]:
        with pymupdf.open(path) as doc:
            doc_metadata = self._document_metadata(doc, path)
            toc = doc.get_toc()
            for page in doc:
                page_number = page.number + 1  # PyMuPDF numbers pages from 0
                header, body = self._split_page(page)
                page_type = self._page_type(header)
                if page_type is None or not self._keep(page_type) or not body:
                    continue
                section, subsection = section_at(toc, page_number)
                yield PageRecord(
                    source=str(path),
                    page=page_number,
                    text="\n\n".join(build_paragraphs(body)),
                    # A new dict per page: page-level fields must not leak into other pages
                    metadata={
                        **doc_metadata,
                        "page_type": page_type,
                        "section": section,
                        "subsection": subsection,
                    },
                )

    def _keep(self, page_type: str) -> bool:
        return self.keep_page_types is None or page_type in self.keep_page_types

    @staticmethod
    def _split_page(page: pymupdf.Page) -> tuple[list[Block], list[Block]]:
        """Split the blocks of a page into header and body, dropping the footer."""
        header, body = [], []
        for block in text_blocks(page):
            y0 = block.bbox[1]
            if y0 < HEADER_MAX_Y:
                header.append(block)
            elif y0 <= FOOTER_MIN_Y:
                body.append(block)
        return header, body

    @staticmethod
    def _page_type(header: list[Block]) -> str | None:
        """Page type named in the header, or None if the page has no such header (cover, index)."""
        for block in header:
            if abs(block.size - PAGE_TYPE_SIZE) < SIZE_TOLERANCE:
                return block.text
        return None

    def _document_metadata(self, doc: pymupdf.Document, path: Path) -> dict:
        number, collection_name = self._read_cover(doc[0])
        if number is None or collection_name is None:
            raise ValueError(f"{path}: cover page doesn't follow the template")
        return {
            "collection": path.parent.name,
            "collection_name": collection_name,
            "doc_number": number,
            "doc_title": self._read_title(doc.get_toc(), path),
        }

    @staticmethod
    def _read_cover(page: pymupdf.Page) -> tuple[int | None, str | None]:
        """Document number and collection name, identified on the cover by their font size."""
        number, collection_name = None, None
        for block in text_blocks(page):
            if abs(block.size - COVER_NUMBER_SIZE) < SIZE_TOLERANCE:
                if m := COVER_NUMBER_RE.match(block.text):
                    number = int(m.group(1))
            elif abs(block.size - COVER_COLLECTION_SIZE) < SIZE_TOLERANCE:
                collection_name = block.text
        return number, collection_name

    @staticmethod
    def _read_title(toc: list, path: Path) -> str:
        """Title from the first TOC entry; falls back to the filename when that entry is broken."""
        if toc and (m := TOC_TITLE_RE.match(normalize(toc[0][1]))):
            return m.group(1)
        if m := FILENAME_TITLE_RE.search(normalize(path.stem)):
            return m.group(1).replace("_", " ")
        raise ValueError(f"{path}: cannot find the document title")
