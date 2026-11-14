# Step 1 · Ingestion

**English** · [Español](01-ingestion.es.md) · [← README](../README.md) · [Next: Chunking →](02-chunking.md)

Turns the source documents into **clean pages**: the useful text of each page, without headers, footers or covers, plus metadata saying where it comes from.

```
data/raw/<collection>/*.pdf  ──ingest──►  data/processed/pages.jsonl
```

## Output: `pages.jsonl`

One JSON object per line, one line per kept page:

```json
{
  "source": "data/raw/course_a/topic_03.pdf",
  "page": 6,
  "text": "## 3.2. A section title\n\nFirst paragraph of the page...\n\n- A list item...",
  "metadata": {
    "collection": "course_a",
    "collection_name": "Course A",
    "doc_number": 3,
    "doc_title": "Topic title",
    "page_type": "Key ideas",
    "section": "3.2. A section title",
    "subsection": null
  }
}
```

- `source` and `page` (1-based) identify the page. Every later step keeps them, so an answer can always point back to the PDF page.
- `text` is made of paragraphs separated by a blank line (`\n\n`). Headings get a Markdown prefix (`##` for sections, `###` for subsections) so the chunker can recognise them. Code listings keep their line breaks.
- `metadata` is free-form and decided by the adapter. Later steps copy it into every chunk and the database stores it as JSON, so it can be used in prompts and in search filters.

## How it works

Ingestion is split in two parts:

- **`ingest.py`**, the generic part. It finds the files, skips duplicates, asks an adapter for the pages, normalizes their text and writes the JSON Lines file. It knows nothing about how any document is laid out.
- **An adapter**, the document-specific part. It knows where the title of a document is, which text is noise and which pages are worth keeping.

```python
class Adapter(ABC):
    file_pattern: str = "*"   # glob of the files this adapter reads

    @abstractmethod
    def extract(self, path: Path) -> Iterator[PageRecord]:
        """Yield the pages of a document worth indexing, with their text and metadata."""
```

`ingest.py` also does three generic things:

- **It finds the files recursively** under `raw_dir`. The subfolder of each file is its *collection*.
- **It skips byte-identical duplicates** (same SHA-256) and prints a warning.
- **It applies Unicode NFKC normalization.** PDFs often contain ligatures (`ﬁ`) and decomposed accents (`a` + a combining accent), which look identical on screen but are different characters. They would make the same word embed and match differently. NFKC turns them into a single standard form.

## The bundled adapter: `master_notes`

It reads course notes exported to PDF from one fixed web template:

| Element | How the adapter finds it |
|---|---|
| Document number and collection name | On the cover page, by **font size** (each field has its own size) |
| Document title | First entry of the PDF's internal table of contents, with the file name as a fallback |
| Section / subsection of a page | The last TOC entry of level 1 / level 2 that starts on or before that page |
| Page type | A block in the page header with a specific font size. Pages without one (cover, index) are skipped |
| Header and footer | By vertical position: blocks above `y = 75 pt` or below `y = 790 pt` |
| Headings and code | Headings by font size (≥ 12 pt, ≥ 16 pt for sections); code by its 9 pt monospaced size |

### Rebuilding paragraphs

PyMuPDF returns a page as *blocks*, often one per visual line, so paragraphs have to be rebuilt. A block **starts a new paragraph** when:

- its kind (heading, code or text) or its font size changes;
- or the vertical gap to the previous block is larger than **15 pt**.

The 15 pt threshold was not guessed. A histogram of the gaps between consecutive blocks over the whole corpus is clearly bimodal: lines inside a paragraph are close together and gaps between paragraphs are larger. 15 pt sits between the two peaks.

Two special cases:

- **List bullets** (`▸`, `•`) come as separate blocks. They become `- ` and always join the text that follows them.
- **List items whose content was an image** (formulas, for instance) are left with a bare `-` and are dropped.

### Options

| Option | Meaning |
|---|---|
| `keep_page_types` | Page types to keep (the example config keeps only the main-content pages). `null` keeps every page that has a page-type header |

### How the template was analysed

Every threshold above comes from measuring the documents, not from guessing:

1. **Look**: render a few pages with their blocks drawn as boxes and coloured by font size.
2. **Dump**: print every block with its position and size.
3. **Guess** a rule, for example "the header ends at y = 75".
4. **Measure** the rule on the whole corpus: all the blocks of all the PDFs in a single table.
5. **Check the outliers**: the documents that don't fit the rule.

On the development corpus this meant 82 PDFs and about 74,000 text blocks. One file was a byte-identical duplicate, and the others gave 2,065 pages.

## Known limitations

- **Scaled pages.** A few pages are rendered at a slightly different scale (body text at 9.8–10.3 pt, headings at 16.6–18.8 pt), so some subsection headings there are not recognised.
- **Not all 9 pt text is code.** Some captions use the same size in a proportional font, so they are kept with their line breaks.
- **Several subsections on one page.** Section and subsection are assigned per page. When several subsections start on the same page, all its text gets the last one.
- **Formulas drawn as images** are lost: there is no OCR.

## Writing an adapter

To ingest other documents (another template, Word files, web pages…), write a class that yields `PageRecord`s and register it:

```python
# src/studies_rag/adapters/my_docs.py
from pathlib import Path
from collections.abc import Iterator
import pymupdf
from studies_rag.adapters.base import Adapter, PageRecord


class MyDocsAdapter(Adapter):
    file_pattern = "*.pdf"

    def __init__(self, min_chars: int = 50):   # options come from "adapter_options"
        self.min_chars = min_chars

    def extract(self, path: Path) -> Iterator[PageRecord]:
        with pymupdf.open(path) as doc:
            for page in doc:
                text = page.get_text().strip()
                if len(text) >= self.min_chars:
                    yield PageRecord(source=str(path), page=page.number + 1, text=text,
                                     metadata={"collection": path.parent.name, "title": doc.metadata.get("title")})
```

```python
# src/studies_rag/adapters/__init__.py
ADAPTERS = {"master_notes": MasterNotesAdapter, "my_docs": MyDocsAdapter}
```

```json
"ingest": {"adapter": "my_docs", "adapter_options": {"min_chars": 50}, ...}
```

If your metadata uses other field names, also update the settings that refer to them: `group_by` (`chunk` section), `context_template` (`embed` section) and `source_template` (`query` section).

## Running it

```bash
PYTHONPATH=src .venv/bin/python -m studies_rag.ingest
```

Settings: the `ingest` section of `config.json` (`adapter`, `adapter_options`, `raw_dir`, `output`). The output file is rewritten on every run.
