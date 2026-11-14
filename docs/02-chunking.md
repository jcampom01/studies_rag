# Step 2 · Chunking

**English** · [Español](02-chunking.es.md) · [← Ingestion](01-ingestion.md) · [Next: Embeddings →](03-embeddings.md)

Splits the pages into **chunks**: the pieces of text that get embedded, retrieved and shown to the LLM.

```
data/processed/pages.jsonl  ──chunk──►  data/processed/chunks_<chunker>.jsonl   (one file per chunker)
```

## Why chunking matters

A chunk is the unit of retrieval, so its size involves a trade-off:

- **Too big**: one vector has to represent several ideas and matches none of them well. It also uses up the LLM's context.
- **Too small**: the chunk loses the context that makes it understandable (a caption without its figure, a list item without its heading).

Where a chunk is cut matters too. A chunk that ends with a heading, or that starts in the middle of a sentence, is harder both to retrieve and to answer from.

The project includes **two chunkers on purpose**: a hand-written one that follows the document structure, and LangChain's standard splitter. Both run over the same pages, so they can be compared in every later step.

## Output: `chunks_<chunker>.jsonl`

```json
{
  "chunk_id": "9d18103009e45f5b",
  "text": "## 3.2. A section title\n\nFirst paragraph...\n\nSecond paragraph...",
  "metadata": {
    "collection": "course_a", "collection_name": "Course A",
    "doc_number": 3, "doc_title": "Topic title", "page_type": "Key ideas",
    "section": "3.2. A section title", "subsection": null,
    "source": "data/raw/course_a/topic_03.pdf",
    "pages": [6, 7]
  }
}
```

- **`chunk_id`** is the first 16 hexadecimal characters of `sha256(source + "\n" + text)`. It is stable: the same text from the same file always gets the same id, across runs and chunkers.
- **`pages`** is always a list, because a chunk can span pages.
- **The rest of the metadata** comes from the pages. When the pages of a chunk disagree on a field, the field becomes a list of the distinct values.

The contract every chunker implements:

```python
class Chunker(ABC):
    @abstractmethod
    def split(self, pages: Iterable[PageRecord]) -> Iterator[Chunk]: ...
```

## Chunker 1: `structured` (hand-written)

It follows the structure the adapter preserved: section, then paragraph, then sentence.

1. **Group the pages.** Consecutive pages of the same file and the same `group_by` fields (by default the section) form a group. A chunk can span pages, but it never mixes two sections.
2. **Glue paragraphs cut by a page break.** If the last paragraph of a page doesn't end with closing punctuation and the next page starts in lowercase, both are one paragraph.
3. **Pack whole paragraphs** into a chunk until adding the next one would exceed `target_chars`.
4. **Split paragraphs that are too long.** A paragraph over `max_chars` is split recursively: first by lines, then by sentences, then by words.
5. **Overlap.** Each chunk starts with the last `overlap_paragraphs` paragraphs of the previous one, so an idea at the boundary appears in both.
6. **Never end with a heading.** Trailing headings move to the start of the next chunk, and they replace the overlap there, since a heading opens a new topic.

| Option | Default | Meaning |
|---|---|---|
| `group_by` | `[]` | Metadata fields that define a group, besides the file. The example config uses `["section"]` |
| `target_chars` | 1200 | Chunks are filled up to this size |
| `max_chars` | 2000 | Paragraphs longer than this are split |
| `overlap_paragraphs` | 1 | Paragraphs repeated at the start of the next chunk |

A heading is a single line that starts with `#` to `######` and a space. The single-line condition matters: Python comments inside code listings (`# compute...`) also start with `#`, and they span several lines once the code block is kept together.

## Chunker 2: `langchain`

It is LangChain's `RecursiveCharacterTextSplitter`, used the usual way: each page is a `Document` and is split on its own, trying the separators `\n\n`, `\n`, `. `, ` ` and `""` in that order.

| Option | Default | Meaning |
|---|---|---|
| `chunk_size` | 1200 | Maximum characters per chunk |
| `chunk_overlap` | 200 | Characters repeated between consecutive chunks of a page |
| `separators` | see above | Split points, tried in order |

It doesn't know about sections or headings, and its chunks never cross a page boundary.

## Results on the development corpus

From 2,065 pages:

| | `structured` | `langchain` |
|---|---|---|
| Chunks | 2,665 | 2,922 |
| Chunks ending with a heading | **0** | 101 |
| Tiny chunks | 27 | 154 |
| Chunks starting mid-sentence | 39 | 376 |
| Chunks ending mid-sentence | 363 | 730 |
| Extra text from overlap | +19 % | +2 % |

- The structured chunker cuts in better places, because it knows where paragraphs and sections start.
- The price is more repeated text: a whole paragraph of overlap against 200 characters, which in practice is only used when a page is longer than `chunk_size`.
- Whether these differences actually improve **retrieval and answers** is what step 6 (evaluation) will measure.

Two bugs found while measuring:

- **Code comments taken as headings.** Before the single-line rule, a comment like `#cancer.data` inside a code listing was taken for a heading, which also produced an oversized chunk.
- **Headings at the end of chunks.** They went from 324 to 0 once the adapter marked headings in Markdown and the chunker moved trailing headings forward.

## Running it

```bash
PYTHONPATH=src .venv/bin/python -m studies_rag.chunk
```

Settings: the `chunk` section (`chunkers`, `input`, `output_dir`, `chunker_options.<name>`). For each chunker the script prints the number of chunks and the quartiles of their sizes.

To add a chunker, implement `Chunker.split`, register it in `chunkers/__init__.py`, and add its name to `chunkers` and its options to `chunker_options`. Steps 3 and 4 then need an entry for it in `embed.inputs`.
