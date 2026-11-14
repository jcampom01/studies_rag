# Step 3 · Embeddings

**English** · [Español](03-embeddings.es.md) · [← Chunking](02-chunking.md) · [Next: Vector store →](04-vector-store.md)

Turns every chunk into a **vector**: a list of numbers that represents its meaning, so that texts about the same thing end up close to each other.

```
chunks_<chunker>.jsonl  ──embed──►  data/processed/embeddings/<chunker>/
                                      ├── vectors.npy       float32 matrix, one row per chunk
                                      ├── chunk_ids.json    the chunk_id of each row
                                      └── manifest.json     how the vectors were made
```

## The model

The default is [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), run locally with `sentence-transformers`:

- **1,024 dimensions**, and a context of up to 32k tokens (much more than a chunk needs).
- **Multilingual**: a Spanish question matches an English passage and vice versa.
- **Instruction-aware** (see below).
- **Small**: about 1.2 GB on the GPU.

It was chosen from the MTEB multilingual retrieval leaderboard, among the models that fit on the GPU next to an LLM.

The model is **pinned to a commit** of its Hugging Face repository (`revision`). Without that, an update on the Hub could change the vectors silently, and the stored vectors would no longer match new queries.

What happens to a text:

```
text ─► tokenizer (≈151k-token vocabulary) ─► Transformer ─► last-token pooling ─► L2 normalization ─► vector (1024)
```

With normalized vectors, cosine similarity is a dot product.

## What gets embedded

### Chunks: a context header + the text

```
<collection_name> › Tema <doc_number>: <doc_title> › <section>

<chunk text>
```

A chunk often doesn't say which subject or topic it belongs to: "the algorithm stops when…" could be about any course. The header adds that context, built from the metadata with `context_template`. The header is only used for embedding; the stored text of the chunk doesn't change.

### Questions: an instruction + the question

```
Instruct: Given a student's question, retrieve the passages of the course notes that answer it
Query: <question>
```

The search is **asymmetric**: a question and the passage that answers it don't look alike, since one is short and asks, the other is long and explains. Instruction-aware models are trained to map an *instructed query* close to its answer. The instruction goes **only on the query**; documents are embedded as they are (`document_prompt: null`).

The instruction is configuration, so it is part of the embedder settings. A query embedded with a different instruction is not comparable with the stored vectors.

## The manifest: skip work and catch mismatches

`manifest.json` stores everything that determines the vectors:

```json
{
  "embedder": "sentence_transformers",
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
  "query_prompt": "Instruct: ...\nQuery:",
  "document_prompt": null,
  "normalized": true,
  "context_template": "{collection_name} › Tema {doc_number}: {doc_title} › {section}",
  "chunks_file": "data/processed/chunks_structured.jsonl",
  "chunks_sha256": "…",
  "dimension": 1024,
  "count": 2665,
  "created_at": "2026-11-09T12:08:29+00:00"
}
```

It is used in three ways:

- **Skip if unchanged.** If the manifest matches the current settings and the SHA-256 of the chunks file, `embed` does nothing. Running it again only recomputes what changed.
- **Compatibility check.** Before searching, `check_compatible` compares the manifest with the embedder that will embed the question. A different model, revision or instruction fails loudly instead of returning meaningless results.
- **Complete or absent.** The old manifest is deleted first and the new one is written last, after the vectors and the ids. A run that fails halfway leaves a folder without a manifest, which counts as "not embedded", never as a half-written set.

The contract every embedder implements:

```python
class Embedder(ABC):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...   # (n, dim), normalized float32
    def embed_query(self, text: str) -> np.ndarray: ...              # (dim,)
    def describe(self) -> dict: ...                                  # settings, stored in the manifest
```

## What was measured

On the development corpus:

- **Tokens.** About 3.85 characters per token. Tokens per chunk: median 294, p95 371, max 548, well within the model's context.
- **The instruction** raised the scores of the right passages and changed the top-1 result in 2 of 5 test questions.
- **The context header** improved topical precision in 4 of 5 questions. The share of chunks whose nearest neighbour comes from the same collection went from 95.9 % to 98.5 %.
- **Known weaknesses of dense embeddings**, measured as cosine similarity between pairs of sentences (an unrelated pair scores about 0.10):

  | Pair | Similarity | Problem |
  |---|---|---|
  | "algorithms that need labelled data" / "… that don't need …" | 0.90 | negation is almost invisible |
  | "topic 3 of …" / "topic 8 of …" | 0.94 | numbers barely count |
  | an acronym / the same acronym with two letters swapped | 0.66 | typos in acronyms |
  | an acronym / its expansion | 0.70 | the model knows some acronyms only partly |

  These are the usual motivations for metadata filters, keyword (hybrid) search and rerankers: see step 7 in the roadmap.

## Running it

```bash
PYTHONPATH=src .venv/bin/python -m studies_rag.embed
```

Settings: the `embed` section.

| Key | Meaning |
|---|---|
| `inputs` | chunker name → chunks file. Each one gets its own folder in `output_dir` |
| `context_template` | Header prepended to each chunk (`null` = no header). Uses metadata fields |
| `embedder`, `embedder_options` | `model`, `revision`, `query_prompt`, `document_prompt`, `batch_size`, `device` (`null` = GPU if available) |

The first run downloads the model to the Hugging Face cache (`~/.cache/huggingface`).

> **GPU note (WSL 2).** When the GPU memory is full, the driver doesn't fail: it falls back to system memory and becomes many times slower. If embedding is unusually slow, check `nvidia-smi` for other processes (an open notebook kernel, for example) holding GPU memory.

**Precision note.** The model computes in half precision, so the stored vectors have norms between 0.998 and 1.004 rather than exactly 1. A plain dot product is then off from the true cosine by up to about 0.003. PostgreSQL's cosine operator divides by the norms, so search results are exact.
