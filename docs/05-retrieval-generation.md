# Step 5 · Retrieval + generation

**English** · [Español](05-retrieval-generation.es.md) · [← Vector store](04-vector-store.md) · [README](../README.md)

Answers a question: it retrieves the closest chunks from the vector store and lets a local LLM write the answer from them, citing the passages it used.

```
question ─► embed_query ─► store.search ─► prompt ─► LLM (Ollama) ─► answer + sources
```

## The LLM service: Ollama

`docker-compose.yml` runs `ollama/ollama:0.40.11` with access to the GPU. Downloaded models live in a named volume (`ollama`), so they survive the container being recreated.

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3.5:4b
```

### The model: `qwen3.5:4b`

- 4.2B parameters, quantized to Q4_K_M: about 3.7 GB on the GPU once loaded.
- Multilingual, so it answers well in the language of the question.

**Memory budget on an 8 GB GPU:**

| | GB |
|---|---|
| Desktop and other processes | ~1.4 |
| Embedding model | ~1.2 |
| LLM | ~3.7 |
| Free margin | ~1.9 |

A bigger model (8–9B) would need the query embedder on the CPU (`"device": "cpu"`); embedding one question on the CPU takes a fraction of a second.

### Two Ollama behaviours that matter for RAG

**1. Thinking mode is on by default.** This model "reasons" before answering, and for a one-sentence answer it wrote about 1,500 words of reasoning first. For RAG that is only latency. Requests send `"think": false`.

**2. Prompts that don't fit are cut silently.** We sent a 2,435-token prompt with smaller context sizes:

| `num_ctx` | Tokens Ollama read | HTTP status |
|---|---|---|
| 1024 | 1,026 | 200 |
| 2048 | 1,026 | 200 |
| 2400 | 1,202 | 200 |
| 2560 | 2,435 (all) | 200 |

When the prompt doesn't fit, Ollama drops a large part of it, keeping the beginning and the end, answers normally and reports a token count well below `num_ctx`. In RAG, the chunks in the middle of the prompt would disappear without a trace. Requests therefore send **`"truncate": false`**, which turns this into an error:

```
400 {"error":"request (2435 tokens) exceeds the available context size (2048 tokens), try increasing it"}
```

These options only exist in Ollama's **native API** (`/api/chat`), not in its OpenAI-compatible one. That is why the adapter uses the native API, through plain `httpx` calls.

## The prompt

Each question sends two messages.

**System**: fixed instructions, from `config.json`:

```
You answer questions using only the numbered context passages.
Cite the passages you use with their numbers, like [1] or [2][3].
If the context does not contain the answer, say so plainly instead of guessing.
Answer in the language of the question.
```

- *Only the passages*: without this, the model mixes in what it "knows", which for a small model is little and sometimes wrong.
- *Cite*: every statement can be checked against its source.
- *Say so*: this is what keeps a missing answer from becoming an invented one.

**User**: the retrieved chunks and the question:

```
Context:

[1] Course A › Tema 3: Topic title › 3.2. A section title (pages 6-7)
<chunk text>

[2] ...

Question: <question>
```

- **The passages are numbered from 1** and labelled with `source_template` plus their pages. The same numbers appear in the printed sources and in the citations.
- **They are ordered by score**, best first. LLMs pay most attention to the beginning and the end of the context.
- **The question comes last**, right before the model answers.
- **Scores are not shown** to the model.

## The pipeline: `rag.py`

```python
class RAG:
    def __init__(self, embedder, store, llm, index_name, k, system_prompt, source_template): ...
    def retrieve(self, question, filters=None) -> list[SearchResult]   # embed + search
    def build_prompt(self, question, results) -> str
    def answer(self, question, filters=None) -> Answer
```

- **It receives its parts already built** (dependency injection) and only knows their contracts. Nothing in it refers to PostgreSQL, Ollama or Hugging Face.
- **It fails at start-up, not at the first question.** The constructor checks that the chunk set exists in the store and that its manifest is compatible with the embedder. A different model or instruction would otherwise produce meaningless retrieval with no error at all.
- **`retrieve` is separate from `answer`**, so retrieval can be evaluated on its own, without calling the LLM.
- **The answer is a data object**, not printed text:

```python
@dataclass
class Answer:
    question: str
    text: str
    sources: list[SearchResult]   # numbered from 1, as in the prompt
    cited: list[int]              # passage numbers found in the answer: [1], [2][3], [1, 3]
    retrieval_seconds: float
    completion: Completion        # prompt_tokens, output_tokens, seconds
```

Citations that point to passages that don't exist (`[9]` when there were 5) are discarded.

The LLM contract:

```python
class LLM(ABC):
    def generate(self, system: str, prompt: str) -> Completion: ...
    def describe(self) -> dict: ...
```

## The terminal entry point: `ask.py`

It builds the pipeline from the configuration and answers the questions typed in:

- **The embedder** comes from the `embed` section, because it has to be the one that made the stored vectors.
- **The vector store** comes from the `index` section.
- **The LLM and the query settings** come from the `query` section.

An empty line or Ctrl+D quits. It also reads from a pipe:

```bash
echo "¿Qué diferencia hay entre bagging y boosting?" | PYTHONPATH=src .venv/bin/python -m studies_rag.ask
```

For each question it prints:

- the answer;
- every retrieved source with its score, marking the cited ones with `*`;
- the timing: retrieval (about 0.1 s) and LLM (a few seconds).

The first question is slower, because the embedding model and, if Ollama had unloaded it, the LLM are loaded then.

Sources that were retrieved but not cited are worth watching. If none is cited, the model probably said the answer isn't in the context.

## Configuration: the `query` section

| Key | Meaning |
|---|---|
| `index` | Which chunk set to search (`structured` or `langchain`) |
| `k` | Number of chunks put in the prompt |
| `system_prompt` | The fixed instructions |
| `source_template` | Label of each passage, built from metadata fields |
| `llm`, `llm_options` | `base_url`, `model`, `think`, `num_ctx`, `temperature` (`null` = the model's default), `timeout` |

`num_ctx` is 8,192 tokens: 5 chunks with their labels, the instructions and the question take about 2,500. Raising `num_ctx` costs GPU memory, and changing it makes Ollama reload the model.

`temperature` is 0.3, lower than the model's default (1.0), so the answers stick to the passages.

## Limitations

- **Retrieval sets the ceiling.** If the chunk with the answer is not among the `k` retrieved, the best the LLM can do is say it doesn't know.
- **"Only the passages" is an instruction, not a guarantee.** A small model sometimes adds knowledge of its own. The citations make that visible, and step 6 will measure it.
- **No conversation.** Each question is independent.
- **No metadata filters in the terminal yet**, although `retrieve` and `answer` accept them.
