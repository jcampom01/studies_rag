"""Step 5 of the pipeline: retrieval-augmented generation.

Given a question: embed it, retrieve the closest chunks from the vector store,
put them in a prompt as numbered passages and let the LLM answer from them,
citing the passages it used.

The pipeline only knows the contracts (Embedder, VectorStore, LLM): which
implementations it gets is decided by whoever builds it.
"""
import re
import time
from dataclasses import dataclass

from studies_rag.chunkers import Chunk
from studies_rag.embed import check_compatible
from studies_rag.embedders import Embedder
from studies_rag.llms import LLM, Completion
from studies_rag.vector_stores import SearchResult, VectorStore

# [1], [2][3] or [1, 3]
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Answer:
    question: str
    text: str                     # the LLM's answer
    sources: list[SearchResult]   # retrieved chunks, numbered from 1 as in the prompt
    cited: list[int]              # passage numbers the answer cites
    retrieval_seconds: float      # embedding the question + searching
    completion: Completion        # tokens and time of the LLM call


def page_range(pages: list[int] | int | None) -> str:
    if not pages:
        return ""
    pages = pages if isinstance(pages, list) else [pages]
    first, last = min(pages), max(pages)
    return f"page {first}" if first == last else f"pages {first}-{last}"


def cited_numbers(text: str, n_passages: int) -> list[int]:
    """Passage numbers cited in the text, ignoring numbers that aren't passages."""
    numbers = {int(n) for group in CITATION_RE.findall(text) for n in group.split(",")}
    return sorted(n for n in numbers if 1 <= n <= n_passages)


class RAG:
    def __init__(self, embedder: Embedder, store: VectorStore, llm: LLM, index_name: str, k: int,
                 system_prompt: str, source_template: str):
        # Fail now, not at the first question, if the set is missing or was embedded differently
        manifest = store.manifest(index_name)
        if manifest is None:
            raise ValueError(f"{index_name!r} is not in the vector store: run the index step")
        check_compatible(manifest, embedder)

        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.index_name = index_name
        self.k = k
        self.system_prompt = system_prompt
        self.source_template = source_template

    def retrieve(self, question: str, filters: dict | None = None) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(question)
        return self.store.search(self.index_name, query_vector, k=self.k, filters=filters)

    def source_label(self, chunk: Chunk) -> str:
        """Where a chunk comes from, built from its metadata."""
        label = self.source_template.format(**chunk.metadata)
        pages = page_range(chunk.metadata.get("pages"))
        return f"{label} ({pages})" if pages else label

    def build_prompt(self, question: str, results: list[SearchResult]) -> str:
        passages = [f"[{n}] {self.source_label(result.chunk)}\n{result.chunk.text}"
                    for n, result in enumerate(results, start=1)]
        return "Context:\n\n" + "\n\n".join(passages) + f"\n\nQuestion: {question}"

    def answer(self, question: str, filters: dict | None = None) -> Answer:
        start = time.perf_counter()
        results = self.retrieve(question, filters)
        retrieval_seconds = time.perf_counter() - start

        completion = self.llm.generate(self.system_prompt, self.build_prompt(question, results))
        return Answer(
            question=question,
            text=completion.text,
            sources=results,
            cited=cited_numbers(completion.text, len(results)),
            retrieval_seconds=retrieval_seconds,
            completion=completion,
        )
