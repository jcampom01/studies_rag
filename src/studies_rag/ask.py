"""Ask questions about the documents from the terminal.

Builds the pipeline from the config: the embedder from the "embed" section
(the one that made the stored vectors), the vector store from "index", and
the LLM and the query settings from "query". Then answers the questions typed
in; an empty line ends.

Usage:
    PYTHONPATH=src .venv/bin/python -m studies_rag.ask
"""
from studies_rag.config import load_config
from studies_rag.embedders import EMBEDDERS
from studies_rag.llms import LLMS
from studies_rag.rag import RAG, Answer
from studies_rag.vector_stores import VECTOR_STORES


def print_answer(answer: Answer, rag: RAG) -> None:
    print(f"\n{answer.text}\n")
    print("Sources (* = cited):")
    for n, result in enumerate(answer.sources, start=1):
        mark = "*" if n in answer.cited else " "
        print(f" {mark}[{n}] {result.score:.3f}  {rag.source_label(result.chunk)}")
    c = answer.completion
    print(f"\n(retrieval {answer.retrieval_seconds:.2f} s, LLM {c.seconds:.1f} s, "
          f"{c.prompt_tokens} prompt tokens, {c.output_tokens} output tokens)")


def main() -> None:
    embed_config = load_config("embed")
    index_config = load_config("index")
    config = load_config("query")

    embedder = EMBEDDERS[embed_config["embedder"]](**embed_config.get("embedder_options", {}))
    llm = LLMS[config["llm"]](**config.get("llm_options", {}))

    with VECTOR_STORES[index_config["vector_store"]](**index_config.get("vector_store_options", {})) as store:
        rag = RAG(embedder, store, llm, index_name=config["index"], k=config["k"],
                  system_prompt=config["system_prompt"], source_template=config["source_template"])
        print(f"Asking {config['index']!r} (k={config['k']}) with {llm.describe()['model']}. Empty line to quit.")
        while True:
            try:
                question = input("\n> ").strip()
            except EOFError:
                break
            if not question:
                break
            print_answer(rag.answer(question), rag)


if __name__ == "__main__":
    main()
