"""Step 4 of the pipeline: load the embeddings into the vector store.

For every input of the "embed" section, reads its embeddings folder (vectors,
chunk ids, manifest) and its chunks file, and stores them in the vector store
as one embedding set named after the input. A set whose stored manifest is
the same as the one on disk is skipped: nothing changed since it was loaded.

Settings: the "index" section (which vector store, its options) and the
"embed" section (where the chunks and the embeddings are).

Usage:
    PYTHONPATH=src .venv/bin/python -m studies_rag.index
"""
import time
from pathlib import Path

from studies_rag.config import load_config
from studies_rag.embed import file_sha256, load_embeddings, read_chunks
from studies_rag.vector_stores import VECTOR_STORES, VectorStore


def index(store: VectorStore, name: str, chunks_path: Path, folder: Path) -> None:
    vectors, chunk_ids, manifest = load_embeddings(folder)
    if file_sha256(chunks_path) != manifest["chunks_sha256"]:
        raise ValueError(f"{chunks_path} changed after it was embedded: run the embed step again")

    if store.manifest(name) == manifest:
        print(f"  up to date, skipped: {name}")
        return

    # Chunks in the same order as the rows of the vectors
    chunks_by_id = {chunk.chunk_id: chunk for chunk in read_chunks(chunks_path)}
    chunks = [chunks_by_id[chunk_id] for chunk_id in chunk_ids]

    start = time.perf_counter()
    store.replace(name, chunks, vectors, manifest)
    elapsed = time.perf_counter() - start
    print(f"  {len(chunks)} chunks x {vectors.shape[1]} dims stored as {name!r} in {elapsed:.1f} s")


def main() -> None:
    config = load_config("index")
    embed_config = load_config("embed")
    store_name = config["vector_store"]
    if store_name not in VECTOR_STORES:
        raise SystemExit(f"unknown vector store {store_name!r}; available: {', '.join(VECTOR_STORES)}")

    with VECTOR_STORES[store_name](**config.get("vector_store_options", {})) as store:
        for input_name, chunks_file in embed_config["inputs"].items():
            print(f"[{input_name}]")
            index(store, input_name, Path(chunks_file), Path(embed_config["output_dir"]) / input_name)


if __name__ == "__main__":
    main()
