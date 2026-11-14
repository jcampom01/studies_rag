"""Step 3 of the pipeline: embeddings.

Embeds every chunk file listed in the config and stores, per file, a folder:
  vectors.npy     float32 matrix, row i = vector of chunk i
  chunk_ids.json  chunk_id of each row (the link back to the chunks file)
  manifest.json   how the vectors were made (embedder settings, template,
                  hash of the chunks file); written last, so it only exists
                  when the vectors are complete

A folder whose manifest matches the current settings is skipped: running the
step again only recomputes what changed.
Settings come from the "embed" section of the config file.

Usage:
    PYTHONPATH=src .venv/bin/python -m studies_rag.embed
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from studies_rag.chunkers import Chunk
from studies_rag.config import load_config
from studies_rag.embedders import EMBEDDERS, Embedder

VECTORS_FILE = "vectors.npy"
IDS_FILE = "chunk_ids.json"
MANIFEST_FILE = "manifest.json"


def embedding_text(chunk: Chunk, template: str | None) -> str:
    """Text that gets embedded: a header built from the metadata (if any) + the chunk text."""
    if not template:
        return chunk.text
    return f"{template.format(**chunk.metadata)}\n\n{chunk.text}"


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(embedder: Embedder, chunks_path: Path, template: str | None) -> dict:
    """Everything that determines the vectors. Same fingerprint = same vectors."""
    return {
        **embedder.describe(),
        "context_template": template,
        "chunks_file": str(chunks_path),
        "chunks_sha256": file_sha256(chunks_path),
    }


def read_manifest(folder: Path) -> dict | None:
    path = folder / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def is_up_to_date(folder: Path, expected: dict) -> bool:
    manifest = read_manifest(folder)
    return manifest is not None and all(manifest.get(k) == v for k, v in expected.items())


def embed(embedder: Embedder, chunks_path: Path, template: str | None, folder: Path) -> None:
    expected = fingerprint(embedder, chunks_path, template)
    if is_up_to_date(folder, expected):
        print(f"  up to date, skipped: {folder}")
        return

    chunks = read_chunks(chunks_path)
    texts = [embedding_text(chunk, template) for chunk in chunks]

    start = time.perf_counter()
    vectors = embedder.embed_documents(texts)
    elapsed = time.perf_counter() - start

    folder.mkdir(parents=True, exist_ok=True)
    # Remove the old manifest first: if writing fails halfway, the folder is not "up to date"
    (folder / MANIFEST_FILE).unlink(missing_ok=True)
    np.save(folder / VECTORS_FILE, vectors)
    (folder / IDS_FILE).write_text(json.dumps([c.chunk_id for c in chunks]), encoding="utf-8")
    manifest = {
        **expected,
        "dimension": int(vectors.shape[1]),
        "count": len(chunks),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    norms = np.linalg.norm(vectors, axis=1)
    print(f"  {vectors.shape[0]} vectors x {vectors.shape[1]} dims in {elapsed:.0f} s -> {folder}")
    print(f"  norms between {norms.min():.3f} and {norms.max():.3f}, {vectors.nbytes / 1e6:.1f} MB")


def load_embeddings(folder: Path) -> tuple[np.ndarray, list[str], dict]:
    """Vectors, chunk ids and manifest of a folder written by embed()."""
    manifest = read_manifest(folder)
    if manifest is None:
        raise FileNotFoundError(f"no complete embeddings in {folder} (missing {MANIFEST_FILE})")
    vectors = np.load(folder / VECTORS_FILE)
    chunk_ids = json.loads((folder / IDS_FILE).read_text(encoding="utf-8"))
    return vectors, chunk_ids, manifest


def check_compatible(manifest: dict, embedder: Embedder) -> None:
    """Fail if a query embedded by this embedder can't be compared with these vectors."""
    current = embedder.describe()
    different = {k: (manifest.get(k), v) for k, v in current.items() if manifest.get(k) != v}
    if different:
        details = "; ".join(f"{k}: stored {old!r}, current {new!r}" for k, (old, new) in different.items())
        raise ValueError(f"embedder settings differ from the stored vectors ({details})")


def main() -> None:
    config = load_config("embed")
    name = config["embedder"]
    if name not in EMBEDDERS:
        raise SystemExit(f"unknown embedder {name!r}; available: {', '.join(EMBEDDERS)}")
    embedder = EMBEDDERS[name](**config.get("embedder_options", {}))
    template = config.get("context_template")

    for input_name, chunks_file in config["inputs"].items():
        print(f"[{input_name}]")
        embed(embedder, Path(chunks_file), template, Path(config["output_dir"]) / input_name)


if __name__ == "__main__":
    main()
