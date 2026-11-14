"""Step 1 of the pipeline: ingestion.

Turns the raw documents into page records (JSON Lines) using an adapter that
knows the layout of those documents. This module knows nothing about layouts.
Settings come from the "ingest" section of the config file.

Usage:
    PYTHONPATH=src .venv/bin/python -m studies_rag.ingest
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from studies_rag.adapters import ADAPTERS, Adapter, normalize
from studies_rag.config import load_config


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_files(raw_dir: Path, pattern: str) -> list[Path]:
    """Files under raw_dir (recursively) matching pattern, skipping byte-identical duplicates."""
    seen: dict[str, Path] = {}
    for path in sorted(raw_dir.rglob(pattern)):
        digest = file_hash(path)
        if digest in seen:
            print(f"WARNING: skipping {path} (duplicate of {seen[digest]})")
            continue
        seen[digest] = path
    return list(seen.values())


def ingest(adapter: Adapter, raw_dir: Path, output: Path) -> None:
    files = find_files(raw_dir, adapter.file_pattern)
    output.parent.mkdir(parents=True, exist_ok=True)
    pages = 0
    with output.open("w", encoding="utf-8") as f:
        for path in files:
            for record in adapter.extract(path):
                record.text = normalize(record.text)
                # ensure_ascii=False writes accents as-is instead of as \uXXXX escape sequences
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                pages += 1
    print(f"{len(files)} files -> {pages} pages written to {output}")


def main() -> None:
    config = load_config("ingest")
    name = config["adapter"]
    if name not in ADAPTERS:
        raise SystemExit(f"unknown adapter {name!r}; available: {', '.join(ADAPTERS)}")

    # Each adapter declares its own options as constructor parameters
    adapter = ADAPTERS[name](**config.get("adapter_options", {}))
    ingest(adapter, Path(config["raw_dir"]), Path(config["output"]))


if __name__ == "__main__":
    main()
