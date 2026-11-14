"""Step 2 of the pipeline: chunking.

Reads the page records written by the ingestion and splits them into chunks
with one or more chunking strategies. Each strategy writes its own file
(chunks_<name>.jsonl), so their results can be compared side by side.
Settings come from the "chunk" section of the config file.

Usage:
    PYTHONPATH=src .venv/bin/python -m studies_rag.chunk
"""
import json
import statistics
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from studies_rag.adapters import PageRecord
from studies_rag.chunkers import CHUNKERS, Chunker
from studies_rag.config import load_config


def read_pages(path: Path) -> Iterator[PageRecord]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield PageRecord(**json.loads(line))


def chunk(chunker: Chunker, input_path: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sizes = []
    with output.open("w", encoding="utf-8") as f:
        for c in chunker.split(read_pages(input_path)):
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            sizes.append(len(c.text))
    quartiles = statistics.quantiles(sizes, n=4)
    print(f"{len(sizes)} chunks -> {output}")
    print(f"  chars: min {min(sizes)}, p25 {quartiles[0]:.0f}, median {quartiles[1]:.0f}, "
          f"p75 {quartiles[2]:.0f}, max {max(sizes)}")


def main() -> None:
    config = load_config("chunk")
    options = config.get("chunker_options", {})
    for name in config["chunkers"]:
        if name not in CHUNKERS:
            raise SystemExit(f"unknown chunker {name!r}; available: {', '.join(CHUNKERS)}")
        print(f"[{name}]")
        chunker = CHUNKERS[name](**options.get(name, {}))
        chunk(chunker, Path(config["input"]), Path(config["output_dir"]) / f"chunks_{name}.jsonl")


if __name__ == "__main__":
    main()
