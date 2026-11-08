"""Project configuration, read from a JSON file.

The file is config.json in the working directory by default; set the
RAG_CONFIG environment variable to use another one (e.g. inside Docker).
Each pipeline step reads its own section of the file.
"""
import json
import os
from pathlib import Path

DEFAULT_PATH = "config.json"


def load_config(section: str) -> dict:
    path = Path(os.environ.get("RAG_CONFIG", DEFAULT_PATH))
    with path.open(encoding="utf-8") as f:
        config = json.load(f)
    if section not in config:
        raise KeyError(f"section {section!r} not found in {path}")
    return config[section]
