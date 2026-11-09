"""Project configuration.

Settings come from a JSON file: config.json in the working directory by default;
set the RAG_CONFIG environment variable to use another one (e.g. inside Docker).
Each pipeline step reads its own section of the file.

Secrets (passwords, tokens) never go in that file: they come from environment
variables. For local runs they can be written in a .env file in the working
directory, which is loaded without overriding variables already set.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_PATH = "config.json"
ENV_FILE = ".env"


def load_config(section: str) -> dict:
    load_dotenv(ENV_FILE)
    path = Path(os.environ.get("RAG_CONFIG", DEFAULT_PATH))
    with path.open(encoding="utf-8") as f:
        config = json.load(f)
    if section not in config:
        raise KeyError(f"section {section!r} not found in {path}")
    return config[section]


def require_env(name: str) -> str:
    """Value of an environment variable, or a clear error if it isn't set."""
    value = os.environ.get(name)
    if not value:
        raise KeyError(f"environment variable {name} is not set (add it to {ENV_FILE} or the environment)")
    return value
