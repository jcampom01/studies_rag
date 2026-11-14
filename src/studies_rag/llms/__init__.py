from studies_rag.llms.base import LLM, Completion
from studies_rag.llms.ollama import OllamaLLM

# Name used in the config -> class
LLMS: dict[str, type[LLM]] = {
    "ollama": OllamaLLM,
}

__all__ = ["LLMS", "LLM", "Completion", "OllamaLLM"]
