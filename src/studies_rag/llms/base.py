"""Contract for LLMs: given instructions and a prompt, generate text."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Completion:
    text: str               # the generated answer
    prompt_tokens: int      # tokens the model read (instructions + prompt)
    output_tokens: int      # tokens it generated
    seconds: float          # wall time of the call


class LLM(ABC):
    @abstractmethod
    def generate(self, system: str, prompt: str) -> Completion:
        """Generate an answer to `prompt` following the instructions in `system`."""

    @abstractmethod
    def describe(self) -> dict:
        """Settings that determine the answers (model, temperature...)."""
