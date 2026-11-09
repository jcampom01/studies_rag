"""LLM served by Ollama, through its native HTTP API (POST /api/chat).

The native API is used instead of the OpenAI-compatible one because it takes
Ollama's own settings: num_ctx (the context window) and think (the reasoning
mode of the models that have one).

Ollama silently cuts prompts longer than num_ctx and answers anyway. Requests
are sent with truncate=false, so a prompt that doesn't fit is an error instead.
"""
import time

import httpx

from studies_rag.llms.base import LLM, Completion


class OllamaLLM(LLM):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", think: bool = False,
                 num_ctx: int = 8192, temperature: float | None = None, timeout: float = 120.0):
        self.model = model
        self.think = think
        self.num_ctx = num_ctx
        self.temperature = temperature  # None = the model's default
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def generate(self, system: str, prompt: str) -> Completion:
        options = {"num_ctx": self.num_ctx}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,      # the whole answer in one response
            "think": self.think,
            "truncate": False,    # error instead of silently cutting the prompt
            "options": options,
        }

        start = time.perf_counter()
        response = self.client.post("/api/chat", json=body)
        seconds = time.perf_counter() - start
        if response.is_error:
            raise RuntimeError(f"Ollama answered {response.status_code}: {response.text}")

        data = response.json()
        return Completion(
            text=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            seconds=seconds,
        )

    def describe(self) -> dict:
        return {
            "llm": "ollama",
            "model": self.model,
            "think": self.think,
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
        }
