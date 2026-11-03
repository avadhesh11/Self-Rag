"""
Local RAG generator: any model served by Ollama.

Exposes the same interface as GeminiGenerator -- generate(query, chunks) -> str
-- so it drops into normal_rag/pipeline.py unchanged. Requires `ollama serve`
to be running locally and the model to be pulled (`ollama pull llama3:8b`).

Defaults can be overridden with the OLLAMA_MODEL / OLLAMA_HOST env vars or
demo.py's --ollama-model / --ollama-host flags.
"""
import os

import requests

DEFAULT_OLLAMA_MODEL = "llama3:8b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# The benchmark (benchmark_addon/benchmark/ollama_generator.py) follows the
# paper's baseline decoding: greedy, 100 new tokens. The interactive demo is
# less strict -- answers are often a sentence or two, so allow more room.
_NUM_PREDICT = 256
_TIMEOUT = 300.0


class OllamaGenerator:
    """Thin wrapper around Ollama's HTTP API for RAG generation."""

    def __init__(self, model: str | None = None, host: str | None = None):
        self._model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self._host = (host or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)).rstrip("/")
        try:
            r = requests.get(f"{self._host}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._host}. Is `ollama serve` running?"
            ) from exc

    @property
    def name(self) -> str:
        return f"{self._model} @ {self._host}"

    def generate(self, query: str, chunks: list[dict]) -> str:
        """Build a RAG prompt from retrieved chunks and answer with the local model.

        Args:
            query:  The user's question.
            chunks: List of passage dicts returned by PassageRetriever.search(),
                    each with at minimum a 'text' key.

        Returns:
            The model's answer as a plain string.
        """
        if not chunks:
            context_block = "(No relevant passages were retrieved.)"
        else:
            parts = []
            for i, chunk in enumerate(chunks, start=1):
                page_info = f" (page {chunk['page']})" if "page" in chunk else ""
                parts.append(f"[{i}]{page_info} {chunk['text']}")
            context_block = "\n\n".join(parts)

        prompt = (
            "You are a helpful assistant. "
            "Answer the question using the context passages provided below. "
            "If the answer is not contained in the context, say so clearly.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        r = requests.post(f"{self._host}/api/chat", timeout=_TIMEOUT, json={
            "model": self._model, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.0, "num_predict": _NUM_PREDICT},
        })
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
