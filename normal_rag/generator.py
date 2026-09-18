"""
Normal RAG generator: Gemini via google-genai SDK.

Reads GEMINI_API_KEY from the environment (or .env via python-dotenv).
Exposes one method: generate(query, chunks) -> str.

The prompt format is deliberately straightforward -- the generator's job
is purely to synthesise an answer from retrieved context; all retrieval
logic lives in normal_rag/pipeline.py.
"""
import os
import time

from dotenv import load_dotenv

load_dotenv()  # loads .env from the project root (or any parent)

# Default Gemini model to use. Can be overridden via GEMINI_MODEL env var.
# Valid values: "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", etc.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"

# Retry settings for transient API errors (503 overloaded, 429 rate-limited).
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0   # seconds; doubles on each attempt


class GeminiGenerator:
    """Thin wrapper around the google-genai client for RAG generation."""

    def __init__(self, model: str | None = None):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and fill in your key."
            )

        # Import here so the module can be imported even when google-genai is
        # not yet installed (e.g. during a quick syntax check).
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. "
                "Run: pip install google-genai"
            ) from exc

        self._client = genai.Client(api_key=api_key)
        self._model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        # Pre-create a chat session. The SDK recommends using Chat.send_message
        # instead of Models.generate_content to avoid the AFC warning.
        self._chat = self._client.chats.create(model=self._model)

    def generate(self, query: str, chunks: list[dict]) -> str:
        """Build a RAG prompt from retrieved chunks and call Gemini.

        Args:
            query:  The user's question.
            chunks: List of passage dicts returned by PassageRetriever.search().
                    Each dict has at minimum a 'text' key.

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
            "Answer the question using ONLY the context passages provided below. "
            "If the answer is not contained in the context, say so clearly.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        # Use the Chat API as recommended by the SDK (avoids AFC warning).
        # Retry on transient server errors (503 overloaded, 429 rate-limited).
        from google.genai.errors import ServerError
        last_exc = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._chat.send_message(prompt)
                return response.text.strip()
            except ServerError as exc:
                status = exc.status_code if hasattr(exc, "status_code") else 0
                if status in (503, 429) and attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    print(f"  [Gemini] {status} transient error, "
                          f"retrying in {delay:.0f}s (attempt {attempt}/{_MAX_RETRIES}) ...")
                    time.sleep(delay)
                    last_exc = exc
                else:
                    raise
        raise last_exc  # should not be reached
