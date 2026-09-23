"""
Query normalizer for the Self-RAG pipeline.

Decouples user queries into:
1. `search_query`: Cleaned semantic search terms without conversational fluff.
2. `generation_instruction`: Clear, imperative task instructions that prevent
   the 7B LLaMA-2 base model from triggering its conversational ambiguity/refusal
   guardrails ("I'm sorry, but I'm not sure what you are asking...").
"""
import re


def clean_text_artifacts(text: str) -> str:
    """Fix common PDF extraction and tokenizer spacing artifacts.
    E.g. 'SELF -RAG' -> 'SELF-RAG', 'Gen- eration' -> 'Generation'.
    """
    if not text:
        return text

    # Fix broken hyphenation with spaces (e.g. 'SELF -RAG' or 'SELF - RAG')
    cleaned = re.sub(r"(\b[A-Za-z0-9]+)\s+-\s*([A-Za-z0-9]+)", r"\1-\2", text)
    # Fix hyphenated words split across lines like 'Gen- eration' -> 'Generation'
    cleaned = re.sub(r"(\b[A-Za-z]+)-\s+([a-z]+)", r"\1\2", cleaned)
    # Collapse multiple whitespace characters
    cleaned = " ".join(cleaned.split())
    return cleaned


# Patterns to convert conversational requests into explicit generation directives
_CONVERSATIONAL_PATTERNS = [
    # "Tell me anything/something/everything about X"
    (
        r"^(?:please\s+)?tell\s+me\s+(?:anything|something|everything)\s+about\s+(.+?)[\?\.]*$",
        r"Explain what \1 is and provide an overview of its key concepts.",
        r"\1 overview",
    ),
    # "Tell me about X" / "Talk about X" / "Can you talk about X"
    (
        r"^(?:please\s+)?(?:tell\s+me\s+about|can\s+you\s+talk\s+about|talk\s+about)\s+(.+?)[\?\.]*$",
        r"Provide an overview and key details about \1.",
        r"\1",
    ),
    # "Give me info/information on X"
    (
        r"^(?:please\s+)?(?:give\s+me\s+(?:info|information)\s+(?:on|about))\s+(.+?)[\?\.]*$",
        r"Provide key factual information about \1 based on the retrieved text.",
        r"\1",
    ),
    # "What is known about X" / "What can you tell me about X"
    (
        r"^(?:what\s+(?:is\s+known\s+about|can\s+you\s+tell\s+me\s+about))\s+(.+?)[\?\.]*$",
        r"Explain what \1 is and describe its main features.",
        r"\1",
    ),
    # "What is/are [term]?" -> Explain what [term] is/are (e.g. "What is SELF-RAG?", "What are reflection tokens?")
    # Does NOT rewrite specific questions like "What is the capital of Australia?" or "What is 2 + 2?"
    (
        r"^what\s+are\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)[\?\.]*$",
        r"Explain what \1 are and summarize their key aspects.",
        r"\1",
    ),
    (
        r"^what\s+is\s+(?:a|an|the\s+)?([A-Za-z0-9_-]+)[\?\.]*$",
        r"Explain what \1 is and summarize its key aspects.",
        r"\1",
    ),
]


def normalize_query(query: str) -> tuple[str, str]:
    """Normalize a user query for both retrieval and generation.

    Args:
        query: Raw user query string.

    Returns:
        (search_query, generation_instruction)
    """
    cleaned = clean_text_artifacts(query).strip()
    if not cleaned:
        return "", ""

    instruction = cleaned
    search_query = cleaned

    for pattern, inst_repl, search_repl in _CONVERSATIONAL_PATTERNS:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            instruction = re.sub(pattern, inst_repl, cleaned, flags=re.IGNORECASE).strip()
            search_query = re.sub(pattern, search_repl, cleaned, flags=re.IGNORECASE).strip()
            break

    # Strip trailing punctuation from search query for vector lookup
    search_query = search_query.rstrip("?.!").strip()

    return search_query, instruction
