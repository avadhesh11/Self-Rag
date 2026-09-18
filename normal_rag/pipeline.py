"""
Normal RAG pipeline.

Flow:
    query
    -> PassageRetriever.search(query, k)   [existing, shared retriever]
    -> top-k chunks
    -> GeminiGenerator.generate(query, chunks)
    -> NormalRAGResult

Nothing in this module touches Self-RAG internals.
"""
import time
from dataclasses import dataclass


@dataclass
class NormalRAGResult:
    """All information produced by one Normal RAG run."""
    query: str
    retrieved_chunks: list[dict]   # raw passage dicts from PassageRetriever
    num_chunks: int
    answer: str
    latency: float                  # wall-clock seconds for retrieve + generate


def normal_rag_answer(
    query: str,
    generator,           # GeminiGenerator instance
    retriever,           # PassageRetriever instance (existing, shared)
    k: int = 5,
) -> NormalRAGResult:
    """Run the Normal RAG pipeline and return a NormalRAGResult.

    Args:
        query:     The user question.
        generator: A GeminiGenerator instance.
        retriever: The shared PassageRetriever (same FAISS index as Self-RAG).
        k:         Number of passages to retrieve.
    """
    start = time.perf_counter()

    chunks = retriever.search(query, k=k)
    answer = generator.generate(query, chunks)

    latency = time.perf_counter() - start

    return NormalRAGResult(
        query=query,
        retrieved_chunks=chunks,
        num_chunks=len(chunks),
        answer=answer,
        latency=latency,
    )
