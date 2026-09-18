"""
Unified comparison runner.

Dispatches to the Normal RAG and/or Self-RAG pipelines based on mode,
then pretty-prints structured results.

Modes:
    "normal" -- Normal RAG only (Gemini)
    "self"   -- Self-RAG only (existing pipeline, unchanged)
    "both"   -- run both, display side-by-side

The Self-RAG pipeline is called exactly as demo.py calls it today:
    self_rag_answer(query, generator, retriever, k=k)
No Self-RAG internals are modified here.
"""
import time
from dataclasses import dataclass

from normal_rag.pipeline import NormalRAGResult, normal_rag_answer
from selfrag.pipeline import SelfRAGResult, self_rag_answer


DIVIDER = "=" * 55


@dataclass
class ComparisonResult:
    """Container for results from one or both pipelines."""
    query: str
    normal: NormalRAGResult | None = None
    self_rag: SelfRAGResult | None = None
    # Wall-clock latencies (seconds); set only when the corresponding
    # pipeline was run.  normal_rag/pipeline.py already measures its own
    # latency internally; we record Self-RAG latency here.
    self_rag_latency: float | None = None


def run_rag(
    query: str,
    mode: str,
    *,
    normal_generator=None,   # GeminiGenerator  (required for "normal" / "both")
    self_generator=None,     # SelfRAGGenerator (required for "self"   / "both")
    retriever,               # shared PassageRetriever
    k: int = 5,
) -> ComparisonResult:
    """Run one or both RAG pipelines and return a ComparisonResult.

    Args:
        query:            User question.
        mode:             "normal" | "self" | "both"
        normal_generator: GeminiGenerator (pass None if not using Normal RAG).
        self_generator:   SelfRAGGenerator (pass None if not using Self-RAG).
        retriever:        Shared PassageRetriever instance.
        k:                Number of passages to retrieve.
    """
    mode = mode.lower().strip()
    result = ComparisonResult(query=query)

    if mode in ("normal", "both"):
        if normal_generator is None:
            raise ValueError("normal_generator is required for Normal RAG mode.")
        result.normal = normal_rag_answer(query, normal_generator, retriever, k=k)

    if mode in ("self", "both"):
        if self_generator is None:
            raise ValueError("self_generator is required for Self-RAG mode.")
        t0 = time.perf_counter()
        result.self_rag = self_rag_answer(query, self_generator, retriever, k=k)
        result.self_rag_latency = time.perf_counter() - t0

    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_normal_rag(r: NormalRAGResult) -> None:
    print(f"\n{DIVIDER}")
    print("  NORMAL RAG  (Gemini)")
    print(DIVIDER)
    print(f"Retrieved chunks : {r.num_chunks}")
    print(f"Latency          : {r.latency:.2f}s")

    if r.retrieved_chunks:
        print("\nRetrieved passages:")
        for i, chunk in enumerate(r.retrieved_chunks, start=1):
            page   = chunk.get("page", "?")
            score  = chunk.get("retrieval_score", 0.0)
            text   = chunk.get("text", "")
            preview = text[:200] + ("..." if len(text) > 200 else "")
            print(f"  [{i}] page {page}  score={score:.3f}")
            print(f"      {preview}")

    print(f"\nAnswer:\n{r.answer}")


def print_self_rag(r: SelfRAGResult, latency: float | None) -> None:
    print(f"\n{DIVIDER}")
    print("  SELF-RAG  (selfrag/selfrag_llama2_7b)")
    print(DIVIDER)

    # Retrieval decision
    if r.retrieved:
        decision_str = f"Yes ({r.decision})"
    elif r.decision is None:
        decision_str = "(none emitted -- model answered directly)"
    else:
        decision_str = f"No ({r.decision})"
    print(f"Retrieval decision: {decision_str}")
    print(f"Retrieved chunks  : {len(r.candidates)}")
    if latency is not None:
        print(f"Latency           : {latency:.2f}s")

    # Per-candidate critique tokens (same info as the original print_trace)
    if r.candidates:
        print("\nPassage critique:")
        for c in r.candidates:
            tag = "-> SELECTED" if c is r.selected else ""
            print(
                f"  page {c.passage.get('page', '?'):>3}  "
                f"ISREL={c.isrel}  ISSUP={c.issup}  "
                f"ISUSE={c.isuse}  score={c.score:.2f}  {tag}"
            )

    print(f"\nAnswer:\n{r.final_answer}")


def print_comparison(result: ComparisonResult) -> None:
    """Print both results followed by a brief side-by-side summary."""
    if result.normal:
        print_normal_rag(result.normal)
    if result.self_rag:
        print_self_rag(result.self_rag, result.self_rag_latency)

    if result.normal and result.self_rag:
        print(f"\n{DIVIDER}")
        print("  COMPARISON SUMMARY")
        print(DIVIDER)

        # Normal RAG row
        print(f"{'':20} {'NORMAL RAG':>15}  {'SELF-RAG':>15}")
        print("-" * 55)

        n_chunks_normal = result.normal.num_chunks
        n_chunks_self   = len(result.self_rag.candidates)
        lat_normal      = f"{result.normal.latency:.2f}s"
        lat_self        = (
            f"{result.self_rag_latency:.2f}s"
            if result.self_rag_latency is not None else "n/a"
        )
        retrieved_self  = "Yes" if result.self_rag.retrieved else "No"
        ans_len_normal  = len(result.normal.answer.split())
        ans_len_self    = len(result.self_rag.final_answer.split())

        print(f"{'Chunks retrieved':20} {n_chunks_normal:>15}  {n_chunks_self:>15}")
        print(f"{'Retrieval decision':20} {'Always':>15}  {retrieved_self:>15}")
        print(f"{'Latency':20} {lat_normal:>15}  {lat_self:>15}")
        print(f"{'Answer length (words)':20} {ans_len_normal:>15}  {ans_len_self:>15}")
