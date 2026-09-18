"""
Lightweight evaluation metrics for comparing Normal RAG vs Self-RAG.

These are surface-level, automatically computable metrics that require no
human annotation and no external judge model. They are a starting point --
deeper answer-quality metrics (faithfulness, correctness, relevance) can
be added on top of these once both pipelines are stable.

Metrics produced per run:
    latency          -- wall-clock seconds (float)
    num_chunks       -- number of passages retrieved (int)
    answer_length    -- answer length in words (int)
    retrieved        -- whether retrieval actually happened (bool)
                        always True for Normal RAG; may be False for Self-RAG
    retrieval_decision -- Self-RAG token, e.g. "[Retrieval]" / "[No Retrieval]"
                          or "always" for Normal RAG
"""
from __future__ import annotations

from normal_rag.pipeline import NormalRAGResult
from selfrag.pipeline import SelfRAGResult


def metrics_from_normal(r: NormalRAGResult) -> dict:
    """Extract comparable metrics from a NormalRAGResult."""
    return {
        "pipeline": "normal_rag",
        "latency": round(r.latency, 3),
        "num_chunks": r.num_chunks,
        "answer_length": len(r.answer.split()),
        "retrieved": True,
        "retrieval_decision": "always",
    }


def metrics_from_self_rag(r: SelfRAGResult, latency: float | None) -> dict:
    """Extract comparable metrics from a SelfRAGResult."""
    return {
        "pipeline": "self_rag",
        "latency": round(latency, 3) if latency is not None else None,
        "num_chunks": len(r.candidates),
        "answer_length": len(r.final_answer.split()),
        "retrieved": r.retrieved,
        "retrieval_decision": r.decision or "(direct answer)",
    }


def print_metrics_table(normal_metrics: dict | None, self_rag_metrics: dict | None) -> None:
    """Print a side-by-side metrics table to stdout."""
    keys = ["latency", "num_chunks", "answer_length", "retrieved", "retrieval_decision"]
    col_w = 22

    header = f"{'Metric':{col_w}} {'Normal RAG':>{col_w}} {'Self-RAG':>{col_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for k in keys:
        n_val = normal_metrics.get(k, "n/a") if normal_metrics else "n/a"
        s_val = self_rag_metrics.get(k, "n/a") if self_rag_metrics else "n/a"
        print(f"{k:{col_w}} {str(n_val):>{col_w}} {str(s_val):>{col_w}}")

    print("=" * len(header))
