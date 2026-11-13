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
import math
import os
import re
import statistics
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Union


def normalize_text(s: str) -> str:
    """
    Standard text normalization for question answering evaluation (SQuAD / Self-RAG standard).
    Lowercases text, removes articles ('a', 'an', 'the'), strips punctuation, and collapses whitespace.
    """
    if not s:
        return ""
    # Lowercase
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Normalize whitespace
    return " ".join(s.split())


def compute_exact_match(prediction: str, references: Union[str, List[str]]) -> float:
    """
    Compute Exact Match (EM) score between prediction and one or more reference answers.
    Returns 1.0 if normalized prediction matches any normalized reference, else 0.0.
    """
    if isinstance(references, str):
        references = [references]
    norm_pred = normalize_text(prediction)
    for ref in references:
        if norm_pred == normalize_text(ref):
            return 1.0
    return 0.0


def compute_token_f1(prediction: str, references: Union[str, List[str]]) -> Dict[str, float]:
    """
    Compute token-level precision, recall, and F1 score against one or more reference answers.
    Returns the maximum F1 score among references along with corresponding precision and recall.
    """
    if isinstance(references, str):
        references = [references]
    norm_pred = normalize_text(prediction)
    pred_tokens = norm_pred.split()

    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for ref in references:
        ref_tokens = normalize_text(ref).split()
        if not pred_tokens and not ref_tokens:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not pred_tokens or not ref_tokens:
            continue

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_metrics["f1"]:
            best_metrics = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

    return best_metrics


def _parse_source_set(sources: Union[str, List[Any]]) -> Set[str]:
    """Parse and normalize source IDs / filenames (basenames in lowercase)."""
    if isinstance(sources, str):
        raw = [s.strip() for s in sources.split(";") if s.strip()]
    elif isinstance(sources, (list, tuple, set)):
        raw = [str(s).strip() for s in sources if str(s).strip()]
    else:
        raw = []
    return {os.path.basename(s).lower() for s in raw}


def compute_citation_metrics(
    predicted_sources: Union[str, List[Any]],
    reference_sources: Union[str, List[Any]],
) -> Dict[str, float]:
    """
    Compute precision, recall, F1, and exact match for cited sources.
    """
    pred_set = _parse_source_set(predicted_sources)
    ref_set = _parse_source_set(reference_sources)

    if not pred_set and not ref_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}
    if not ref_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "exact_match": 0.0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}

    intersection = pred_set & ref_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(ref_set)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    em = 1.0 if pred_set == ref_set else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": round(em, 4),
    }


def compute_latency_summary(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate mean, median, P95, min, max, std, and count from a list of latencies in ms.
    """
    if not latencies:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
        }

    sorted_lats = sorted(latencies)
    n = len(sorted_lats)
    mean_val = sum(sorted_lats) / n
    med_val = statistics.median(sorted_lats)

    # 95th percentile
    p95_idx = int(math.ceil(0.95 * n)) - 1
    p95_idx = max(0, min(p95_idx, n - 1))
    p95_val = sorted_lats[p95_idx]

    min_val = sorted_lats[0]
    max_val = sorted_lats[-1]
    std_val = statistics.stdev(sorted_lats) if n > 1 else 0.0

    return {
        "count": n,
        "mean_ms": round(mean_val, 2),
        "median_ms": round(med_val, 2),
        "p95_ms": round(p95_val, 2),
        "min_ms": round(min_val, 2),
        "max_ms": round(max_val, 2),
        "std_ms": round(std_val, 2),
    }


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
