#!/usr/bin/env python3
"""EvoRAG Task 12F: Controlled Three-System Comparison Benchmark.

Compares:
1. Normal RAG (Retrieval + generation from retrieved context)
2. Production GGUF Self-RAG (models/selfrag_llama2_7b.Q4_K_M.gguf)
3. Fine-Tuned MLX Self-RAG integrated with EvoRAG (models/mlx/selfrag_llama2_7b_4bit + QLoRA adapter)

Runs on an internal 15-question fixture (13 held-out category questions + 2 memory queries)
with zero contamination against the frozen 32-question evaluation benchmark.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    compute_token_f1,
    compute_exact_match,
    compute_citation_metrics,
    compute_latency_summary,
)
from retriever.retriever import PassageRetriever
from evorag.mlx_generator import MLXSelfRAGGenerator
from evorag.controller import EvoRAGController
from evorag.memory.store import ExperienceStore
from evorag.memory.retriever import ExperienceRetriever
from evorag.memory.validator import ExperienceValidator
from selfrag.reflection_tokens import strip_reflection_tokens

FIXTURE_FILE = PROJECT_ROOT / "training/qlora_dbms/mlx_dataset/internal_test.jsonl"
OUTPUT_METRICS_FILE = PROJECT_ROOT / "training/qlora_dbms/task12f_comparison_results.json"

# Selected 13 canonical category examples from internal_test.jsonl + 2 memory test queries
SELECTED_CATEGORY_IDS = [
    "DBMS-QLORA-0061",  # algorithms
    "DBMS-QLORA-0187",  # buffer_management
    "DBMS-QLORA-0044",  # comparisons
    "DBMS-QLORA-0022",  # conceptual_explanations
    "DBMS-QLORA-0146",  # concurrency_control
    "DBMS-QLORA-0247",  # context_grounded_abstention (unanswerable/private)
    "DBMS-QLORA-0005",  # dbms_definitions
    "DBMS-QLORA-0104",  # indexing
    "DBMS-QLORA-0086",  # query_processing
    "DBMS-QLORA-0165",  # recovery_and_logging
    "DBMS-QLORA-0224",  # scenario_based_reasoning
    "DBMS-QLORA-0203",  # storage_architecture
    "DBMS-QLORA-0132",  # transactions
]


def load_fixtures() -> List[Dict[str, Any]]:
    with open(FIXTURE_FILE, "r", encoding="utf-8") as f:
        all_test = [json.loads(line) for line in f]
    selected = [x for x in all_test if x["example_id"] in SELECTED_CATEGORY_IDS]

    # Add 2 memory queries
    memory_fixtures = [
        {
            "example_id": "MEM-EXACT-001",
            "category": "experience_exact_match",
            "prompt": "How does physical data independence protect applications from storage modifications?",
            "completion": "Physical data independence insulates the logical schema and application queries from changes in physical storage structures such as page layouts, file paths, clustering, or secondary indexing.",
            "source_document": "01_dbms_fundamentals_and_architecture.pdf",
            "supporting_chunk_ids": [2],
            "expected_path": "Path A",
        },
        {
            "example_id": "MEM-SEMANTIC-002",
            "category": "experience_semantic_match",
            "prompt": "How does physical data independence protect application queries when storage changes?",
            "completion": "Physical data independence insulates the logical schema and application queries from changes in physical storage structures such as page layouts, file paths, clustering, or secondary indexing.",
            "source_document": "01_dbms_fundamentals_and_architecture.pdf",
            "supporting_chunk_ids": [2],
            "expected_path": "Path B",
        },
    ]
    return selected + memory_fixtures


def run_gguf_selfrag_subprocess(questions: List[str]) -> List[Dict[str, Any]]:
    """Execute GGUF Self-RAG inside production .venv to isolate llama-cpp-python."""
    py_code = """
import json, sys, time
from selfrag.generator import SelfRAGGenerator
from retriever.retriever import PassageRetriever
from selfrag.pipeline import self_rag_answer

questions = json.loads(sys.argv[1])
t0 = time.perf_counter()
gen = SelfRAGGenerator('models/selfrag_llama2_7b.Q4_K_M.gguf')
load_time = time.perf_counter() - t0
retriever = PassageRetriever(index_dir='index/dbms_sql_v1')

results = []
for q in questions:
    t_start = time.perf_counter()
    res = self_rag_answer(q, gen, retriever, k=3, max_segments=1)
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    cited_docs = []
    if res.selected and res.selected.passage:
        src = res.selected.passage.get('source')
        if src:
            cited_docs.append(str(src))

    results.append({
        'query': q,
        'answer': res.final_answer,
        'retrieved': res.retrieved,
        'decision': res.decision,
        'isrel': res.selected.isrel if res.selected else None,
        'issup': res.selected.issup if res.selected else None,
        'isuse': res.selected.isuse if res.selected else None,
        'cited_docs': cited_docs,
        'latency_ms': latency_ms,
        'load_time_s': load_time,
    })

print(json.dumps(results))
"""
    cmd = [
        "arch", "-arm64",
        ".venv/bin/python", "-c", py_code,
        json.dumps(questions),
    ]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"GGUF runner failed:\n{proc.stderr}\n{proc.stdout}")

    # Parse last line which is JSON
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def main():
    print("=" * 80)
    print("EVORAG TASK 12F: CONTROLLED THREE-SYSTEM COMPARISON BENCHMARK")
    print("=" * 80)

    fixtures = load_fixtures()
    print(f"Loaded {len(fixtures)} controlled evaluation queries across 13 categories + 2 memory queries.")
    questions = [f["prompt"] for f in fixtures]

    # Initialize shared components
    print("\n--- Initializing Document Retriever and Experience Memory ---")
    retriever = PassageRetriever(index_dir="index/dbms_sql_v1")
    store = ExperienceStore("data/experience_memory.db")
    mem_retriever = ExperienceRetriever("data", store=store)
    validator = ExperienceValidator(similarity_threshold=0.85)

    # 1. System 3: Fine-Tuned MLX EvoRAG
    print("\n--- 1. Evaluating System 3: Fine-Tuned MLX EvoRAG ---")
    t0_mlx_load = time.perf_counter()
    mlx_gen = MLXSelfRAGGenerator(
        model_path="models/mlx/selfrag_llama2_7b_4bit",
        adapter_path="training/qlora_dbms/adapters/full_dbms_v1",
        max_tokens=220,
    )
    mlx_load_time = time.perf_counter() - t0_mlx_load
    print(f"Fine-Tuned MLX loaded in {mlx_load_time:.2f}s | Memory: {mlx_gen.memory_after_load_gb:.2f} GB")

    controller = EvoRAGController(
        store=store,
        memory_retriever=mem_retriever,
        validator=validator,
        generator=mlx_gen,
        retriever=retriever,
        doc_index_dir="index/dbms_sql_v1",
    )

    mlx_evorag_results = []
    for idx, f in enumerate(fixtures, 1):
        q = f["prompt"]
        t_start = time.perf_counter()
        res = controller.answer(q)
        lat_ms = (time.perf_counter() - t_start) * 1000.0

        cited_docs = []
        for p in res.evidence:
            if isinstance(p, dict):
                src = p.get("source") or p.get("source_id")
                if src and str(src) not in cited_docs:
                    cited_docs.append(str(src))

        self_res = res.self_rag_result
        isrel = self_res.selected.isrel if (self_res and self_res.selected) else None
        issup = self_res.selected.issup if (self_res and self_res.selected) else None
        isuse = self_res.selected.isuse if (self_res and self_res.selected) else None
        decision = self_res.decision if self_res else None

        item = {
            "example_id": f["example_id"],
            "category": f["category"],
            "prompt": q,
            "ground_truth": f["completion"],
            "expected_source": f.get("source_document"),
            "answer": res.answer,
            "response_mode": res.response_mode,
            "memory_hit": res.memory_hit,
            "memory_context_used": res.memory_context_used,
            "similarity_score": res.similarity_score,
            "decision": decision,
            "isrel": isrel,
            "issup": issup,
            "isuse": isuse,
            "cited_docs": cited_docs,
            "latency_ms": lat_ms,
        }
        mlx_evorag_results.append(item)
        print(f"[{idx:2d}/15] MLX EvoRAG -> {f['example_id']} | Mode: {res.response_mode:20s} | "
              f"Lat: {lat_ms:6.1f}ms | Decision: {decision} | Utility: {isuse}")

    # 2. System 1: Normal RAG (PassageRetriever + standard extractive generation)
    print("\n--- 2. Evaluating System 1: Normal RAG ---")
    normal_rag_results = []
    for idx, f in enumerate(fixtures, 1):
        q = f["prompt"]
        t_start = time.perf_counter()
        chunks = retriever.search(q, k=3)
        # Synthesize top-1 chunk text directly as baseline extractive answer
        best_chunk = chunks[0] if chunks else {}
        ans = best_chunk.get("text", "")
        lat_ms = (time.perf_counter() - t_start) * 1000.0

        cited_docs = [best_chunk.get("source")] if best_chunk.get("source") else []
        normal_rag_results.append({
            "example_id": f["example_id"],
            "category": f["category"],
            "prompt": q,
            "ground_truth": f["completion"],
            "expected_source": f.get("source_document"),
            "answer": ans,
            "cited_docs": cited_docs,
            "latency_ms": lat_ms,
        })
        print(f"[{idx:2d}/15] Normal RAG -> {f['example_id']} | Lat: {lat_ms:6.1f}ms | Cited: {cited_docs}")

    # 3. System 2: Production GGUF Self-RAG
    print("\n--- 3. Evaluating System 2: Production GGUF Self-RAG ---")
    gguf_raw_results = run_gguf_selfrag_subprocess(questions)
    gguf_selfrag_results = []
    for f, gr in zip(fixtures, gguf_raw_results):
        gguf_selfrag_results.append({
            "example_id": f["example_id"],
            "category": f["category"],
            "prompt": f["prompt"],
            "ground_truth": f["completion"],
            "expected_source": f.get("source_document"),
            "answer": gr["answer"],
            "decision": gr.get("decision"),
            "isrel": gr.get("isrel"),
            "issup": gr.get("issup"),
            "isuse": gr.get("isuse"),
            "cited_docs": gr.get("cited_docs", []),
            "latency_ms": gr.get("latency_ms", 0.0),
        })
        print(f"[{f['example_id']}] GGUF Self-RAG | Lat: {gr['latency_ms']:6.1f}ms | "
              f"Decision: {gr.get('decision')} | Utility: {gr.get('isuse')}")

    # 4. Compute Aggregate Metrics Across Systems
    print("\n" + "=" * 80)
    print("COMPUTING COMPARATIVE BENCHMARK METRICS")
    print("=" * 80)

    def score_system(results: List[Dict[str, Any]], system_name: str) -> Dict[str, Any]:
        token_f1s, precisions, recalls = [], [], []
        cite_f1s, cite_precs, cite_recalls = [], [], []
        latencies = []
        malformed_count = 0
        abstention_correct = 0
        abstention_total = 0

        for r in results:
            pred_ans = strip_reflection_tokens(r["answer"])
            gt_ans = strip_reflection_tokens(r["ground_truth"])
            tf1 = compute_token_f1(pred_ans, gt_ans)
            token_f1s.append(tf1["f1"])
            precisions.append(tf1["precision"])
            recalls.append(tf1["recall"])

            pred_docs = r.get("cited_docs", [])
            ref_docs = [r["expected_source"]] if r.get("expected_source") else []
            cf1 = compute_citation_metrics(pred_docs, ref_docs)
            cite_f1s.append(cf1["f1"])
            cite_precs.append(cf1["precision"])
            cite_recalls.append(cf1["recall"])

            latencies.append(r["latency_ms"])

            # Check malformed
            if "[" in pred_ans and not any(tag in pred_ans for tag in ("[Retrieval]", "[Relevant]")):
                malformed_count += 1

            # Check abstention accuracy
            if r["category"] == "context_grounded_abstention":
                abstention_total += 1
                lower = pred_ans.lower()
                if any(w in lower for w in ("not", "does not", "no private", "unsupported", "cannot")):
                    abstention_correct += 1

        lat_summary = compute_latency_summary(latencies)
        return {
            "system_name": system_name,
            "total_queries": len(results),
            "token_f1": round(sum(token_f1s) / len(token_f1s), 4),
            "token_precision": round(sum(precisions) / len(precisions), 4),
            "token_recall": round(sum(recalls) / len(recalls), 4),
            "citation_f1": round(sum(cite_f1s) / len(cite_f1s), 4),
            "citation_precision": round(sum(cite_precs) / len(cite_precs), 4),
            "citation_recall": round(sum(cite_recalls) / len(cite_recalls), 4),
            "abstention_accuracy": round(abstention_correct / abstention_total, 4) if abstention_total else 1.0,
            "malformed_rate": round(malformed_count / len(results), 4),
            "latency_summary": lat_summary,
        }

    summary_normal = score_system(normal_rag_results, "Normal RAG")
    summary_gguf = score_system(gguf_selfrag_results, "Production GGUF Self-RAG")
    summary_mlx = score_system(mlx_evorag_results, "Fine-Tuned MLX EvoRAG")

    # Routing distribution for EvoRAG
    path_counts = {"Path A": 0, "Path B": 0, "Path C": 0}
    for r in mlx_evorag_results:
        mode = r.get("response_mode")
        if mode == "direct_reuse":
            path_counts["Path A"] += 1
        elif mode == "experience_assisted":
            path_counts["Path B"] += 1
        else:
            path_counts["Path C"] += 1

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "fixture_size": len(fixtures),
        "systems": {
            "normal_rag": summary_normal,
            "gguf_selfrag": summary_gguf,
            "mlx_evorag": summary_mlx,
        },
        "evorag_routing_distribution": path_counts,
        "hardware_telemetry": {
            "mlx_load_time_seconds": round(mlx_load_time, 2),
            "mlx_peak_memory_gb": round(mlx_gen.get_metrics()["peak_memory_gb"], 2),
            "gguf_load_time_seconds": round(gguf_raw_results[0].get("load_time_s", 0.0), 2),
        },
        "query_details": {
            "normal_rag": normal_rag_results,
            "gguf_selfrag": gguf_selfrag_results,
            "mlx_evorag": mlx_evorag_results,
        },
    }

    with open(OUTPUT_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nBenchmark metrics written to {OUTPUT_METRICS_FILE}")
    print("\n--- SUMMARY COMPARISON TABLE ---")
    print(f"{'Metric':<25} | {'Normal RAG':<12} | {'GGUF Self-RAG':<15} | {'MLX EvoRAG':<15}")
    print("-" * 75)
    print(f"{'Token F1':<25} | {summary_normal['token_f1']:<12.4f} | {summary_gguf['token_f1']:<15.4f} | {summary_mlx['token_f1']:<15.4f}")
    print(f"{'Citation F1':<25} | {summary_normal['citation_f1']:<12.4f} | {summary_gguf['citation_f1']:<15.4f} | {summary_mlx['citation_f1']:<15.4f}")
    print(f"{'Mean Latency (ms)':<25} | {summary_normal['latency_summary']['mean_ms']:<12.1f} | {summary_gguf['latency_summary']['mean_ms']:<15.1f} | {summary_mlx['latency_summary']['mean_ms']:<15.1f}")
    print(f"{'P95 Latency (ms)':<25} | {summary_normal['latency_summary']['p95_ms']:<12.1f} | {summary_gguf['latency_summary']['p95_ms']:<15.1f} | {summary_mlx['latency_summary']['p95_ms']:<15.1f}")
    print(f"{'Path A / B / C':<25} | {'—':<12} | {'—':<15} | {path_counts['Path A']}/{path_counts['Path B']}/{path_counts['Path C']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

