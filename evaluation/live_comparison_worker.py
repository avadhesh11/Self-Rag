"""Worker process for individual system execution in the Five-System Comparison.

Executes exactly one of the five systems:
  1. normal         : Normal RAG (Gemini API via google-genai)
  2. selfrag_gguf   : Original Self-RAG baseline (selfrag_llama2_7b.Q4_K_M.gguf)
  3. evorag_gguf    : EvoRAG GGUF (GGUF + Experience Memory)
  4. selfrag_mlx    : Original Self-RAG MLX base model (models/mlx/selfrag_llama2_7b_4bit, no adapter)
  5. evorag_mlx     : Fine-Tuned EvoRAG MLX (4-bit base + QLoRA adapter + Experience Memory)

Emits a machine-readable JSON result containing detailed timing, answers, and routing metadata.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TimedRetriever:
    """Wrapper around PassageRetriever that accumulates exact retrieval search latency."""

    def __init__(self, base_retriever: Any):
        self.base = base_retriever
        self.total_retrieval_ms: float = 0.0
        self.call_count: int = 0

    @property
    def corpus(self) -> Any:
        return self.base.corpus

    @property
    def index(self) -> Any:
        return self.base.index

    def search(self, query: str, k: int = 5) -> List[dict]:
        t0 = time.perf_counter()
        results = self.base.search(query, k=k)
        self.total_retrieval_ms += (time.perf_counter() - t0) * 1000
        self.call_count += 1
        return results

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def _extract_citations(evidence: List[dict]) -> List[str]:
    citations: List[str] = []
    seen = set()
    for p in evidence:
        src = p.get("source") or p.get("source_id") or "Unknown"
        pg = p.get("page")
        cid = p.get("chunk_id")
        key = (src, pg, cid)
        if key not in seen:
            seen.add(key)
            loc_parts = []
            if pg is not None:
                loc_parts.append(f"page {pg}")
            if cid is not None:
                loc_parts.append(f"chunk {cid}")
            loc = f" ({', '.join(loc_parts)})" if loc_parts else ""
            citations.append(f"{src}{loc}")
    return citations


def run_worker(args: argparse.Namespace) -> Dict[str, Any]:
    system_id = args.system
    question = args.question
    index_dir = args.index
    memory_db = args.memory_db
    memory_index = args.memory_index
    threshold = args.threshold
    k = args.k
    max_segments = args.max_segments
    max_tokens = args.max_tokens

    record: Dict[str, Any] = {
        "system_id": system_id,
        "system_name": "",
        "backend": "",
        "model": "",
        "success": False,
        "error": None,
        "answer": "",
        "answer_status": "Unknown",
        "total_query_latency_ms": None,
        "retrieval_latency_ms": None,
        "generation_latency_ms": None,
        "model_load_time_ms": None,
        "is_cold_start": True,
        "retrieval_decision": None,
        "retrieved_passage_count": 0,
        "routing_path": None,
        "memory_hit": False,
        "memory_match_id": None,
        "memory_match_score": None,
        "citations": [],
        "evidence": [],
    }

    try:
        # -------------------------------------------------------------
        # System 1: Normal RAG (Gemini API)
        # -------------------------------------------------------------
        if system_id == "normal":
            record["system_name"] = "Normal RAG (Gemini)"
            record["backend"] = "Google Gemini API (google-genai)"

            from retriever.retriever import PassageRetriever
            from normal_rag.generator import GeminiGenerator

            t0_load = time.perf_counter()
            try:
                generator = GeminiGenerator()
            except Exception as e:
                record["answer_status"] = f"Configuration Error: {e}"
                record["error"] = str(e)
                return record
            record["model_load_time_ms"] = round((time.perf_counter() - t0_load) * 1000, 2)
            record["model"] = getattr(generator, "_model", "gemini-3.5-flash-lite")

            retriever = TimedRetriever(PassageRetriever(index_dir))

            t0_query = time.perf_counter()
            passages = retriever.search(question, k=k)
            retrieval_ms = retriever.total_retrieval_ms

            t0_gen = time.perf_counter()
            answer = generator.generate(question, passages)
            gen_ms = (time.perf_counter() - t0_gen) * 1000
            total_ms = (time.perf_counter() - t0_query) * 1000

            record["success"] = True
            record["answer"] = answer
            record["answer_status"] = "Success"
            record["total_query_latency_ms"] = round(total_ms, 2)
            record["retrieval_latency_ms"] = round(retrieval_ms, 2)
            record["generation_latency_ms"] = round(gen_ms, 2)
            record["retrieval_decision"] = "[Retrieval] (Forced by RAG architecture)"
            record["retrieved_passage_count"] = len(passages)
            record["citations"] = _extract_citations(passages)
            record["evidence"] = passages
            return record

        # -------------------------------------------------------------
        # System 2: Original Self-RAG GGUF
        # -------------------------------------------------------------
        elif system_id == "selfrag_gguf":
            record["system_name"] = "Original Self-RAG (GGUF)"
            record["backend"] = "llama.cpp (Q4_K_M)"
            record["model"] = args.model

            from retriever.retriever import PassageRetriever
            from selfrag.generator import SelfRAGGenerator
            from selfrag.pipeline import self_rag_answer

            if not Path(args.model).exists() and os.environ.get("SELFRAG_BACKEND") != "vllm":
                record["answer_status"] = "Model Error: GGUF model file not found"
                record["error"] = f"File not found: {args.model}"
                return record

            retriever = TimedRetriever(PassageRetriever(index_dir))

            t0_load = time.perf_counter()
            generator = SelfRAGGenerator(args.model)
            record["model_load_time_ms"] = round((time.perf_counter() - t0_load) * 1000, 2)

            t0_query = time.perf_counter()
            sr_res = self_rag_answer(question, generator, retriever, k=k, max_segments=max_segments)
            total_ms = (time.perf_counter() - t0_query) * 1000
            retrieval_ms = retriever.total_retrieval_ms
            gen_ms = max(0.0, total_ms - retrieval_ms)

            passages = [c.passage for seg in sr_res.segments for c in getattr(seg, "candidates", []) if hasattr(c, "passage")]

            record["success"] = True
            record["answer"] = sr_res.final_answer
            record["answer_status"] = "Success"
            record["total_query_latency_ms"] = round(total_ms, 2)
            record["retrieval_latency_ms"] = round(retrieval_ms, 2) if retriever.call_count > 0 else 0.0
            record["generation_latency_ms"] = round(gen_ms, 2)
            record["retrieval_decision"] = sr_res.decision or ("[Retrieval]" if sr_res.retrieved else "[No Retrieval]")
            record["retrieved_passage_count"] = len(passages)
            record["citations"] = _extract_citations(passages)
            record["evidence"] = passages
            return record

        # -------------------------------------------------------------
        # System 3: EvoRAG GGUF
        # -------------------------------------------------------------
        elif system_id == "evorag_gguf":
            record["system_name"] = "EvoRAG (GGUF + Experience Memory)"
            record["backend"] = "llama.cpp (Q4_K_M) + SQLite/FAISS Memory"
            record["model"] = args.model

            from retriever.retriever import PassageRetriever
            from selfrag.generator import SelfRAGGenerator
            from evorag.controller import EvoRAGController
            from evorag.memory.store import ExperienceStore
            from evorag.memory.retriever import ExperienceRetriever
            from evorag.memory.validator import ExperienceValidator
            from evorag.memory.index_builder import ExperienceIndexBuilder

            if not Path(args.model).exists() and os.environ.get("SELFRAG_BACKEND") != "vllm":
                record["answer_status"] = "Model Error: GGUF model file not found"
                record["error"] = f"File not found: {args.model}"
                return record

            store = ExperienceStore(memory_db)
            mem_retriever = ExperienceRetriever(memory_index, store=store)
            validator = ExperienceValidator(similarity_threshold=threshold)
            index_builder = ExperienceIndexBuilder(index_dir=memory_index)
            retriever = TimedRetriever(PassageRetriever(index_dir))

            t0_load = time.perf_counter()
            generator = SelfRAGGenerator(args.model)
            record["model_load_time_ms"] = round((time.perf_counter() - t0_load) * 1000, 2)

            controller = EvoRAGController(
                store=store,
                memory_retriever=mem_retriever,
                validator=validator,
                generator=generator,
                retriever=retriever,
                doc_index_dir=index_dir,
                index_builder=index_builder,
                k=k,
                max_segments=max_segments,
            )

            t0_query = time.perf_counter()
            try:
                ev_res = controller.answer(question)
            finally:
                store.close()
            total_ms = (time.perf_counter() - t0_query) * 1000

            if ev_res.response_mode == "direct_reuse":
                route_str = "Path A (Direct Reuse)"
                retrieval_ms = 0.0
                gen_ms = 0.0
            elif ev_res.response_mode == "experience_assisted":
                route_str = "Path B (Experience-Assisted Self-RAG)"
                retrieval_ms = retriever.total_retrieval_ms
                gen_ms = max(0.0, total_ms - retrieval_ms)
            else:
                route_str = "Path C (Normal Self-RAG Fallback)"
                retrieval_ms = retriever.total_retrieval_ms
                gen_ms = max(0.0, total_ms - retrieval_ms)

            dec = None
            if ev_res.self_rag_result:
                dec = ev_res.self_rag_result.decision or ("[Retrieval]" if ev_res.self_rag_result.retrieved else "[No Retrieval]")
            elif ev_res.response_mode == "direct_reuse":
                dec = "[Memory Hit - No LLM Probe]"

            record["success"] = True
            record["answer"] = ev_res.answer
            record["answer_status"] = "Success"
            record["total_query_latency_ms"] = round(total_ms, 2)
            record["retrieval_latency_ms"] = round(retrieval_ms, 2) if retrieval_ms is not None else None
            record["generation_latency_ms"] = round(gen_ms, 2) if gen_ms is not None else None
            record["retrieval_decision"] = dec
            record["retrieved_passage_count"] = len(ev_res.evidence)
            record["routing_path"] = route_str
            record["memory_hit"] = ev_res.memory_hit or (ev_res.response_mode == "experience_assisted")
            record["memory_match_id"] = ev_res.experience_id or (ev_res.memory_experience_ids[0] if getattr(ev_res, "memory_experience_ids", None) else None)
            record["memory_match_score"] = round(ev_res.similarity_score, 4) if ev_res.similarity_score is not None else None
            record["citations"] = _extract_citations(ev_res.evidence)
            record["evidence"] = ev_res.evidence
            return record

        # -------------------------------------------------------------
        # System 4: Original Self-RAG MLX (Base Model, No Adapter)
        # -------------------------------------------------------------
        elif system_id == "selfrag_mlx":
            record["system_name"] = "Original Self-RAG (MLX Base)"
            record["backend"] = "Apple Silicon MLX (Metal GPU)"
            record["model"] = f"{args.mlx_model} (No Adapter)"

            from retriever.retriever import PassageRetriever
            from evorag.mlx_generator import MLXSelfRAGGenerator
            from selfrag.pipeline import self_rag_answer

            if not Path(args.mlx_model).exists():
                record["answer_status"] = "Model Error: MLX base model not found"
                record["error"] = f"Directory not found: {args.mlx_model}"
                return record

            retriever = TimedRetriever(PassageRetriever(index_dir))

            t0_load = time.perf_counter()
            generator = MLXSelfRAGGenerator(
                model_path=args.mlx_model,
                adapter_path=None,  # Crucial: base model without fine-tuned adapter!
                max_tokens=max_tokens,
            )
            record["model_load_time_ms"] = round((time.perf_counter() - t0_load) * 1000, 2)

            t0_query = time.perf_counter()
            sr_res = self_rag_answer(question, generator, retriever, k=k, max_segments=max_segments)
            total_ms = (time.perf_counter() - t0_query) * 1000
            retrieval_ms = retriever.total_retrieval_ms
            gen_ms = max(0.0, total_ms - retrieval_ms)

            passages = [c.passage for seg in sr_res.segments for c in getattr(seg, "candidates", []) if hasattr(c, "passage")]

            record["success"] = True
            record["answer"] = sr_res.final_answer
            record["answer_status"] = "Success"
            record["total_query_latency_ms"] = round(total_ms, 2)
            record["retrieval_latency_ms"] = round(retrieval_ms, 2) if retriever.call_count > 0 else 0.0
            record["generation_latency_ms"] = round(gen_ms, 2)
            record["retrieval_decision"] = sr_res.decision or ("[Retrieval]" if sr_res.retrieved else "[No Retrieval]")
            record["retrieved_passage_count"] = len(passages)
            record["citations"] = _extract_citations(passages)
            record["evidence"] = passages
            return record

        # -------------------------------------------------------------
        # System 5: Fine-Tuned EvoRAG MLX (QLoRA + Experience Memory)
        # -------------------------------------------------------------
        elif system_id == "evorag_mlx":
            record["system_name"] = "Fine-Tuned EvoRAG (MLX QLoRA)"
            record["backend"] = "Apple Silicon MLX (Metal GPU) + SQLite/FAISS Memory"
            record["model"] = f"{args.mlx_model} + {args.mlx_adapter}"

            from retriever.retriever import PassageRetriever
            from evorag.mlx_generator import MLXSelfRAGGenerator
            from evorag.controller import EvoRAGController
            from evorag.memory.store import ExperienceStore
            from evorag.memory.retriever import ExperienceRetriever
            from evorag.memory.validator import ExperienceValidator
            from evorag.memory.index_builder import ExperienceIndexBuilder

            if not Path(args.mlx_model).exists():
                record["answer_status"] = "Model Error: MLX base model not found"
                record["error"] = f"Directory not found: {args.mlx_model}"
                return record

            if args.mlx_adapter and not Path(args.mlx_adapter).exists():
                record["answer_status"] = "Adapter Error: MLX adapter not found"
                record["error"] = f"Directory not found: {args.mlx_adapter}"
                return record

            store = ExperienceStore(memory_db)
            mem_retriever = ExperienceRetriever(memory_index, store=store)
            validator = ExperienceValidator(similarity_threshold=threshold)
            index_builder = ExperienceIndexBuilder(index_dir=memory_index)
            retriever = TimedRetriever(PassageRetriever(index_dir))

            t0_load = time.perf_counter()
            generator = MLXSelfRAGGenerator(
                model_path=args.mlx_model,
                adapter_path=args.mlx_adapter,
                max_tokens=max_tokens,
            )
            record["model_load_time_ms"] = round((time.perf_counter() - t0_load) * 1000, 2)

            controller = EvoRAGController(
                store=store,
                memory_retriever=mem_retriever,
                validator=validator,
                generator=generator,
                retriever=retriever,
                doc_index_dir=index_dir,
                index_builder=index_builder,
                k=k,
                max_segments=max_segments,
            )

            t0_query = time.perf_counter()
            try:
                ev_res = controller.answer(question)
            finally:
                store.close()
            total_ms = (time.perf_counter() - t0_query) * 1000

            if ev_res.response_mode == "direct_reuse":
                route_str = "Path A (Direct Reuse)"
                retrieval_ms = 0.0
                gen_ms = 0.0
            elif ev_res.response_mode == "experience_assisted":
                route_str = "Path B (Experience-Assisted Self-RAG)"
                retrieval_ms = retriever.total_retrieval_ms
                gen_ms = max(0.0, total_ms - retrieval_ms)
            else:
                route_str = "Path C (Normal Self-RAG Fallback)"
                retrieval_ms = retriever.total_retrieval_ms
                gen_ms = max(0.0, total_ms - retrieval_ms)

            dec = None
            if ev_res.self_rag_result:
                dec = ev_res.self_rag_result.decision or ("[Retrieval]" if ev_res.self_rag_result.retrieved else "[No Retrieval]")
            elif ev_res.response_mode == "direct_reuse":
                dec = "[Memory Hit - No LLM Probe]"

            record["success"] = True
            record["answer"] = ev_res.answer
            record["answer_status"] = "Success"
            record["total_query_latency_ms"] = round(total_ms, 2)
            record["retrieval_latency_ms"] = round(retrieval_ms, 2) if retrieval_ms is not None else None
            record["generation_latency_ms"] = round(gen_ms, 2) if gen_ms is not None else None
            record["retrieval_decision"] = dec
            record["retrieved_passage_count"] = len(ev_res.evidence)
            record["routing_path"] = route_str
            record["memory_hit"] = ev_res.memory_hit or (ev_res.response_mode == "experience_assisted")
            record["memory_match_id"] = ev_res.experience_id or (ev_res.memory_experience_ids[0] if getattr(ev_res, "memory_experience_ids", None) else None)
            record["memory_match_score"] = round(ev_res.similarity_score, 4) if ev_res.similarity_score is not None else None
            record["citations"] = _extract_citations(ev_res.evidence)
            record["evidence"] = ev_res.evidence
            return record

        else:
            record["answer_status"] = f"Unknown system: {system_id}"
            record["error"] = f"Unrecognized system ID: {system_id}"
            return record

    except Exception as e:
        record["answer_status"] = f"Execution Error: {e}"
        record["error"] = str(e)
        return record


def main():
    parser = argparse.ArgumentParser(description="Live Comparison Worker")
    parser.add_argument("--system", required=True, choices=["normal", "selfrag_gguf", "evorag_gguf", "selfrag_mlx", "evorag_mlx"])
    parser.add_argument("--question", required=True)
    parser.add_argument("--category", default="pdf", choices=["pdf", "general"])
    parser.add_argument("--index", default="index/dbms_sql_v1")
    parser.add_argument("--memory-db", default="data/experience_memory.db")
    parser.add_argument("--memory-index", default="data")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--model", default="models/selfrag_llama2_7b.Q4_K_M.gguf")
    parser.add_argument("--mlx-model", default="models/mlx/selfrag_llama2_7b_4bit")
    parser.add_argument("--mlx-adapter", default="training/qlora_dbms/adapters/full_dbms_v1")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--output-json", default=None)

    args = parser.parse_args()
    result = run_worker(args)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

