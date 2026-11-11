"""
EvoRAG Blind Evaluation Runner and Prediction Freeze (Task 9A.1).

Executes a reproducible, blind evaluation of 32 held-out test questions across:
1. Normal RAG
2. Self-RAG
3. EvoRAG (Hybrid Experience Learning)

Key Safeguards:
- No answer key, ground-truth label, or scoring data is loaded or inspected.
- The held-out questions are strictly read-only and never modified.
- Held-out questions and predictions are NEVER used to create, approve, or update
  experiences before or during evaluation (no memory contamination).
- All systems share the exact same underlying document corpus (e.g. index/dbms_sql_v1).
- Freezes all predictions in structured JSONL and CSV formats with complete runtime metadata.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from retriever.retriever import PassageRetriever
from normal_rag.pipeline import normal_rag_answer, NormalRAGResult
from selfrag.pipeline import self_rag_answer, SelfRAGResult
from evorag.controller import EvoRAGController
from evorag.memory.store import ExperienceStore
from evorag.memory.retriever import ExperienceRetriever
from evorag.memory.validator import ExperienceValidator
from evorag.config import (
    DEFAULT_DB_PATH,
    DEFAULT_INDEX_DIR,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
)

CSV_COLUMNS = [
    "question_id",
    "system",
    "predicted_answer",
    "cited_source_ids",
    "cited_pages",
    "retrieved_chunk_ids",
    "retrieved_passage_count",
    "retrieval_segment_count",
    "llm_call_count",
    "latency_ms",
    "memory_candidate_found",
    "memory_hit",
    "memory_reuse",
    "fallback_reason",
]


def load_held_out_questions(questions_path: Path) -> List[Dict[str, str]]:
    """Load held-out evaluation questions. Verifies schema without loading answer keys."""
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "question_id" not in item or "question" not in item:
                raise ValueError(f"Line {line_num} in {questions_path} missing required fields.")

            # Blind validation safeguard: verify no answer key fields exist
            forbidden_keys = {"answer", "expected_answer", "ground_truth", "label", "target"}
            present_forbidden = forbidden_keys.intersection(item.keys())
            if present_forbidden:
                raise ValueError(
                    f"Blind evaluation violation: Question {item['question_id']} contains "
                    f"forbidden answer key fields: {present_forbidden}"
                )

            questions.append({
                "question_id": str(item["question_id"]),
                "question": str(item["question"]),
            })

    if not questions:
        raise ValueError(f"No questions found in {questions_path}")

    return questions


def _extract_citations_and_chunks(passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Helper to extract distinct source IDs, page numbers, and chunk IDs from passage dicts."""
    sources = []
    pages = []
    chunk_ids = []

    for p in passages:
        if not isinstance(p, dict):
            continue
        src = p.get("source") or p.get("source_id")
        if src and src not in sources:
            sources.append(str(src))

        pg = p.get("page")
        if pg is not None and str(pg) not in pages:
            pages.append(str(pg))

        cid = p.get("chunk_id")
        if cid is not None and str(cid) not in chunk_ids:
            chunk_ids.append(str(cid))

    return {
        "sources": sources,
        "pages": pages,
        "chunk_ids": chunk_ids,
        "count": len(passages),
    }


def run_normal_rag_eval(
    q_item: Dict[str, str],
    normal_generator: Any,
    retriever: PassageRetriever,
    k: int = 5,
) -> Dict[str, Any]:
    """Execute question on Normal RAG and capture timing and trace."""
    qid = q_item["question_id"]
    query = q_item["question"]

    t_start = time.perf_counter()
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None

    try:
        # Measure retrieval separately
        t_ret_0 = time.perf_counter()
        passages = retriever.search(query, k=k)
        retrieval_ms = round((time.perf_counter() - t_ret_0) * 1000, 2)

        # Measure generation separately
        t_gen_0 = time.perf_counter()
        answer = normal_generator.generate(query, passages)
        generation_ms = round((time.perf_counter() - t_gen_0) * 1000, 2)
        total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        extracted = _extract_citations_and_chunks(passages)

        return {
            "question_id": qid,
            "question": query,
            "system": "normal_rag",
            "predicted_answer": answer,
            "cited_source_ids": extracted["sources"],
            "cited_pages": extracted["pages"],
            "retrieved_chunk_ids": extracted["chunk_ids"],
            "retrieved_passage_count": extracted["count"],
            "retrieval_segment_count": 1,
            "llm_call_count": 1,
            "latency_ms": total_latency_ms,
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": generation_ms,
            "memory_candidate_found": False,
            "memory_hit": False,
            "memory_reuse": False,
            "fallback_reason": None,
            "retrieved_passages": passages,
            "citations": extracted["sources"],
            "routing_path": "normal_rag",
            "selfrag_trace": None,
            "error": None,
        }
    except Exception as exc:
        total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return {
            "question_id": qid,
            "question": query,
            "system": "normal_rag",
            "predicted_answer": "",
            "cited_source_ids": [],
            "cited_pages": [],
            "retrieved_chunk_ids": [],
            "retrieved_passage_count": 0,
            "retrieval_segment_count": 0,
            "llm_call_count": 0,
            "latency_ms": total_latency_ms,
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": generation_ms,
            "memory_candidate_found": False,
            "memory_hit": False,
            "memory_reuse": False,
            "fallback_reason": None,
            "retrieved_passages": [],
            "citations": [],
            "routing_path": "normal_rag",
            "selfrag_trace": None,
            "error": str(exc),
        }


def run_self_rag_eval(
    q_item: Dict[str, str],
    self_generator: Any,
    retriever: PassageRetriever,
    k: int = 5,
    max_segments: int = 3,
) -> Dict[str, Any]:
    """Execute question on Self-RAG and capture multi-segment traces and latency."""
    qid = q_item["question_id"]
    query = q_item["question"]

    t_start = time.perf_counter()

    try:
        res: SelfRAGResult = self_rag_answer(
            query,
            self_generator,
            retriever,
            k=k,
            max_segments=max_segments,
        )
        total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Collect evidence passages across all segments
        all_passages = []
        if getattr(res, "segments", None):
            for seg in res.segments:
                if getattr(seg, "evidence", None):
                    all_passages.extend(seg.evidence)
        if not all_passages and getattr(res, "candidates", None):
            for cand in res.candidates:
                if getattr(cand, "passage", None):
                    all_passages.append(cand.passage)

        extracted = _extract_citations_and_chunks(all_passages)

        # Calculate segments and approx LLM calls
        seg_count = len(res.segments) if getattr(res, "segments", None) else (1 if res.retrieved else 0)
        # 1 decision call + candidate batch calls per retrieval segment
        llm_calls = 1
        if getattr(res, "segments", None):
            for seg in res.segments:
                if seg.decision == "[Retrieval]":
                    llm_calls += len(seg.candidates) if seg.candidates else 1
                elif seg.decision == "[Continue to Use Evidence]":
                    llm_calls += 1

        trace = {
            "retrieved": res.retrieved,
            "decision": res.decision,
            "normalized_query": res.normalized_query,
            "num_segments": len(res.segments) if getattr(res, "segments", None) else 0,
            "segments": [
                {
                    "index": s.index,
                    "decision": s.decision,
                    "reused_evidence": s.reused_evidence,
                    "answer_text": s.answer_text,
                    "num_candidates": len(s.candidates),
                    "selected_score": s.selected.score if s.selected else None,
                    "selected_isrel": s.selected.isrel if s.selected else None,
                    "selected_issup": s.selected.issup if s.selected else None,
                    "selected_isuse": s.selected.isuse if s.selected else None,
                }
                for s in res.segments
            ] if getattr(res, "segments", None) else [],
        }

        return {
            "question_id": qid,
            "question": query,
            "system": "self_rag",
            "predicted_answer": res.final_answer,
            "cited_source_ids": extracted["sources"],
            "cited_pages": extracted["pages"],
            "retrieved_chunk_ids": extracted["chunk_ids"],
            "retrieved_passage_count": extracted["count"],
            "retrieval_segment_count": seg_count,
            "llm_call_count": llm_calls,
            "latency_ms": total_latency_ms,
            "retrieval_latency_ms": None,
            "generation_latency_ms": None,
            "memory_candidate_found": False,
            "memory_hit": False,
            "memory_reuse": False,
            "fallback_reason": None,
            "retrieved_passages": all_passages,
            "citations": extracted["sources"],
            "routing_path": "self_rag",
            "selfrag_trace": trace,
            "error": None,
        }
    except Exception as exc:
        total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return {
            "question_id": qid,
            "question": query,
            "system": "self_rag",
            "predicted_answer": "",
            "cited_source_ids": [],
            "cited_pages": [],
            "retrieved_chunk_ids": [],
            "retrieved_passage_count": 0,
            "retrieval_segment_count": 0,
            "llm_call_count": 0,
            "latency_ms": total_latency_ms,
            "retrieval_latency_ms": None,
            "generation_latency_ms": None,
            "memory_candidate_found": False,
            "memory_hit": False,
            "memory_reuse": False,
            "fallback_reason": None,
            "retrieved_passages": [],
            "citations": [],
            "routing_path": "self_rag",
            "selfrag_trace": None,
            "error": str(exc),
        }


def run_evorag_eval(
    q_item: Dict[str, str],
    controller: EvoRAGController,
) -> Dict[str, Any]:
    """Execute question through EvoRAGController recording three-path decisions."""
    qid = q_item["question_id"]
    query = q_item["question"]

    t_start = time.perf_counter()

    try:
        # Strict safeguard: controller answer ONLY. Never approve_and_save!
        res = controller.answer(query)
        total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        extracted = _extract_citations_and_chunks(res.evidence)

        # Tracing memory candidate details
        memory_candidate_found = bool(res.similarity_score is not None and res.similarity_score > 0)
        memory_hit = bool(res.memory_hit)
        memory_reuse = bool(res.response_mode == "direct_reuse")

        # Self-RAG segment tracing if Self-RAG was executed (Paths B or C)
        self_res = res.self_rag_result
        seg_count = 0
        llm_calls = 0
        trace = None

        if self_res:
            seg_count = len(self_res.segments) if getattr(self_res, "segments", None) else (1 if self_res.retrieved else 0)
            llm_calls = 1
            if getattr(self_res, "segments", None):
                for seg in self_res.segments:
                    if seg.decision == "[Retrieval]":
                        llm_calls += len(seg.candidates) if seg.candidates else 1
                    elif seg.decision == "[Continue to Use Evidence]":
                        llm_calls += 1

            trace = {
                "retrieved": self_res.retrieved,
                "decision": self_res.decision,
                "num_segments": len(self_res.segments) if getattr(self_res, "segments", None) else 0,
                "segments": [
                    {
                        "index": s.index,
                        "decision": s.decision,
                        "reused_evidence": s.reused_evidence,
                        "answer_text": s.answer_text,
                        "num_candidates": len(s.candidates),
                        "selected_score": s.selected.score if s.selected else None,
                    }
                    for s in self_res.segments
                ] if getattr(self_res, "segments", None) else [],
            }
        elif memory_reuse:
            seg_count = 0
            llm_calls = 0

        return {
            "question_id": qid,
            "question": query,
            "system": "evorag",
            "predicted_answer": res.answer,
            "cited_source_ids": extracted["sources"],
            "cited_pages": extracted["pages"],
            "retrieved_chunk_ids": extracted["chunk_ids"],
            "retrieved_passage_count": extracted["count"],
            "retrieval_segment_count": seg_count,
            "llm_call_count": llm_calls,
            "latency_ms": total_latency_ms,
            "retrieval_latency_ms": None,
            "generation_latency_ms": None,
            "memory_candidate_found": memory_candidate_found,
            "memory_hit": memory_hit,
            "memory_reuse": memory_reuse,
            "fallback_reason": res.fallback_reason,
            "experience_id": res.experience_id,
            "similarity_score": res.similarity_score,
            "routing_path": res.response_mode,
            "retrieved_passages": res.evidence,
            "citations": extracted["sources"],
            "selfrag_trace": trace,
            "error": None,
        }
    except Exception as exc:
        total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return {
            "question_id": qid,
            "question": query,
            "system": "evorag",
            "predicted_answer": "",
            "cited_source_ids": [],
            "cited_pages": [],
            "retrieved_chunk_ids": [],
            "retrieved_passage_count": 0,
            "retrieval_segment_count": 0,
            "llm_call_count": 0,
            "latency_ms": total_latency_ms,
            "retrieval_latency_ms": None,
            "generation_latency_ms": None,
            "memory_candidate_found": False,
            "memory_hit": False,
            "memory_reuse": False,
            "fallback_reason": None,
            "experience_id": None,
            "similarity_score": None,
            "routing_path": "error",
            "retrieved_passages": [],
            "citations": [],
            "selfrag_trace": None,
            "error": str(exc),
        }


def execute_blind_evaluation(
    questions_file: Path,
    out_dir: Path,
    systems: List[str],
    index_dir: str = "index/dbms_sql_v1",
    model_path: Optional[str] = None,
    gemini_model: Optional[str] = None,
    memory_db_path: Optional[str] = None,
    memory_index_dir: Optional[str] = None,
    k: int = 5,
    max_segments: int = 3,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    normal_generator: Any = None,
    self_generator: Any = None,
    controller: Any = None,
    retriever: Any = None,
) -> Dict[str, Any]:
    """Orchestrates blind evaluation across all 32 questions and freezes predictions."""
    questions = load_held_out_questions(questions_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

    # 1. Initialize shared PassageRetriever if not provided
    if retriever is None:
        if not Path(index_dir).exists():
            raise FileNotFoundError(f"Document index not found at {index_dir}")
        print(f"[EVAL] Loading shared document retriever from {index_dir} ...")
        retriever = PassageRetriever(index_dir)

    # 2. Lazily load generator instances as needed
    if "normal_rag" in systems and normal_generator is None:
        from normal_rag.generator import GeminiGenerator
        print(f"[EVAL] Initializing Gemini generator ({gemini_model or 'default'}) ...")
        normal_generator = GeminiGenerator(model=gemini_model)

    if ("self_rag" in systems or "evorag" in systems) and self_generator is None:
        resolved_model = model_path or os.environ.get(
            "SELF_RAG_MODEL_PATH", "models/selfrag_llama2_7b.Q4_K_M.gguf"
        )
        if not Path(resolved_model).exists() and os.environ.get("SELFRAG_BACKEND") != "vllm":
            raise FileNotFoundError(
                f"Self-RAG model checkpoint not found at {resolved_model}."
            )
        from selfrag.generator import SelfRAGGenerator
        print(f"[EVAL] Initializing Self-RAG generator ({resolved_model}) ...")
        self_generator = SelfRAGGenerator(resolved_model)

    # 3. Lazily initialize EvoRAG controller if not provided
    if "evorag" in systems and controller is None:
        db_path = memory_db_path or DEFAULT_DB_PATH
        m_index = memory_index_dir or DEFAULT_INDEX_DIR
        print(f"[EVAL] Initializing EvoRAG Controller (db: {db_path}, memory_index: {m_index}) ...")
        store = ExperienceStore(db_path)
        memory_retriever = ExperienceRetriever(index_dir=m_index, store=store)
        validator = ExperienceValidator(similarity_threshold=threshold)

        controller = EvoRAGController(
            store=store,
            memory_retriever=memory_retriever,
            validator=validator,
            generator=self_generator,
            retriever=retriever,
            doc_index_dir=index_dir,
            k=k,
            max_segments=max_segments,
        )

    config_metadata = {
        "timestamp_utc": timestamp_str,
        "questions_file": str(questions_file),
        "num_questions": len(questions),
        "systems": systems,
        "document_index": str(index_dir),
        "k": k,
        "max_segments": max_segments,
        "similarity_threshold": threshold,
        "blind_evaluation_mode": True,
        "answer_key_loaded": False,
        "experience_saving_enabled": False,
    }

    detailed_predictions: List[Dict[str, Any]] = []
    template_rows: List[Dict[str, Any]] = []

    print(f"\n[EVAL] Starting blind evaluation for {len(questions)} questions across systems: {systems}")

    for idx, q_item in enumerate(questions, 1):
        qid = q_item["question_id"]
        q_text = q_item["question"]
        print(f"[{idx:02d}/{len(questions):02d}] {qid}: {q_text[:60]}...")

        for sys_name in systems:
            record: Dict[str, Any]
            if sys_name == "normal_rag":
                record = run_normal_rag_eval(q_item, normal_generator, retriever, k=k)
            elif sys_name == "self_rag":
                record = run_self_rag_eval(q_item, self_generator, retriever, k=k, max_segments=max_segments)
            elif sys_name == "evorag":
                record = run_evorag_eval(q_item, controller)
            else:
                raise ValueError(f"Unknown system name: {sys_name}")

            # Append config to record
            record["configuration"] = {
                "system": sys_name,
                "k": k,
                "document_index": str(index_dir),
            }
            detailed_predictions.append(record)

            # Build standardized template row
            template_rows.append({
                "question_id": qid,
                "system": sys_name,
                "predicted_answer": record.get("predicted_answer", ""),
                "cited_source_ids": ";".join(record.get("cited_source_ids", [])),
                "cited_pages": ";".join(record.get("cited_pages", [])),
                "retrieved_chunk_ids": ";".join(record.get("retrieved_chunk_ids", [])),
                "retrieved_passage_count": record.get("retrieved_passage_count", 0),
                "retrieval_segment_count": record.get("retrieval_segment_count", 0),
                "llm_call_count": record.get("llm_call_count", 0),
                "latency_ms": record.get("latency_ms", 0.0),
                "memory_candidate_found": record.get("memory_candidate_found", False),
                "memory_hit": record.get("memory_hit", False),
                "memory_reuse": record.get("memory_reuse", False),
                "fallback_reason": record.get("fallback_reason") or "",
            })

    # 4. Freeze predictions
    jsonl_path = out_dir / f"predictions_{timestamp_str}.jsonl"
    csv_path = out_dir / f"predictions_{timestamp_str}.csv"
    meta_path = out_dir / f"freeze_metadata_{timestamp_str}.json"

    # Write detailed JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in detailed_predictions:
            f.write(json.dumps(r) + "\n")

    # Write standardized CSV template
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in template_rows:
            writer.writerow(r)

    # Write execution and freeze metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "configuration": config_metadata,
            "prediction_counts": {
                "total_rows": len(template_rows),
                "per_system": {s: sum(1 for r in template_rows if r["system"] == s) for s in systems},
            },
            "output_files": {
                "detailed_jsonl": str(jsonl_path),
                "predictions_csv": str(csv_path),
            }
        }, f, indent=2)

    print(f"\n[EVAL COMPLETE] Predictions frozen successfully:")
    print(f"  -> JSONL : {jsonl_path}")
    print(f"  -> CSV   : {csv_path}")
    print(f"  -> Meta  : {meta_path}\n")

    return {
        "timestamp": timestamp_str,
        "jsonl_path": jsonl_path,
        "csv_path": csv_path,
        "meta_path": meta_path,
        "detailed_predictions": detailed_predictions,
        "template_rows": template_rows,
        "config_metadata": config_metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="EvoRAG Blind Evaluation Runner (Task 9A.1)")
    parser.add_argument(
        "--questions",
        default="EvoRAG_Blind_Test_Questions/held_out_test_questions_only.jsonl",
        help="Path to held-out test questions JSONL",
    )
    parser.add_argument(
        "--out-dir",
        default="evaluation/predictions/frozen",
        help="Output directory for frozen predictions",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["normal_rag", "self_rag", "evorag"],
        choices=["normal_rag", "self_rag", "evorag"],
        help="Systems to evaluate",
    )
    parser.add_argument(
        "--index",
        default="index/dbms_sql_v1",
        help="Path to shared document FAISS index directory",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SELF_RAG_MODEL_PATH", "models/selfrag_llama2_7b.Q4_K_M.gguf"),
        help="Path to Self-RAG GGUF model",
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help="Gemini model identifier for Normal RAG",
    )
    parser.add_argument(
        "--memory-db",
        default=DEFAULT_DB_PATH,
        help="Path to SQLite Experience Memory database",
    )
    parser.add_argument(
        "--memory-index",
        default=DEFAULT_INDEX_DIR,
        help="Path to FAISS Experience Memory index directory",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Passages to retrieve",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=3,
        help="Maximum Self-RAG segments",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Similarity threshold for Experience Memory validation",
    )

    args = parser.parse_args()

    execute_blind_evaluation(
        questions_file=Path(args.questions),
        out_dir=Path(args.out_dir),
        systems=args.systems,
        index_dir=args.index,
        model_path=args.model,
        gemini_model=args.gemini_model,
        memory_db_path=args.memory_db,
        memory_index_dir=args.memory_index,
        k=args.k,
        max_segments=args.max_segments,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()

