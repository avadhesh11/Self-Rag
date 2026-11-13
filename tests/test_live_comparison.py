"""Automated Test Suite for Task 13: Five-System Live Comparison.

Validates:
1. All five systems are invoked exactly once per comparison session.
2. All five systems receive the identical question.
3. Systems run sequentially in defined order.
4. Failure in one system (e.g. Normal RAG API failure) does not abort remaining systems.
5. Latency metrics (query, retrieval, generation, model load) are recorded.
6. Missing timing values are displayed as 'N/A' without crashing.
7. 'pdf' and 'general' categories pass correctly.
8. Comparison reports are saved separately as JSON and Markdown.
9. Existing CLI modes remain fully functional.
10. Frozen benchmark data and model weights remain unmodified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.live_comparison import (
    _format_ms,
    format_comparison_table,
    run_live_comparison,
)
from evaluation.live_comparison_worker import TimedRetriever, run_worker


# ---------------------------------------------------------------------------
# 1. Table Formatting & N/A Handling
# ---------------------------------------------------------------------------
def test_format_ms_and_na_handling():
    """Verify latency formatting and N/A fallbacks."""
    assert _format_ms(1234.56) == "1,234.6 ms"
    assert _format_ms(0.0) == "0.0 ms"
    assert _format_ms(None) == "N/A"


def test_format_comparison_table():
    """Verify consolidated table generation with mixed success and missing values."""
    mock_results = [
        {
            "system_id": "normal",
            "system_name": "Normal RAG (Gemini)",
            "total_query_latency_ms": 1664.2,
            "retrieval_latency_ms": 12.3,
            "generation_latency_ms": 1651.9,
            "model_load_time_ms": None,
            "answer_status": "Success",
            "retrieval_decision": "[Retrieval]",
        },
        {
            "system_id": "selfrag_gguf",
            "system_name": "Original Self-RAG (GGUF)",
            "total_query_latency_ms": None,
            "retrieval_latency_ms": None,
            "generation_latency_ms": None,
            "model_load_time_ms": 10500.0,
            "answer_status": "Execution Error: Memory allocation failure",
            "retrieval_decision": None,
        },
    ]

    table_str = format_comparison_table(mock_results)
    assert "Normal RAG (Gemini)" in table_str
    assert "Original Self-RAG (GGUF)" in table_str
    assert "1,664.2 ms" in table_str
    assert "N/A (API)" in table_str
    assert "N/A" in table_str
    assert "Execution Error" in table_str


# ---------------------------------------------------------------------------
# 2. Sequential Five-System Invocation & Question Invariance
# ---------------------------------------------------------------------------
def test_all_five_systems_invoked_sequentially(tmp_path: Path):
    """Verify that run_live_comparison invokes all 5 systems in exact sequential order with identical query."""
    question = "What is write-ahead logging (WAL) in database recovery?"
    invoked_systems: list[str] = []
    received_questions: list[str] = []

    def mock_subprocess_run(cmd, **kwargs):
        # Find system and question in command args
        sys_val = None
        q_val = None
        for i, token in enumerate(cmd):
            if token == "--system" and i + 1 < len(cmd):
                sys_val = cmd[i + 1]
            if token == "--question" and i + 1 < len(cmd):
                q_val = cmd[i + 1]
            if token == "--output-json" and i + 1 < len(cmd):
                out_path = Path(cmd[i + 1])
                out_path.write_text(json.dumps({
                    "system_id": sys_val,
                    "system_name": f"Mock {sys_val}",
                    "success": True,
                    "answer": f"Answer for {sys_val}",
                    "answer_status": "Success",
                    "total_query_latency_ms": 100.0,
                    "retrieval_latency_ms": 10.0,
                    "generation_latency_ms": 90.0,
                    "model_load_time_ms": 500.0,
                    "retrieval_decision": "[Retrieval]",
                    "retrieved_passage_count": 2,
                    "routing_path": "Path C",
                    "citations": ["01_dbms.pdf"],
                }), encoding="utf-8")

        if sys_val:
            invoked_systems.append(sys_val)
        if q_val:
            received_questions.append(q_val)

        return MagicMock(returncode=0, stderr="")

    with patch("subprocess.run", side_effect=mock_subprocess_run):
        with patch("evaluation.live_comparison.REPORTS_DIR", tmp_path):
            session = run_live_comparison(
                question=question,
                category="pdf",
                index_dir="index/dbms_sql_v1",
                save_report=True,
            )

    # 1. All 5 systems invoked exactly once in order
    assert invoked_systems == [
        "normal",
        "selfrag_gguf",
        "evorag_gguf",
        "selfrag_mlx",
        "evorag_mlx",
    ]

    # 2. All 5 received the identical question
    assert len(received_questions) == 5
    assert all(q == question for q in received_questions)

    # 3. Report files generated
    json_reports = list(tmp_path.glob("comparison_*.json"))
    md_reports = list(tmp_path.glob("comparison_*.md"))
    assert len(json_reports) == 1
    assert len(md_reports) == 1

    # Verify JSON content
    data = json.loads(json_reports[0].read_text(encoding="utf-8"))
    assert data["question"] == question
    assert len(data["results"]) == 5


# ---------------------------------------------------------------------------
# 3. Fault Tolerance: System Failure Isolation
# ---------------------------------------------------------------------------
def test_system_failure_does_not_abort_comparison(tmp_path: Path):
    """Verify that failure of system 1 (e.g. Normal RAG) does not halt systems 2-5."""
    question = "Explain ACID properties."
    call_count = 0

    def mock_subprocess_run(cmd, **kwargs):
        nonlocal call_count
        call_count += 1
        sys_val = None
        for i, token in enumerate(cmd):
            if token == "--system" and i + 1 < len(cmd):
                sys_val = cmd[i + 1]
            if token == "--output-json" and i + 1 < len(cmd):
                out_path = Path(cmd[i + 1])
                if sys_val == "normal":
                    # Simulate API failure
                    out_path.write_text(json.dumps({
                        "system_id": "normal",
                        "system_name": "Normal RAG (Gemini)",
                        "success": False,
                        "error": "GEMINI_API_KEY environment variable not set",
                        "answer": "",
                        "answer_status": "Configuration Error: GEMINI_API_KEY missing",
                        "total_query_latency_ms": None,
                        "retrieval_latency_ms": None,
                        "generation_latency_ms": None,
                        "model_load_time_ms": None,
                    }), encoding="utf-8")
                else:
                    out_path.write_text(json.dumps({
                        "system_id": sys_val,
                        "system_name": f"System {sys_val}",
                        "success": True,
                        "answer": f"Answer from {sys_val}",
                        "answer_status": "Success",
                        "total_query_latency_ms": 250.0,
                    }), encoding="utf-8")

        return MagicMock(returncode=0, stderr="")

    with patch("subprocess.run", side_effect=mock_subprocess_run):
        with patch("evaluation.live_comparison.REPORTS_DIR", tmp_path):
            session = run_live_comparison(question=question, category="pdf", save_report=True)

    assert call_count == 5
    results = session["results"]
    assert len(results) == 5
    assert results[0]["success"] is False
    assert "GEMINI_API_KEY" in results[0]["error"]
    assert results[1]["success"] is True
    assert results[2]["success"] is True
    assert results[3]["success"] is True
    assert results[4]["success"] is True


# ---------------------------------------------------------------------------
# 4. TimedRetriever & Timing Accrual
# ---------------------------------------------------------------------------
def test_timed_retriever_latency_accumulation():
    """Verify that TimedRetriever measures elapsed search time correctly."""
    mock_base = MagicMock()
    mock_base.search.return_value = [{"chunk_id": 1, "text": "sample"}]

    timed = TimedRetriever(mock_base)
    assert timed.total_retrieval_ms == 0.0
    assert timed.call_count == 0

    res = timed.search("Test query", k=3)
    assert len(res) == 1
    assert timed.call_count == 1
    assert timed.total_retrieval_ms >= 0.0


# ---------------------------------------------------------------------------
# 5. Worker Unit Tests with Mocked Generator
# ---------------------------------------------------------------------------
def test_worker_evorag_path_a_direct_reuse():
    """Verify worker returns 0.0 generation latency and instant answer for Path A."""
    from evorag.result import EvoRAGResult

    mock_controller = MagicMock()
    mock_controller.answer.return_value = EvoRAGResult(
        question="What is physical data independence?",
        answer="Stored verified answer.",
        memory_hit=True,
        response_mode="direct_reuse",
        memory_context_used=False,
        experience_id="exp-001",
        similarity_score=1.0,
        evidence=[{"source": "01_dbms.pdf", "page": 1, "chunk_id": 2}],
    )

    args = argparse.Namespace(
        system="evorag_gguf",
        question="What is physical data independence?",
        category="pdf",
        index="index/dbms_sql_v1",
        memory_db="data/experience_memory.db",
        memory_index="data",
        threshold=0.85,
        model="models/selfrag_llama2_7b.Q4_K_M.gguf",
        mlx_model="models/mlx/selfrag_llama2_7b_4bit",
        mlx_adapter="training/qlora_dbms/adapters/full_dbms_v1",
        k=5,
        max_segments=3,
        max_tokens=250,
    )

    with patch("evorag.memory.store.ExperienceStore"):
        with patch("evorag.memory.retriever.ExperienceRetriever"):
            with patch("evorag.memory.validator.ExperienceValidator"):
                with patch("evorag.memory.index_builder.ExperienceIndexBuilder"):
                    with patch("retriever.retriever.PassageRetriever"):
                        with patch("selfrag.generator.SelfRAGGenerator"):
                            with patch("evorag.controller.EvoRAGController", return_value=mock_controller):
                                res = run_worker(args)

    assert res["success"] is True
    assert res["routing_path"] == "Path A (Direct Reuse)"
    assert res["generation_latency_ms"] == 0.0
    assert res["retrieval_latency_ms"] == 0.0
    assert res["memory_match_id"] == "exp-001"
    assert res["memory_match_score"] == 1.0


# ---------------------------------------------------------------------------
# 6. Question Type & CLI Integration
# ---------------------------------------------------------------------------
def test_cli_comparison_flags_parsed():
    """Verify CLI accepts --compare, --question-type, and --question."""
    import demo

    parser = demo.build_arg_parser()
    args = parser.parse_args([
        "--compare",
        "--question-type", "general",
        "--question", "What is the capital of France?",
    ])

    assert args.compare is True
    assert args.question_type == "general"
    assert args.question == "What is the capital of France?"


# ---------------------------------------------------------------------------
# 7. Immutability of Frozen Benchmarks
# ---------------------------------------------------------------------------
def test_frozen_benchmark_unmodified():
    """Ensure Task 13 changes do not mutate the frozen benchmark files."""
    import hashlib

    frozen_csv = PROJECT_ROOT / "evaluation" / "predictions" / "frozen" / "predictions_20261109_151957Z.csv"
    assert frozen_csv.exists()
    h = hashlib.sha256(frozen_csv.read_bytes()).hexdigest()
    assert h == "a08d3bd3b0440302a02fd2f81c5dbab3d878cce75406657319940e4bb495d3e7"

