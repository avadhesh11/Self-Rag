"""
Automated Unit Tests for EvoRAG Frozen Prediction Scoring (Task 9A.2).

Validates:
1. Question ID and system matching.
2. Duplicate prediction record detection.
3. Missing prediction detection.
4. Answer-key schema validation (missing IDs, empty answers, duplicates, bad formats).
5. Exact match and token-level F1 calculations with normalization.
6. Citation correctness metrics (precision, recall, F1).
7. Latency calculation (mean, median, P95, min, max, std, hit vs. miss).
8. Memory-hit and efficiency metrics (hit rate, reuse rate, avoided calls).
9. Prediction file immutability (SHA-256 integrity).
10. Scoring report output schema (JSON, Markdown, CSV).
11. Zero inference calls during scoring (no LLM, generator, or embedder calls).
"""
import csv
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from evaluation.metrics import (
    compute_citation_metrics,
    compute_exact_match,
    compute_latency_summary,
    compute_token_f1,
    normalize_text,
)
from evaluation.score_predictions import (
    compute_comparative_analysis,
    compute_file_sha256,
    evaluate_system,
    execute_scoring,
    load_answer_key,
    load_predictions,
    validate_records,
)


@pytest.fixture
def sample_predictions_csv(tmp_path: Path) -> Path:
    """Create a valid synthetic 2-question x 3-system predictions CSV."""
    file_path = tmp_path / "predictions_20261110_000000Z.csv"
    rows = [
        {
            "question_id": "DBMS-EVAL-0001",
            "system": "normal_rag",
            "predicted_answer": "A B-tree is a balanced tree data structure.",
            "cited_source_ids": "doc_storage.pdf",
            "cited_pages": "1;2",
            "retrieved_chunk_ids": "101;102",
            "retrieved_passage_count": "2",
            "retrieval_segment_count": "1",
            "llm_call_count": "1",
            "latency_ms": "1500.0",
            "memory_candidate_found": "False",
            "memory_hit": "False",
            "memory_reuse": "False",
            "fallback_reason": "",
        },
        {
            "question_id": "DBMS-EVAL-0001",
            "system": "self_rag",
            "predicted_answer": "A B-tree is a self-balancing tree data structure.",
            "cited_source_ids": "doc_storage.pdf",
            "cited_pages": "1",
            "retrieved_chunk_ids": "101",
            "retrieved_passage_count": "2",
            "retrieval_segment_count": "2",
            "llm_call_count": "4",
            "latency_ms": "12000.0",
            "memory_candidate_found": "False",
            "memory_hit": "False",
            "memory_reuse": "False",
            "fallback_reason": "",
        },
        {
            "question_id": "DBMS-EVAL-0001",
            "system": "evorag",
            "predicted_answer": "A B-tree is a balanced tree data structure.",
            "cited_source_ids": "doc_storage.pdf",
            "cited_pages": "1",
            "retrieved_chunk_ids": "101",
            "retrieved_passage_count": "0",
            "retrieval_segment_count": "0",
            "llm_call_count": "0",
            "latency_ms": "25.0",
            "memory_candidate_found": "True",
            "memory_hit": "True",
            "memory_reuse": "True",
            "fallback_reason": "",
        },
        {
            "question_id": "DBMS-EVAL-0002",
            "system": "normal_rag",
            "predicted_answer": "Atomicity means all operations succeed or none do.",
            "cited_source_ids": "doc_acid.pdf",
            "cited_pages": "3",
            "retrieved_chunk_ids": "201",
            "retrieved_passage_count": "1",
            "retrieval_segment_count": "1",
            "llm_call_count": "1",
            "latency_ms": "1600.0",
            "memory_candidate_found": "False",
            "memory_hit": "False",
            "memory_reuse": "False",
            "fallback_reason": "",
        },
        {
            "question_id": "DBMS-EVAL-0002",
            "system": "self_rag",
            "predicted_answer": "Atomicity ensures all-or-nothing transaction execution.",
            "cited_source_ids": "doc_acid.pdf",
            "cited_pages": "3",
            "retrieved_chunk_ids": "201",
            "retrieved_passage_count": "2",
            "retrieval_segment_count": "2",
            "llm_call_count": "5",
            "latency_ms": "14000.0",
            "memory_candidate_found": "False",
            "memory_hit": "False",
            "memory_reuse": "False",
            "fallback_reason": "",
        },
        {
            "question_id": "DBMS-EVAL-0002",
            "system": "evorag",
            "predicted_answer": "Atomicity ensures all-or-nothing transaction execution.",
            "cited_source_ids": "doc_acid.pdf",
            "cited_pages": "3",
            "retrieved_chunk_ids": "201",
            "retrieved_passage_count": "2",
            "retrieval_segment_count": "2",
            "llm_call_count": "5",
            "latency_ms": "13900.0",
            "memory_candidate_found": "False",
            "memory_hit": "False",
            "memory_reuse": "False",
            "fallback_reason": "No candidate experiences found in Experience Memory.",
        },
    ]

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return file_path


@pytest.fixture
def sample_answer_key_jsonl(tmp_path: Path) -> Path:
    """Create a valid synthetic answer key JSONL file."""
    file_path = tmp_path / "answer_key.jsonl"
    entries = [
        {
            "question_id": "DBMS-EVAL-0001",
            "reference_answer": "A B-tree is a balanced tree data structure.",
            "reference_sources": ["doc_storage.pdf"],
        },
        {
            "question_id": "DBMS-EVAL-0002",
            "reference_answers": [
                "Atomicity guarantees all or nothing execution.",
                "Atomicity ensures all-or-nothing transaction execution.",
            ],
            "reference_sources": ["doc_acid.pdf"],
        },
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return file_path


# ---------------------------------------------------------------------------
# 1. Question ID & System Matching
# ---------------------------------------------------------------------------
def test_question_id_and_system_matching(sample_predictions_csv: Path, sample_answer_key_jsonl: Path):
    records, freeze_id, sha = load_predictions(sample_predictions_csv)
    assert len(records) == 6
    assert freeze_id == "20261110_000000Z"

    key, key_sha = load_answer_key(sample_answer_key_jsonl)
    assert len(key) == 2
    assert "DBMS-EVAL-0001" in key
    assert "DBMS-EVAL-0002" in key

    validation = validate_records(records, key)
    assert validation["total_records"] == 6
    assert validation["unique_questions"] == 2
    assert len(validation["missing_predictions"]) == 0
    assert len(validation["key_coverage_issues"]) == 0


# ---------------------------------------------------------------------------
# 2. Duplicate Prediction Detection
# ---------------------------------------------------------------------------
def test_duplicate_prediction_detection(tmp_path: Path):
    dup_file = tmp_path / "predictions_dup.csv"
    with open(dup_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "system",
                "predicted_answer",
                "latency_ms",
                "llm_call_count",
                "memory_candidate_found",
                "memory_hit",
                "memory_reuse",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "question_id": "DBMS-EVAL-0001",
            "system": "normal_rag",
            "predicted_answer": "ans1",
            "latency_ms": "100",
            "llm_call_count": "1",
            "memory_candidate_found": "False",
            "memory_hit": "False",
            "memory_reuse": "False",
        })
        writer.writerow({
            "question_id": "DBMS-EVAL-0001",
            "system": "normal_rag",
            "predicted_answer": "ans2 duplicate",
            "latency_ms": "100",
            "llm_call_count": "1",
            "memory_candidate_found": "False",
            "memory_hit": "False",
            "memory_reuse": "False",
        })

    with pytest.raises(ValueError, match="Duplicate prediction record"):
        load_predictions(dup_file)


# ---------------------------------------------------------------------------
# 3. Missing Prediction Detection
# ---------------------------------------------------------------------------
def test_missing_prediction_detection(tmp_path: Path):
    partial_file = tmp_path / "predictions_partial.csv"
    with open(partial_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "system",
                "predicted_answer",
                "latency_ms",
                "llm_call_count",
                "memory_candidate_found",
                "memory_hit",
                "memory_reuse",
            ],
        )
        writer.writeheader()
        # Question 1 has normal_rag and self_rag, but missing evorag
        writer.writerow({
            "question_id": "DBMS-EVAL-0001",
            "system": "normal_rag",
            "predicted_answer": "ans1",
            "latency_ms": "100",
            "llm_call_count": "1",
            "memory_candidate_found": "False",
            "memory_hit": "False",
            "memory_reuse": "False",
        })
        writer.writerow({
            "question_id": "DBMS-EVAL-0001",
            "system": "self_rag",
            "predicted_answer": "ans2",
            "latency_ms": "100",
            "llm_call_count": "1",
            "memory_candidate_found": "False",
            "memory_hit": "False",
            "memory_reuse": "False",
        })

    records, _, _ = load_predictions(partial_file)
    val = validate_records(records)
    assert len(val["missing_predictions"]) == 1
    assert val["missing_predictions"][0] == {"question_id": "DBMS-EVAL-0001", "system": "evorag"}


# ---------------------------------------------------------------------------
# 4. Answer-Key Schema Validation
# ---------------------------------------------------------------------------
def test_answer_key_schema_validation(tmp_path: Path):
    # Test missing question_id
    bad_key1 = tmp_path / "bad_key1.jsonl"
    bad_key1.write_text(json.dumps({"reference_answer": "Only answer without ID"}) + "\n")
    with pytest.raises(ValueError, match="missing question_id"):
        load_answer_key(bad_key1)

    # Test missing reference answer
    bad_key2 = tmp_path / "bad_key2.jsonl"
    bad_key2.write_text(json.dumps({"question_id": "DBMS-EVAL-0001"}) + "\n")
    with pytest.raises(ValueError, match="missing reference answer field"):
        load_answer_key(bad_key2)

    # Test empty string reference answer
    bad_key3 = tmp_path / "bad_key3.jsonl"
    bad_key3.write_text(json.dumps({"question_id": "DBMS-EVAL-0001", "expected_answer": "   "}) + "\n")
    with pytest.raises(ValueError, match="empty reference answer"):
        load_answer_key(bad_key3)

    # Test duplicate question_id in answer key
    bad_key4 = tmp_path / "bad_key4.jsonl"
    bad_key4.write_text(
        json.dumps({"question_id": "DBMS-EVAL-0001", "answer": "ans1"}) + "\n" +
        json.dumps({"question_id": "DBMS-EVAL-0001", "answer": "ans2"}) + "\n"
    )
    with pytest.raises(ValueError, match="Duplicate question_id"):
        load_answer_key(bad_key4)

    # Test unsupported file format
    bad_key5 = tmp_path / "bad_key5.txt"
    bad_key5.write_text("plain text")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_answer_key(bad_key5)


# ---------------------------------------------------------------------------
# 5. Exact Match & Token F1 Calculations
# ---------------------------------------------------------------------------
def test_exact_match_and_token_f1_calculations():
    # Identical normalized string
    em = compute_exact_match(
        "A B-tree is a balanced tree data structure.",
        "b-tree is balanced tree data structure"
    )
    assert em == 1.0

    # Punctuation and article stripping
    assert normalize_text("The B-Tree, an index...") == "btree index"

    # Token F1
    pred = "A B-tree is a balanced tree data structure."
    ref = "A B-tree is a self-balancing tree."
    f1_dict = compute_token_f1(pred, ref)
    assert f1_dict["precision"] > 0.0
    assert f1_dict["recall"] > 0.0
    assert f1_dict["f1"] > 0.0

    # Multiple references: returns max score
    refs = ["completely different words", "A B-tree is a balanced tree data structure."]
    em_multi = compute_exact_match(pred, refs)
    assert em_multi == 1.0
    f1_multi = compute_token_f1(pred, refs)
    assert f1_multi["f1"] == 1.0


# ---------------------------------------------------------------------------
# 6. Citation Metrics Calculation
# ---------------------------------------------------------------------------
def test_citation_metrics():
    # Exact citation match
    metrics = compute_citation_metrics("doc1.pdf;doc2.pdf", ["doc1.pdf", "doc2.pdf"])
    assert metrics["exact_match"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0

    # Partial citation match (predicted 2 docs, reference only 1)
    metrics_partial = compute_citation_metrics("doc1.pdf;doc2.pdf", ["doc1.pdf"])
    assert metrics_partial["precision"] == 0.5
    assert metrics_partial["recall"] == 1.0
    assert metrics_partial["exact_match"] == 0.0
    assert round(metrics_partial["f1"], 4) == 0.6667


# ---------------------------------------------------------------------------
# 7. Latency Calculations (Mean, Median, P95, Deltas, Hit vs Miss)
# ---------------------------------------------------------------------------
def test_latency_calculations():
    lats = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    summary = compute_latency_summary(lats)
    assert summary["count"] == 10
    assert summary["mean_ms"] == 550.0
    assert summary["median_ms"] == 550.0
    assert summary["min_ms"] == 100.0
    assert summary["max_ms"] == 1000.0
    assert summary["p95_ms"] == 1000.0

    # Comparative analysis deltas
    mock_system_summaries = {
        "normal_rag": {"latency": {"mean_ms": 1500.0}, "total_llm_calls": 32},
        "self_rag": {"latency": {"mean_ms": 12000.0}, "total_llm_calls": 160},
        "evorag": {"latency": {"mean_ms": 3000.0}, "total_llm_calls": 40},
    }
    comp = compute_comparative_analysis(mock_system_summaries)
    assert comp["latency_deltas"]["evorag_vs_self_rag_mean_diff_ms"] == -9000.0
    assert comp["latency_deltas"]["evorag_latency_reduction_pct_vs_self_rag"] == 75.0
    assert comp["llm_call_deltas"]["llm_calls_avoided_by_evorag"] == 120


# ---------------------------------------------------------------------------
# 8. Memory-Hit and Efficiency Metrics
# ---------------------------------------------------------------------------
def test_memory_hit_and_efficiency_metrics(sample_predictions_csv: Path):
    records, _, _ = load_predictions(sample_predictions_csv)
    evorag_records = [r for r in records if r["system"] == "evorag"]
    evorag_summary = evaluate_system("evorag", evorag_records)

    assert evorag_summary["total_evaluated"] == 2
    assert evorag_summary["candidate_found_count"] == 1
    assert evorag_summary["candidate_found_rate"] == 0.5
    assert evorag_summary["memory_hit_count"] == 1
    assert evorag_summary["memory_hit_rate"] == 0.5
    assert evorag_summary["memory_reuse_count"] == 1
    assert evorag_summary["memory_reuse_rate"] == 0.5

    # Memory hit vs miss latency separation
    assert evorag_summary["latency_memory_hit"]["count"] == 1
    assert evorag_summary["latency_memory_hit"]["mean_ms"] == 25.0
    assert evorag_summary["latency_memory_miss"]["count"] == 1
    assert evorag_summary["latency_memory_miss"]["mean_ms"] == 13900.0


# ---------------------------------------------------------------------------
# 9. Prediction File Immutability
# ---------------------------------------------------------------------------
def test_prediction_file_immutability(sample_predictions_csv: Path, sample_answer_key_jsonl: Path, tmp_path: Path):
    initial_sha = compute_file_sha256(sample_predictions_csv)
    out_dir = tmp_path / "reports"

    # Run scoring
    result = execute_scoring(
        predictions_path=sample_predictions_csv,
        answer_key_path=sample_answer_key_jsonl,
        out_dir=out_dir,
    )

    final_sha = compute_file_sha256(sample_predictions_csv)
    assert initial_sha == final_sha, "Prediction file hash must remain unchanged after scoring."
    assert result["predictions_sha256"] == initial_sha


# ---------------------------------------------------------------------------
# 10. Scoring Report Output Schema & Artifacts
# ---------------------------------------------------------------------------
def test_scoring_report_schema_and_artifacts(
    sample_predictions_csv: Path,
    sample_answer_key_jsonl: Path,
    tmp_path: Path,
):
    out_dir = tmp_path / "reports_test"
    result = execute_scoring(
        predictions_path=sample_predictions_csv,
        answer_key_path=sample_answer_key_jsonl,
        out_dir=out_dir,
    )

    report_paths = result["report_paths"]
    md_path = Path(report_paths["markdown"])
    json_path = Path(report_paths["json"])
    csv_path = Path(report_paths["csv"])

    assert md_path.is_file()
    assert json_path.is_file()
    assert csv_path.is_file()

    # Verify JSON structure
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "metadata" in json_data
    assert "validation" in json_data
    assert "system_summaries" in json_data
    assert "comparative_analysis" in json_data
    assert "per_question_scores" in json_data
    assert set(json_data["system_summaries"].keys()) == {"normal_rag", "self_rag", "evorag"}

    # Verify Markdown contains critical sections
    md_text = md_path.read_text(encoding="utf-8")
    assert "Executive Summary" in md_text
    assert "Latency & Execution Performance" in md_text
    assert "Experience Memory & Cost Efficiency" in md_text
    assert "Answer Quality & Citation Accuracy" in md_text
    assert "Comparative Analysis & Findings" in md_text

    # Verify CSV summary rows
    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 4  # Header + 3 systems
    assert "normal_rag" in csv_lines[1]
    assert "self_rag" in csv_lines[2]
    assert "evorag" in csv_lines[3]


# ---------------------------------------------------------------------------
# 11. No Inference Calls During Scoring
# ---------------------------------------------------------------------------
def test_no_inference_calls_during_scoring(
    sample_predictions_csv: Path,
    sample_answer_key_jsonl: Path,
    tmp_path: Path,
    monkeypatch,
):
    """Confirm scoring is strictly an offline analysis that triggers zero LLM/embedding inference."""
    # Mock generator and retriever calls to raise errors if accidentally invoked
    mock_llm = MagicMock(side_effect=RuntimeError("LLM should not be called during scoring!"))
    monkeypatch.setattr("normal_rag.generator.GeminiGenerator.generate", mock_llm, raising=False)
    monkeypatch.setattr("selfrag.generator.SelfRAGGenerator.generate", mock_llm, raising=False)
    monkeypatch.setattr("retriever.retriever.PassageRetriever.retrieve", mock_llm, raising=False)

    out_dir = tmp_path / "reports_no_inference"
    result = execute_scoring(
        predictions_path=sample_predictions_csv,
        answer_key_path=sample_answer_key_jsonl,
        out_dir=out_dir,
    )
    assert result["freeze_id"] == "20261110_000000Z"
    mock_llm.assert_not_called()

