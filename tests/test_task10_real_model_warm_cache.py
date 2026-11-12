"""Tests for EvoRAG Task 10: Real-Model Warm-Cache Evaluation & Benchmark."""
import csv
import json
from pathlib import Path
import pytest

from evaluation.score_predictions import compute_file_sha256

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FROZEN_BASELINE_CSV = PROJECT_ROOT / "evaluation" / "predictions" / "frozen" / "predictions_20261109_151957Z.csv"
FROZEN_BASELINE_SHA256 = "a08d3bd3b0440302a02fd2f81c5dbab3d878cce75406657319940e4bb495d3e7"
REPORT_FILE = PROJECT_ROOT / "evaluation" / "reports" / "task10_real_model_warm_cache_report.md"


def test_frozen_baseline_immutability():
    """Verify that the frozen baseline predictions CSV has not been modified."""
    assert FROZEN_BASELINE_CSV.exists()
    sha = compute_file_sha256(FROZEN_BASELINE_CSV)
    assert sha == FROZEN_BASELINE_SHA256


def test_task10_report_exists_and_valid():
    """Verify that the Task 10 report exists and contains required sections."""
    assert REPORT_FILE.exists()
    content = REPORT_FILE.read_text(encoding="utf-8")
    assert "EvoRAG Task 10: Real-Model Warm-Cache Evaluation" in content
    assert "Comparative Benchmark Matrix" in content
    assert "Execution Path Distribution & Memory Routing" in content
    assert "Deep-Dive Validation of Path B" in content
    assert "Unanswerability & Abstention Performance" in content
    assert "Full Per-Question Results Table" in content
    assert "Model and Environment Specifications" in content


def test_task10_predictions_directory_and_schema():
    """Verify that real-model warm-cache predictions directory contains all 32 questions with correct schema."""
    pred_dirs = list((PROJECT_ROOT / "evaluation" / "predictions").glob("real_model_warm_cache_*"))
    assert len(pred_dirs) >= 1
    latest_dir = sorted(pred_dirs)[-1]

    jsonl_files = list(latest_dir.glob("*.jsonl"))
    csv_files = [f for f in latest_dir.glob("*.csv") if "predictions" in f.name]
    meta_files = list(latest_dir.glob("*.json"))

    assert len(jsonl_files) == 1
    assert len(csv_files) == 1
    assert len(meta_files) == 1

    # Check JSONL
    detailed = []
    with open(jsonl_files[0], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                detailed.append(json.loads(line))

    assert len(detailed) == 32
    required_jsonl_keys = {
        "question_id", "question", "system", "predicted_answer", "execution_path",
        "response_mode", "real_model_invoked", "latency_ms", "cited_source_ids",
        "retrieved_document_passages", "retrieved_experience_ids", "experience_similarity_score",
    }
    for rec in detailed:
        assert required_jsonl_keys.issubset(rec.keys())
        assert rec["real_model_invoked"] is True
        assert rec["latency_ms"] > 0
        assert len(rec["predicted_answer"].strip()) > 0

    # Check CSV
    with open(csv_files[0], "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    assert len(csv_rows) == 32


def test_task10_path_b_activations_and_provenance():
    """Verify that the 5 expected questions routed to Path B with valid provenance."""
    pred_dirs = list((PROJECT_ROOT / "evaluation" / "predictions").glob("real_model_warm_cache_*"))
    latest_dir = sorted(pred_dirs)[-1]
    jsonl_file = list(latest_dir.glob("*.jsonl"))[0]

    with open(jsonl_file, "r", encoding="utf-8") as f:
        detailed = [json.loads(line) for line in f if line.strip()]

    path_b_qids = {"DBMS-EVAL-0041", "DBMS-EVAL-0042", "DBMS-EVAL-0044", "DBMS-EVAL-0047", "DBMS-EVAL-0076"}
    actual_path_b = [d for d in detailed if d["execution_path"] == "Path B"]
    actual_qids = {d["question_id"] for d in actual_path_b}

    assert actual_qids == path_b_qids
    for d in actual_path_b:
        assert d["experience_similarity_score"] >= 0.85
        assert len(d["retrieved_experience_ids"]) > 0
        assert len(d["retrieved_document_passages"]) > 0
        assert len(d["cited_source_ids"]) > 0

