"""
Automated Unit Tests for EvoRAG Task 9B: Verified Experience Learning and Benchmark Validation.

Tests:
1. Answer key PDF conversion and parsing.
2. Canonical ID matching and complete coverage.
3. Missing and duplicate ID detection in answer keys.
4. Experience ingestion validation (approval, evidence, source versions).
5. Provenance preservation (chunk hashes, source versions).
6. Training and evaluation contamination safeguards.
7. Memory index rebuild and synchronization.
8. Frozen prediction immutability (SHA-256 integrity).
9. Post-learning evaluation schema and comparative report.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from evorag.config import DEFAULT_SIMILARITY_THRESHOLD
from evorag.memory.schema import (
    Experience,
    VerificationStatus,
    generate_experience_id,
    normalize_question,
)
from evorag.memory.store import ExperienceStore
from evorag.memory.index_builder import ExperienceIndexBuilder
from evaluation.convert_answer_key_pdf import (
    convert_answer_key,
    parse_answer_key_pdf,
    validate_converted_answer_key,
)
from evaluation.ingest_experiences import (
    ingest_experiences,
    load_corpus_source_versions,
    load_held_out_guard,
)
from evaluation.run_post_learning_evaluation import (
    MockSelfRAGGenerator,
    run_post_learning_evaluation,
)
from evaluation.score_predictions import compute_file_sha256, load_answer_key


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_PDF_PATH = PROJECT_ROOT / "EvoRAG_Answer_Key_For_Frozen_Run" / "answer_key_32_for_frozen_run.pdf"
HELD_OUT_PATH = PROJECT_ROOT / "EvoRAG_Blind_Test_Questions" / "held_out_test_questions_only.jsonl"
FROZEN_CSV_PATH = PROJECT_ROOT / "evaluation" / "predictions" / "frozen" / "predictions_20261109_151957Z.csv"


# ---------------------------------------------------------------------------
# 1. Answer Key PDF Conversion & Parsing
# ---------------------------------------------------------------------------
def test_answer_key_pdf_conversion(tmp_path: Path):
    """Test extracting all 32 questions from the official answer key PDF."""
    if not OFFICIAL_PDF_PATH.is_file():
        pytest.skip("Official answer key PDF not found")

    out_jsonl = tmp_path / "converted_key.jsonl"
    result = convert_answer_key(
        pdf_path=OFFICIAL_PDF_PATH,
        out_jsonl_path=out_jsonl,
        reference_questions_path=HELD_OUT_PATH,
    )

    assert out_jsonl.is_file()
    assert result["validation"]["total_parsed"] == 32
    assert result["validation"]["unique_questions"] == 32

    # Check unanswerable questions detected
    unans = result["validation"]["unanswerable_questions"]
    assert "DBMS-EVAL-0149" in unans
    assert "DBMS-EVAL-0150" in unans

    # Verify first entry structure
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 32
    first_item = json.loads(lines[0])
    assert first_item["question_id"] == "DBMS-EVAL-0041"
    assert "buffer manager" in first_item["question"].lower()
    assert len(first_item["supporting_sources"]) > 0
    assert first_item["supporting_sources"][0]["filename"] == "08_database_storage_pages_and_buffer_management.pdf"


# ---------------------------------------------------------------------------
# 2. Canonical ID Matching & Complete Coverage
# ---------------------------------------------------------------------------
def test_canonical_id_matching():
    """Verify that all parsed question IDs match the held-out question archive."""
    if not OFFICIAL_PDF_PATH.is_file() or not HELD_OUT_PATH.is_file():
        pytest.skip("Required benchmark files missing")

    entries = parse_answer_key_pdf(OFFICIAL_PDF_PATH)
    parsed_qids = [e["question_id"] for e in entries]

    held_out_qids = []
    with open(HELD_OUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                held_out_qids.append(json.loads(line)["question_id"])

    assert len(parsed_qids) == 32
    assert parsed_qids == held_out_qids


# ---------------------------------------------------------------------------
# 3. Missing and Duplicate ID Detection in Answer Keys
# ---------------------------------------------------------------------------
def test_missing_and_duplicate_id_detection(tmp_path: Path):
    """Test validation detects duplicate question IDs or missing questions."""
    # Duplicate ID
    dup_entries = [
        {"question_id": "DBMS-EVAL-0041", "answerable_from_corpus": True},
        {"question_id": "DBMS-EVAL-0041", "answerable_from_corpus": True},
    ]
    with pytest.raises(ValueError, match="Duplicate question IDs"):
        validate_converted_answer_key(dup_entries)

    # Missing question ID relative to reference file
    ref_file = tmp_path / "ref.jsonl"
    ref_file.write_text(
        json.dumps({"question_id": "DBMS-EVAL-0041", "question": "q1"}) + "\n" +
        json.dumps({"question_id": "DBMS-EVAL-0042", "question": "q2"}) + "\n"
    )
    partial_entries = [
        {"question_id": "DBMS-EVAL-0041", "answerable_from_corpus": True},
    ]
    with pytest.raises(ValueError, match="Answer key missing 1 held-out questions"):
        validate_converted_answer_key(partial_entries, reference_questions_path=ref_file)


# ---------------------------------------------------------------------------
# 4. Experience Ingestion Validation
# ---------------------------------------------------------------------------
def test_experience_ingestion_validation(tmp_path: Path):
    """Test rejection of unapproved, evidence-lacking, or malformed experiences."""
    db_path = tmp_path / "exp.db"
    index_dir = tmp_path / "idx"

    # 1. Reject unapproved status
    unapp_file = tmp_path / "unapproved.jsonl"
    unapp_file.write_text(json.dumps({
        "question_id": "DBMS-TRAIN-0099",
        "question": "What is 2PL?",
        "answer": "Two-phase locking.",
        "verification_status": "pending",
        "evidence": [{"text": "ev", "source": "doc.pdf"}],
        "source_versions": {"doc.pdf": "ver1"},
    }) + "\n")

    res1 = ingest_experiences(unapp_file, db_path=db_path, index_dir=index_dir, rebuild_index=False)
    assert res1["ingested_count"] == 0
    assert res1["rejected_count"] == 1
    assert "NOT_APPROVED" in res1["rejected_records"][0]["reason"]

    # 2. Reject missing evidence
    no_ev_file = tmp_path / "no_ev.jsonl"
    no_ev_file.write_text(json.dumps({
        "question_id": "DBMS-TRAIN-0099",
        "question": "What is 2PL?",
        "answer": "Two-phase locking.",
        "verification_status": "approved",
        "evidence": [],
        "source_versions": {"doc.pdf": "ver1"},
    }) + "\n")

    res2 = ingest_experiences(no_ev_file, db_path=db_path, index_dir=index_dir, rebuild_index=False)
    assert res2["ingested_count"] == 0
    assert res2["rejected_count"] == 1
    assert "MISSING_EVIDENCE" in res2["rejected_records"][0]["reason"]

    # 3. Reject stale source versions
    stale_file = tmp_path / "stale.jsonl"
    stale_file.write_text(json.dumps({
        "question_id": "DBMS-TRAIN-0099",
        "question": "What is 2PL?",
        "answer": "Two-phase locking.",
        "verification_status": "approved",
        "evidence": [{"text": "ev", "source": "doc.pdf"}],
        "source_versions": {"doc.pdf": "old_hash"},
    }) + "\n")
    corpus_meta = tmp_path / "meta.json"
    corpus_meta.write_text(json.dumps({"source_versions": {"doc.pdf": "new_hash"}}))

    res3 = ingest_experiences(
        stale_file,
        db_path=db_path,
        index_dir=index_dir,
        corpus_meta_path=corpus_meta,
        rebuild_index=False,
    )
    assert res3["ingested_count"] == 0
    assert "STALE_SOURCE_VERSION" in res3["rejected_records"][0]["reason"]


# ---------------------------------------------------------------------------
# 5. Provenance Preservation
# ---------------------------------------------------------------------------
def test_provenance_preservation(tmp_path: Path):
    """Test that chunk hashes, source versions, and page numbers are preserved."""
    db_path = tmp_path / "test_store.db"
    store = ExperienceStore(db_path)

    exp = Experience(
        question="How does WAL work in database recovery?",
        answer="WAL ensures changes are written to the log before data pages are flushed to disk.",
        evidence=[{
            "text": "WAL protocol excerpt...",
            "source": "12_database_recovery_logging_wal_and_backup.pdf",
            "page": 2,
            "chunk_hash": "a1b2c3d4e5f6",
            "source_version": "v123456",
        }],
        source_versions={"12_database_recovery_logging_wal_and_backup.pdf": "v123456"},
        chunk_hashes=["a1b2c3d4e5f6"],
        verification_status=VerificationStatus.APPROVED,
    )

    store.add_experience(exp)
    fetched = store.get_experience(exp.experience_id)
    assert fetched is not None
    assert fetched.chunk_hashes == ["a1b2c3d4e5f6"]
    assert fetched.source_versions == {"12_database_recovery_logging_wal_and_backup.pdf": "v123456"}
    assert fetched.evidence[0]["page"] == 2
    assert fetched.evidence[0]["chunk_hash"] == "a1b2c3d4e5f6"


# ---------------------------------------------------------------------------
# 6. Training/Evaluation Contamination Safeguards
# ---------------------------------------------------------------------------
def test_training_evaluation_contamination_safeguards(tmp_path: Path):
    """Confirm that attempting to ingest held-out questions is rejected."""
    db_path = tmp_path / "contam.db"
    index_dir = tmp_path / "contam_idx"

    held_out_file = tmp_path / "held_out.jsonl"
    held_out_file.write_text(json.dumps({
        "question_id": "DBMS-EVAL-0041",
        "question": "What is the purpose of a buffer manager?",
    }) + "\n")

    # Ingestion candidate with identical held-out ID
    bad_id_file = tmp_path / "bad_id.jsonl"
    bad_id_file.write_text(json.dumps({
        "question_id": "DBMS-EVAL-0041",
        "question": "A different question text",
        "answer": "Test answer",
        "verification_status": "approved",
        "evidence": [{"text": "ev", "source": "08_doc.pdf"}],
        "source_versions": {"08_doc.pdf": "h1"},
    }) + "\n")

    res1 = ingest_experiences(bad_id_file, db_path=db_path, index_dir=index_dir, held_out_path=held_out_file, rebuild_index=False)
    assert res1["ingested_count"] == 0
    assert "CONTAMINATION" in res1["rejected_records"][0]["reason"]

    # Ingestion candidate with identical held-out question text
    bad_text_file = tmp_path / "bad_text.jsonl"
    bad_text_file.write_text(json.dumps({
        "question_id": "DBMS-TRAIN-9999",
        "question": "What is the purpose of a buffer manager?",
        "answer": "Test answer",
        "verification_status": "approved",
        "evidence": [{"text": "ev", "source": "08_doc.pdf"}],
        "source_versions": {"08_doc.pdf": "h1"},
    }) + "\n")

    res2 = ingest_experiences(bad_text_file, db_path=db_path, index_dir=index_dir, held_out_path=held_out_file, rebuild_index=False)
    assert res2["ingested_count"] == 0
    assert "CONTAMINATION" in res2["rejected_records"][0]["reason"]


# ---------------------------------------------------------------------------
# 7. Memory Index Rebuild & Synchronization
# ---------------------------------------------------------------------------
def test_memory_index_rebuild(tmp_path: Path):
    """Test FAISS index builder updates vector count and explicit ID map."""
    index_dir = tmp_path / "faiss_test"
    builder = ExperienceIndexBuilder(index_dir=index_dir)

    exp1 = Experience(
        question="What is a transaction?",
        answer="A logical unit of work.",
        verification_status=VerificationStatus.APPROVED,
    )
    exp2 = Experience(
        question="What is durability?",
        answer="Committed changes survive failure.",
        verification_status=VerificationStatus.APPROVED,
    )

    meta = builder.build_from_experiences([exp1, exp2])
    assert meta["num_vectors"] == 2
    assert meta["id_map"] == [exp1.experience_id, exp2.experience_id]
    assert (index_dir / "experience_faiss.index").is_file()
    assert (index_dir / "experience_meta.json").is_file()


# ---------------------------------------------------------------------------
# 8. Frozen Prediction Immutability
# ---------------------------------------------------------------------------
def test_frozen_prediction_immutability():
    """Verify that frozen prediction CSV remains byte-for-byte identical."""
    if not FROZEN_CSV_PATH.is_file():
        pytest.skip("Frozen prediction file missing")

    expected_sha256 = "a08d3bd3b0440302a02fd2f81c5dbab3d878cce75406657319940e4bb495d3e7"
    actual_sha256 = compute_file_sha256(FROZEN_CSV_PATH)
    assert actual_sha256 == expected_sha256, "Frozen predictions file was mutated!"


# ---------------------------------------------------------------------------
# 9. Post-Learning Evaluation Schema & Mock Execution
# ---------------------------------------------------------------------------
def test_post_learning_evaluation_schema(tmp_path: Path):
    """Test post-learning evaluation harness runs in mock mode and generates valid schema."""
    if not HELD_OUT_PATH.is_file() or not FROZEN_CSV_PATH.is_file():
        pytest.skip("Benchmark files missing")

    # Create a small answer key fixture
    test_key_path = tmp_path / "test_key.jsonl"
    with open(test_key_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "question_id": "DBMS-EVAL-0041",
            "question": "What is the purpose of a buffer manager?",
            "expected_answer": "Caches pages in memory to reduce storage I/O.",
            "reference_answers": ["Caches pages in memory to reduce storage I/O."],
            "supporting_sources": [{"filename": "08_database_storage_pages_and_buffer_management.pdf"}],
        }) + "\n")

    # Create small question set fixture
    test_q_path = tmp_path / "test_q.jsonl"
    with open(test_q_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "question_id": "DBMS-EVAL-0041",
            "question": "What is the purpose of a buffer manager?",
        }) + "\n")

    out_dir = tmp_path / "eval_out"
    res = run_post_learning_evaluation(
        questions_path=test_q_path,
        corpus_index_dir=PROJECT_ROOT / "index" / "dbms_sql_v1",
        db_path=PROJECT_ROOT / "data" / "experience_memory.db",
        memory_index_dir=PROJECT_ROOT / "data",
        frozen_csv_path=FROZEN_CSV_PATH,
        answer_key_path=test_key_path,
        out_dir=out_dir,
        mock_model=True,
    )

    assert Path(res["report_path"]).is_file()
    assert Path(res["predictions_csv"]).is_file()
    assert Path(res["predictions_jsonl"]).is_file()

    # Check report contents
    md = Path(res["report_path"]).read_text(encoding="utf-8")
    assert "Post-Learning Benchmark Comparison Report" in md
    assert "Experience Memory Routing & Hit Rates" in md
    assert "Answer Quality & Citation Accuracy" in md
    assert "Latency & LLM Inference Cost" in md

