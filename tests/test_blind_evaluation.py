"""
Automated tests for Blind Evaluation Runner and Prediction Freeze (Task 9A.1).

Validates:
- All 32 held-out questions are loaded correctly.
- Rejection of any file containing answer keys / target labels (blindness safeguard).
- Each question is evaluated across Normal RAG, Self-RAG, and EvoRAG (using mocks).
- Error resilience: individual system/question failures are caught and recorded without dropping rows.
- Frozen predictions follow both the detailed JSONL schema and the standardized CSV template schema.
- Predictions contain NO answer-key fields.
- Safeguards: the runner does not modify the source question archive, model weights, or Experience Memory database.
- Configuration metadata is fully populated and traceable.
"""
import csv
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from evaluation.run_blind_evaluation import (
    load_held_out_questions,
    execute_blind_evaluation,
    CSV_COLUMNS,
)
from selfrag.pipeline import SelfRAGResult, Candidate, Segment
from evorag.result import EvoRAGResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "EvoRAG_Blind_Test_Questions" / "held_out_test_questions_only.jsonl"
TEMPLATE_CSV_PATH = PROJECT_ROOT / "EvoRAG_Blind_Test_Questions" / "held_out_predictions_template.csv"


class TestBlindQuestionLoadingAndSafeguards:
    """Tests question loading, schema compliance, and blind evaluation safeguards."""

    def test_load_all_32_held_out_questions(self):
        """Verify that load_held_out_questions accurately loads all 32 questions without modification."""
        assert QUESTIONS_PATH.exists(), f"Missing questions file: {QUESTIONS_PATH}"
        questions = load_held_out_questions(QUESTIONS_PATH)
        assert len(questions) == 32

        # Verify first and last IDs match archive
        assert questions[0]["question_id"] == "DBMS-EVAL-0041"
        assert "buffer manager" in questions[0]["question"]
        assert questions[-1]["question_id"] == "DBMS-EVAL-0160"

    def test_blindness_safeguard_rejects_answer_keys(self, tmp_path: Path):
        """Verify that loading an archive containing answer key / ground truth fields raises an error."""
        contaminated_file = tmp_path / "contaminated_questions.jsonl"
        with open(contaminated_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": "TEST-01", "question": "What is ACID?", "expected_answer": "Atomicity..."}) + "\n")

        with pytest.raises(ValueError) as excinfo:
            load_held_out_questions(contaminated_file)

        assert "Blind evaluation violation" in str(excinfo.value)
        assert "expected_answer" in str(excinfo.value)

    def test_question_archive_is_not_modified_during_run(self, tmp_path: Path):
        """Verify that running the evaluation leaves the question archive bit-identical."""
        mtime_before = QUESTIONS_PATH.stat().st_mtime
        content_before = QUESTIONS_PATH.read_bytes()

        # Run mock evaluation
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [{"text": "Sample text", "source": "01_dbms.pdf", "page": 1, "chunk_id": 0}]

        mock_normal_gen = MagicMock()
        mock_normal_gen.generate.return_value = "Mock Normal Answer"

        mock_self_gen = MagicMock()
        mock_self_res = SelfRAGResult(
            query="test",
            retrieved=True,
            candidates=[],
            selected=None,
            final_answer="Mock Self Answer",
            decision="[Retrieval]",
        )

        mock_controller = MagicMock()
        mock_evo_res = EvoRAGResult(
            question="test",
            answer="Mock Evo Answer",
            memory_hit=False,
            response_mode="normal_self_rag",
            evidence=[{"text": "Sample text", "source": "01_dbms.pdf", "page": 1, "chunk_id": 0}],
        )
        mock_controller.answer.return_value = mock_evo_res

        from unittest.mock import patch
        with patch("evaluation.run_blind_evaluation.self_rag_answer", return_value=mock_self_res):
            execute_blind_evaluation(
                questions_file=QUESTIONS_PATH,
                out_dir=tmp_path / "frozen",
                systems=["normal_rag"],
                normal_generator=mock_normal_gen,
                retriever=mock_retriever,
            )

        assert QUESTIONS_PATH.read_bytes() == content_before
        assert QUESTIONS_PATH.stat().st_mtime == mtime_before


class TestBlindEvaluationExecutionAndOutputs:
    """Tests evaluation across all systems using mock components (no real weights needed)."""

    @pytest.fixture
    def mock_eval_components(self):
        """Fixture providing mock implementations for retriever, normal generator, Self-RAG, and EvoRAG."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"text": "DBMS storage page text", "source": "08_database_storage.pdf", "page": 2, "chunk_id": 42}
        ]

        mock_normal_gen = MagicMock()
        mock_normal_gen.generate.return_value = "Normal RAG generated answer based on page 2."

        mock_self_cand = Candidate(
            passage={"text": "DBMS storage page text", "source": "08_database_storage.pdf", "page": 2, "chunk_id": 42},
            raw_text="[Relevant]Self-RAG answer continuation.[Fully supported][Utility:5]",
            isrel="[Relevant]",
            issup="[Fully supported]",
            isuse=5,
            score=3.0,
        )
        mock_self_seg = Segment(
            index=1,
            decision="[Retrieval]",
            evidence=[{"text": "DBMS storage page text", "source": "08_database_storage.pdf", "page": 2, "chunk_id": 42}],
            candidates=[mock_self_cand],
            selected=mock_self_cand,
            answer_text="Self-RAG answer continuation.",
        )
        mock_self_res = SelfRAGResult(
            query="query",
            retrieved=True,
            candidates=[mock_self_cand],
            selected=mock_self_cand,
            final_answer="Self-RAG generated answer continuation.",
            decision="[Retrieval]",
            segments=[mock_self_seg],
        )

        mock_self_gen = MagicMock()

        mock_controller = MagicMock()
        mock_evo_res = EvoRAGResult(
            question="query",
            answer="EvoRAG final answer.",
            memory_hit=False,
            response_mode="experience_assisted",
            memory_context_used=True,
            similarity_score=0.91,
            experience_id="exp_test_123",
            evidence=[{"text": "DBMS storage page text", "source": "08_database_storage.pdf", "page": 2, "chunk_id": 42}],
            self_rag_result=mock_self_res,
        )
        mock_controller.answer.return_value = mock_evo_res

        return {
            "retriever": mock_retriever,
            "normal_gen": mock_normal_gen,
            "self_gen": mock_self_gen,
            "self_res": mock_self_res,
            "controller": mock_controller,
        }

    def test_full_32_questions_evaluated_across_all_three_systems(self, tmp_path: Path, mock_eval_components):
        """Verify that all 32 questions are executed across Normal RAG, Self-RAG, and EvoRAG (96 predictions total)."""
        c = mock_eval_components
        out_dir = tmp_path / "predictions_freeze"

        from unittest.mock import patch
        with patch("evaluation.run_blind_evaluation.self_rag_answer", return_value=c["self_res"]):
            res = execute_blind_evaluation(
                questions_file=QUESTIONS_PATH,
                out_dir=out_dir,
                systems=["normal_rag", "self_rag", "evorag"],
                normal_generator=c["normal_gen"],
                self_generator=c["self_gen"],
                controller=c["controller"],
                retriever=c["retriever"],
            )

        assert res["config_metadata"]["num_questions"] == 32
        assert len(res["detailed_predictions"]) == 96  # 32 * 3
        assert len(res["template_rows"]) == 96

        # Check files exist
        assert res["jsonl_path"].exists()
        assert res["csv_path"].exists()
        assert res["meta_path"].exists()

        # Verify CSV column headers match the template CSV exactly
        with open(res["csv_path"], "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == CSV_COLUMNS

        # Verify controller never called approve_and_save (no memory contamination)
        assert c["controller"].approve_and_save.call_count == 0

    def test_error_resilience_records_failures_without_dropping_rows(self, tmp_path: Path, mock_eval_components):
        """Verify that when a system raises an exception on a question, the row is recorded with error details."""
        c = mock_eval_components
        out_dir = tmp_path / "predictions_errors"

        # Make normal generator fail on all calls
        c["normal_gen"].generate.side_effect = RuntimeError("Simulated API 503 rate-limit error")

        from unittest.mock import patch
        with patch("evaluation.run_blind_evaluation.self_rag_answer", return_value=c["self_res"]):
            res = execute_blind_evaluation(
                questions_file=QUESTIONS_PATH,
                out_dir=out_dir,
                systems=["normal_rag"],
                normal_generator=c["normal_gen"],
                retriever=c["retriever"],
            )

        assert len(res["detailed_predictions"]) == 32
        assert len(res["template_rows"]) == 32

        for row in res["detailed_predictions"]:
            assert row["error"] == "Simulated API 503 rate-limit error"
            assert row["predicted_answer"] == ""

    def test_prediction_output_schema_and_metadata(self, tmp_path: Path, mock_eval_components):
        """Verify JSONL and CSV predictions adhere strictly to required fields without answers."""
        c = mock_eval_components
        out_dir = tmp_path / "predictions_schema"

        from unittest.mock import patch
        with patch("evaluation.run_blind_evaluation.self_rag_answer", return_value=c["self_res"]):
            res = execute_blind_evaluation(
                questions_file=QUESTIONS_PATH,
                out_dir=out_dir,
                systems=["normal_rag", "self_rag", "evorag"],
                normal_generator=c["normal_gen"],
                self_generator=c["self_gen"],
                controller=c["controller"],
                retriever=c["retriever"],
            )

        # Inspect JSONL
        with open(res["jsonl_path"], "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 96
        for item in lines:
            assert "question_id" in item
            assert "question" in item
            assert "system" in item
            assert "predicted_answer" in item
            assert "latency_ms" in item
            assert "cited_source_ids" in item
            assert "retrieved_passages" in item
            assert "configuration" in item
            # Assert NO ground-truth fields exist in prediction output
            assert "expected_answer" not in item
            assert "ground_truth" not in item
            assert "score" not in item

        # Inspect Freeze Metadata
        with open(res["meta_path"], "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["configuration"]["blind_evaluation_mode"] is True
        assert meta["configuration"]["answer_key_loaded"] is False
        assert meta["configuration"]["experience_saving_enabled"] is False
        assert meta["prediction_counts"]["total_rows"] == 96
        assert meta["prediction_counts"]["per_system"]["normal_rag"] == 32
        assert meta["prediction_counts"]["per_system"]["self_rag"] == 32
        assert meta["prediction_counts"]["per_system"]["evorag"] == 32

