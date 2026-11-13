"""Test suite for EvoRAG Unified CLI (demo.py).

Covers:
1. CLI argument parsing and defaults.
2. Invalid mode handling.
3. Missing model, adapter, and index error handling.
4. Environment mismatch and --no-reexec handling.
5. Execution across all primary modes with mock components.
6. Graceful inference exception handling.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import demo
from evorag.result import EvoRAGResult
from selfrag.pipeline import Candidate, Segment, SelfRAGResult


# ---------------------------------------------------------------------------
# 1. Argument Parsing & Defaults
# ---------------------------------------------------------------------------
def test_cli_argument_parser_defaults():
    """Verify that default CLI arguments match production expectations."""
    parser = demo.build_arg_parser()
    args = parser.parse_args([])

    assert args.query is None
    assert args.mode is None
    assert args.threshold == 0.85
    assert args.k == 5
    assert args.max_segments == 3
    assert args.max_tokens == 250
    assert not args.verbose
    assert not args.no_reexec
    assert "selfrag_llama2_7b.Q4_K_M.gguf" in args.model
    assert "models/mlx/selfrag_llama2_7b_4bit" in args.mlx_model
    assert "training/qlora_dbms/adapters/full_dbms_v1" in args.mlx_adapter


def test_cli_argument_parser_overrides():
    """Verify custom CLI flag overrides."""
    parser = demo.build_arg_parser()
    cmd = [
        "What is WAL?",
        "--mode", "evorag-mlx",
        "--threshold", "0.90",
        "--k", "7",
        "--max-segments", "4",
        "--max-tokens", "300",
        "--verbose",
        "--no-reexec",
    ]
    args = parser.parse_args(cmd)

    assert args.query == "What is WAL?"
    assert args.mode == "evorag-mlx"
    assert args.threshold == 0.90
    assert args.k == 7
    assert args.max_segments == 4
    assert args.max_tokens == 300
    assert args.verbose is True
    assert args.no_reexec is True


def test_cli_invalid_mode_handling():
    """Verify invalid mode triggers clean argparse rejection."""
    parser = demo.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "unsupported_mode_xyz"])


# ---------------------------------------------------------------------------
# 2. Missing Resource Error Handling
# ---------------------------------------------------------------------------
def test_cli_missing_index_handling(capsys):
    """Verify missing document index displays error and returns 1."""
    args = argparse.Namespace(
        index="/non/existent/index_path_xyz",
        k=5,
        verbose=False,
    )
    ret = demo.run_normal_mode("Test query", args)
    assert ret == 1
    captured = capsys.readouterr()
    assert "[INDEX ERROR]" in captured.out


def test_cli_missing_gguf_model_handling(capsys):
    """Verify missing GGUF model displays error and returns 1."""
    with patch.dict("os.environ", {}, clear=True):
        args = argparse.Namespace(
            index=str(PROJECT_ROOT / "index" / "dbms_sql_v1"),
            model="/non/existent/model.gguf",
            k=5,
            max_segments=3,
            verbose=False,
        )
        ret = demo.run_selfrag_mode("Test query", args)
        assert ret == 1
        captured = capsys.readouterr()
        assert "[MODEL ERROR]" in captured.out


def test_cli_missing_mlx_model_handling(capsys):
    """Verify missing MLX base model displays error and returns 1."""
    args = argparse.Namespace(
        index=str(PROJECT_ROOT / "index" / "dbms_sql_v1"),
        mlx_model="/non/existent/mlx_model_dir",
        mlx_adapter=str(PROJECT_ROOT / "training" / "qlora_dbms" / "adapters" / "full_dbms_v1"),
        memory_db="data/experience_memory.db",
        memory_index="data",
        threshold=0.85,
        max_tokens=250,
        k=5,
        max_segments=3,
        verbose=False,
    )
    ret = demo.run_evorag_mlx_mode("Test query", args)
    assert ret == 1
    captured = capsys.readouterr()
    assert "[MODEL ERROR]" in captured.out


def test_cli_missing_mlx_adapter_handling(capsys):
    """Verify missing MLX adapter displays error and returns 1."""
    args = argparse.Namespace(
        index=str(PROJECT_ROOT / "index" / "dbms_sql_v1"),
        mlx_model=str(PROJECT_ROOT / "models" / "mlx" / "selfrag_llama2_7b_4bit"),
        mlx_adapter="/non/existent/adapter_path_xyz",
        memory_db="data/experience_memory.db",
        memory_index="data",
        threshold=0.85,
        max_tokens=250,
        k=5,
        max_segments=3,
        verbose=False,
    )
    ret = demo.run_evorag_mlx_mode("Test query", args)
    assert ret == 1
    captured = capsys.readouterr()
    assert "[ADAPTER ERROR]" in captured.out


# ---------------------------------------------------------------------------
# 3. Environment Dispatch & Mismatch Handling
# ---------------------------------------------------------------------------
def test_environment_dispatch_evorag_mlx_no_reexec(capsys):
    """Verify that selecting evorag-mlx without MLX and --no-reexec shows environment error."""
    args = argparse.Namespace(no_reexec=True)
    with patch.dict("sys.modules", {"mlx": None, "mlx.core": None, "mlx_lm": None}):
        ret = demo.check_and_dispatch_environment("evorag-mlx", args)
        assert ret == 1
        captured = capsys.readouterr()
        assert "[ENVIRONMENT ERROR]" in captured.out
        assert ".venv-mlx/bin/python demo.py" in captured.out


def test_environment_dispatch_gguf_no_reexec(capsys):
    """Verify that selecting GGUF mode without llama_cpp and --no-reexec shows environment error."""
    args = argparse.Namespace(no_reexec=True)
    with patch.dict("sys.modules", {"llama_cpp": None}):
        with patch.dict("os.environ", {"SELFRAG_BACKEND": "llamacpp"}):
            ret = demo.check_and_dispatch_environment("selfrag", args)
            assert ret == 1
            captured = capsys.readouterr()
            assert "[ENVIRONMENT ERROR]" in captured.out
            assert ".venv/bin/python demo.py" in captured.out


# ---------------------------------------------------------------------------
# 4. Mock Execution for All Four Modes
# ---------------------------------------------------------------------------
def test_mock_normal_mode(capsys):
    """Test Normal RAG execution with mocked generator and retriever."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {"source": "01_dbms.pdf", "page": 1, "chunk_id": 10, "text": "DBMS evidence snippet"}
    ]
    mock_generator = MagicMock()
    mock_generator._model = "gemini-3.5-flash-lite"
    mock_generator.generate.return_value = "Normal RAG synthesized answer."

    with patch("retriever.retriever.PassageRetriever", return_value=mock_retriever):
        with patch("normal_rag.generator.GeminiGenerator", return_value=mock_generator):
            args = argparse.Namespace(
                index=str(PROJECT_ROOT / "index" / "dbms_sql_v1"),
                k=3,
                verbose=False,
            )
            ret = demo.run_normal_mode("What is a DBMS?", args)
            assert ret == 0
            captured = capsys.readouterr()
            assert "NORMAL RAG EXECUTION RESULT" in captured.out
            assert "Normal RAG synthesized answer." in captured.out
            assert "01_dbms.pdf" in captured.out


def test_mock_selfrag_mode(capsys):
    """Test Self-RAG execution with mocked pipeline."""
    mock_cand = MagicMock()
    mock_cand.passage = {"source": "08_storage.pdf", "page": 2, "chunk_id": 45}
    mock_cand.isrel = "[Relevant]"
    mock_cand.issup = "[Fully supported]"
    mock_cand.isuse = 5
    mock_cand.score = 5.0

    mock_seg = MagicMock()
    mock_seg.index = 1
    mock_seg.decision = "[Continue to Use Evidence]"
    mock_seg.reused_evidence = False
    mock_seg.candidates = [mock_cand]
    mock_seg.selected = mock_cand

    mock_sr_res = SelfRAGResult(
        query="What is a buffer manager?",
        retrieved=True,
        candidates=[mock_cand],
        selected=mock_cand,
        final_answer="The buffer pool manager manages memory frames.",
        decision="[Retrieval]",
        segments=[mock_seg],
    )

    with patch("retriever.retriever.PassageRetriever"):
        with patch("selfrag.generator.SelfRAGGenerator"):
            with patch("selfrag.pipeline.self_rag_answer", return_value=mock_sr_res):
                args = argparse.Namespace(
                    index=str(PROJECT_ROOT / "index" / "dbms_sql_v1"),
                    model="models/selfrag_llama2_7b.Q4_K_M.gguf",
                    k=3,
                    max_segments=2,
                    verbose=True,
                )
                ret = demo.run_selfrag_mode("What is a buffer manager?", args)
                assert ret == 0
                captured = capsys.readouterr()
                assert "SELF-RAG EXECUTION RESULT" in captured.out
                assert "The buffer pool manager manages memory frames." in captured.out
                assert "08_storage.pdf" in captured.out


def test_mock_evorag_mode_path_a(capsys):
    """Test EvoRAG execution for Path A (Direct Reuse)."""
    mock_evorag_res = EvoRAGResult(
        question="How does physical data independence protect applications?",
        answer="Physical data independence insulates queries from storage changes.",
        memory_hit=True,
        response_mode="direct_reuse",
        memory_context_used=False,
        experience_id="exp-direct-001",
        similarity_score=1.0,
        evidence=[{"source": "01_dbms.pdf", "page": 1, "chunk_id": 2}],
    )

    mock_controller = MagicMock()
    mock_controller.answer.return_value = mock_evorag_res

    with patch("evorag.memory.store.ExperienceStore"):
        with patch("evorag.memory.retriever.ExperienceRetriever"):
            with patch("evorag.memory.validator.ExperienceValidator"):
                with patch("evorag.memory.index_builder.ExperienceIndexBuilder"):
                    with patch("retriever.retriever.PassageRetriever"):
                        with patch("selfrag.generator.SelfRAGGenerator"):
                            with patch("evorag.controller.EvoRAGController", return_value=mock_controller):
                                args = argparse.Namespace(
                                    index=str(PROJECT_ROOT / "index" / "dbms_sql_v1"),
                                    model="models/selfrag_llama2_7b.Q4_K_M.gguf",
                                    memory_db="data/experience_memory.db",
                                    memory_index="data",
                                    threshold=0.85,
                                    k=5,
                                    max_segments=3,
                                    verbose=False,
                                )
                                ret = demo.run_evorag_mode("How does physical data independence protect applications?", args)
                                assert ret == 0
                                captured = capsys.readouterr()
                                assert "EVORAG EXECUTION RESULT (GGUF)" in captured.out
                                assert "Path A (Direct Reuse)" in captured.out
                                assert "exp-direct-001" in captured.out
                                assert "Physical data independence insulates queries" in captured.out


def test_mock_evorag_mlx_mode_path_b(capsys):
    """Test EvoRAG MLX execution for Path B (Experience-Assisted Self-RAG)."""
    mock_evorag_res = EvoRAGResult(
        question="Explain buffer pool frame replacement.",
        answer="Buffer pool uses CLOCK or LRU replacement policy.",
        memory_hit=False,
        response_mode="experience_assisted",
        memory_context_used=True,
        similarity_score=0.88,
        memory_experience_ids=["exp-assist-002"],
        document_evidence=[{"source": "08_storage.pdf", "page": 3, "chunk_id": 49}],
        experience_evidence=[{"source": "exp_memory", "page": None, "chunk_id": 99}],
        evidence=[
            {"source": "08_storage.pdf", "page": 3, "chunk_id": 49},
            {"source": "exp_memory", "page": None, "chunk_id": 99, "source_type": "experience_memory"},
        ],
    )

    mock_controller = MagicMock()
    mock_controller.answer.return_value = mock_evorag_res

    mock_mlx_gen = MagicMock()
    mock_mlx_gen.memory_after_load_gb = 3.79

    with patch("evorag.memory.store.ExperienceStore"):
        with patch("evorag.memory.retriever.ExperienceRetriever"):
            with patch("evorag.memory.validator.ExperienceValidator"):
                with patch("evorag.memory.index_builder.ExperienceIndexBuilder"):
                    with patch("retriever.retriever.PassageRetriever"):
                        with patch("evorag.mlx_generator.MLXSelfRAGGenerator", return_value=mock_mlx_gen):
                            with patch("evorag.controller.EvoRAGController", return_value=mock_controller):
                                args = argparse.Namespace(
                                    index=str(PROJECT_ROOT / "index" / "dbms_sql_v1"),
                                    mlx_model=str(PROJECT_ROOT / "models" / "mlx" / "selfrag_llama2_7b_4bit"),
                                    mlx_adapter=str(PROJECT_ROOT / "training" / "qlora_dbms" / "adapters" / "full_dbms_v1"),
                                    memory_db="data/experience_memory.db",
                                    memory_index="data",
                                    threshold=0.85,
                                    max_tokens=250,
                                    k=5,
                                    max_segments=3,
                                    verbose=False,
                                )
                                ret = demo.run_evorag_mlx_mode("Explain buffer pool frame replacement.", args)
                                assert ret == 0
                                captured = capsys.readouterr()
                                assert "FINE-TUNED EVORAG EXECUTION RESULT (MLX)" in captured.out
                                assert "Path B (Experience-Assisted Self-RAG)" in captured.out
                                assert "exp-assist-002" in captured.out
                                assert "CLOCK or LRU" in captured.out


# ---------------------------------------------------------------------------
# 5. Graceful Inference Error Handling
# ---------------------------------------------------------------------------
def test_cli_graceful_inference_error_handling(capsys):
    """Verify that unhandled controller exceptions produce a clean error message and return 1."""
    mock_controller = MagicMock()
    mock_controller.answer.side_effect = RuntimeError("Fatal computation failure")

    with patch("evorag.memory.store.ExperienceStore"):
        with patch("evorag.memory.retriever.ExperienceRetriever"):
            with patch("evorag.memory.validator.ExperienceValidator"):
                with patch("evorag.memory.index_builder.ExperienceIndexBuilder"):
                    with patch("retriever.retriever.PassageRetriever"):
                        with patch("selfrag.generator.SelfRAGGenerator"):
                            with patch("evorag.controller.EvoRAGController", return_value=mock_controller):
                                args = argparse.Namespace(
                                    index=str(PROJECT_ROOT / "index" / "dbms_sql_v1"),
                                    model="models/selfrag_llama2_7b.Q4_K_M.gguf",
                                    memory_db="data/experience_memory.db",
                                    memory_index="data",
                                    threshold=0.85,
                                    k=5,
                                    max_segments=3,
                                    verbose=False,
                                )
                                ret = demo.run_evorag_mode("Trigger failure", args)
                                assert ret == 1
                                captured = capsys.readouterr()
                                assert "[INFERENCE ERROR] Fatal computation failure" in captured.out
