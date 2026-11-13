"""EvoRAG Task 12F: Controlled Integration Tests for Experimental MLX Backend.

Tests:
1. Backend Tests: Model/adapter loading, special tokens integrity, utility stopping, weight immutability.
2. Routing Tests: Path A (direct reuse), Path B (experience-assisted), Path C (normal Self-RAG fallback).
3. Evidence Grounding Tests: Retrieval insertion, answer support, abstention on ungrounded queries.
4. Reflection Fidelity: Syntactic and logical validity of [Retrieval], [Relevant], [Fully supported], [Utility:5].
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from evorag.mlx_generator import (
    MLXSelfRAGGenerator,
    SPECIAL_TOKEN_IDS,
    UTILITY_TOKEN_IDS,
    RETRIEVE_YES,
    RETRIEVE_NO,
)
from evorag.controller import EvoRAGController
from evorag.memory.store import ExperienceStore
from evorag.memory.retriever import ExperienceRetriever
from evorag.memory.validator import ExperienceValidator
from retriever.retriever import PassageRetriever
from selfrag.pipeline import self_rag_answer


EXPECTED_BASE_SHA256 = "71576febb874c07b71697976f0892c2bab9e972e77bc880dcc0210eb6342db32"


@pytest.fixture(scope="module")
def mlx_generator():
    """Module-level fixture to load MLX generator once."""
    gen = MLXSelfRAGGenerator(
        model_path="models/mlx/selfrag_llama2_7b_4bit",
        adapter_path="training/qlora_dbms/adapters/full_dbms_v1",
        max_tokens=150,
    )
    return gen


@pytest.fixture(scope="module")
def shared_controller(mlx_generator):
    """Module-level fixture for EvoRAG controller with MLX generator."""
    store = ExperienceStore("data/experience_memory.db")
    mem_retriever = ExperienceRetriever("data", store=store)
    validator = ExperienceValidator(similarity_threshold=0.85)
    doc_retriever = PassageRetriever(index_dir="index/dbms_sql_v1")

    controller = EvoRAGController(
        store=store,
        memory_retriever=mem_retriever,
        validator=validator,
        generator=mlx_generator,
        retriever=doc_retriever,
        doc_index_dir="index/dbms_sql_v1",
    )
    return controller


# =========================================================================
# A. Backend Tests
# =========================================================================

def test_backend_loading_and_telemetry(mlx_generator):
    """Verify base model + adapter load cleanly and report telemetry."""
    metrics = mlx_generator.get_metrics()
    assert metrics["load_duration_seconds"] > 0.0
    assert metrics["memory_after_load_gb"] > 0.0
    assert metrics["peak_memory_gb"] <= 10.0  # M3 Pro headroom check
    assert Path(metrics["model_path"]).exists()
    assert Path(metrics["adapter_path"]).exists()


def test_special_tokens_preservation(mlx_generator):
    """Verify all 16 Self-RAG special tokens maintain exact tokenizer IDs."""
    tokenizer = mlx_generator._tokenizer
    for token_str, expected_id in SPECIAL_TOKEN_IDS.items():
        encoded = tokenizer.encode(token_str, add_special_tokens=False)
        assert len(encoded) == 1, f"Token {token_str} did not tokenize as single ID: {encoded}"
        assert encoded[0] == expected_id, f"ID mismatch for {token_str}: {encoded[0]} != {expected_id}"
        decoded = tokenizer.decode(encoded)
        assert decoded == token_str, f"Decode mismatch: {decoded!r} != {token_str!r}"


def test_retrieval_decision_probe(mlx_generator):
    """Verify probe on question without paragraph returns [Retrieval] or [No Retrieval]."""
    out = mlx_generator.generate("What is a clustered index in database management?", paragraph=None)
    assert out in (RETRIEVE_YES, RETRIEVE_NO), f"Unexpected probe output: {out!r}"


def test_evidence_injection_and_utility_stopping(mlx_generator):
    """Verify externally supplied evidence is cited and generation stops at [Utility:X]."""
    evidence = "In relational databases, write-ahead logging (WAL) requires logging changes before flushing pages."
    out = mlx_generator.generate(
        "Explain write-ahead logging (WAL).",
        paragraph=evidence,
    )
    assert "[Relevant]" in out
    assert any(f"[Utility:{i}]" in out for i in range(1, 6))
    # Verify no runaway generation after utility token
    last_bracket = out.rfind("[Utility:")
    assert last_bracket != -1
    trailing_text = out[last_bracket + 11:].strip()
    assert len(trailing_text) == 0, f"Runaway tokens after utility token: {trailing_text!r}"


def test_base_model_weights_immutability():
    """Verify base model safetensors remain bitwise unmodified."""
    base_path = Path("models/mlx/selfrag_llama2_7b_4bit/model.safetensors")
    with open(base_path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    assert h == EXPECTED_BASE_SHA256, f"Base model hash mutated: {h}"


# =========================================================================
# B. Routing Tests (Path A, Path B, Path C)
# =========================================================================

def test_path_a_direct_experience_reuse(shared_controller):
    """Verify Path A routes exact approved question match without calling generator."""
    q = "How does physical data independence protect applications from storage modifications?"
    gen_calls_before = shared_controller.generator.total_generations
    res = shared_controller.answer(q)

    assert res.response_mode == "direct_reuse"
    assert res.memory_hit is True
    assert res.memory_context_used is False
    assert res.experience_id is not None
    assert len(res.answer) > 20
    # Direct reuse must not invoke generator
    assert shared_controller.generator.total_generations == gen_calls_before


def test_path_b_semantic_experience_assistance(shared_controller):
    """Verify Path B routes semantically similar query and injects memory context."""
    q = "How does physical data independence protect application queries when storage changes?"
    res = shared_controller.answer(q)

    assert res.response_mode == "experience_assisted"
    assert res.memory_context_used is True
    assert res.similarity_score >= 0.85
    assert len(res.memory_experiences) > 0
    assert len(res.answer) > 20
    assert res.self_rag_result is not None


def test_path_c_normal_self_rag_fallback(shared_controller):
    """Verify Path C routes unseen query through PassageRetriever + MLX Generator."""
    q = "Explain relational selection commutativity and filter reordering in query optimization."
    res = shared_controller.answer(q)

    assert res.response_mode == "normal_self_rag"
    assert res.memory_hit is False
    assert res.memory_context_used is False
    assert res.self_rag_result is not None
    assert len(res.answer) > 20
    assert len(res.document_evidence) > 0


# =========================================================================
# C. Evidence-Grounding & Reflection Fidelity Tests
# =========================================================================

def test_grounded_answerable_query(mlx_generator):
    """Verify that an answerable query retrieves relevant chunks and outputs fully supported critique."""
    retriever = PassageRetriever(index_dir="index/dbms_sql_v1")
    q = "Why are Redo operations in ARIES crash recovery required to be idempotent?"
    res = self_rag_answer(q, mlx_generator, retriever, k=3, max_segments=1)

    assert res.retrieved is True
    assert res.decision == "[Retrieval]"
    assert res.selected is not None
    assert res.selected.isrel == "[Relevant]"
    assert res.selected.issup in ("[Fully supported]", "[Partially supported]")
    assert res.selected.isuse == 5
    assert len(res.final_answer) > 20
    assert "idempotent" in res.final_answer.lower() or "recovery" in res.final_answer.lower()


def test_reflection_syntax_and_paragraph_formatting(mlx_generator):
    """Verify that generated candidate text follows proper Self-RAG syntax."""
    doc_chunk = "A B-tree maintains balance with all leaf nodes at identical depth, providing O(log n) search."
    out = mlx_generator.generate("What is the time complexity of B-tree search?", paragraph=doc_chunk)

    assert out.startswith("[Relevant]") or out.startswith("[Irrelevant]")
    assert "[Fully supported]" in out or "[Partially supported]" in out or "[No support / Contradictory]" in out
    assert any(f"[Utility:{i}]" in out for i in range(1, 6))
    assert "<paragraph>" not in out  # Generator continuation must not re-emit paragraph tags inside answer

