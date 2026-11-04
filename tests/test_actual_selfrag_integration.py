"""
EvoRAG Task 7B: Actual Self-RAG Model Integration and Validation Suite.

Validates the complete integration between EvoRAG (Experience Memory, hybrid routing,
conservative validation, provenance extraction) and the actual 7B Self-RAG model:
- Tokenizer reflection-token IDs and decoding fidelity.
- Path A: Direct experience reuse (exact match -> verbatim answer, 0 LLM calls).
- Path B: Experience-assisted Self-RAG (related experience -> both document and experience
  contexts reach the actual model input; model generates fresh answer with combined provenance).
- Path C: Normal Self-RAG fallback (unrelated query -> standard document retrieval + model generation).
- Reflection token candidate scoring and critique behavior.
- Provenance guarantees: original source versions and chunk hashes preserved; synthetic 'mem_...'
  hashes excluded.
- Continuous learning loop: approve_and_save persists fresh answers to SQLite, enabling Path A reuse.
"""
import os
from pathlib import Path
import sys
from typing import Any
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Detect availability of real dependencies
try:
    import faiss
    HAS_FAISS = hasattr(faiss, "IndexFlatIP") and hasattr(faiss, "write_index")
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import llama_cpp
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

from evorag.config import DEFAULT_EMBED_MODEL
from evorag.memory.schema import Experience, VerificationStatus, normalize_question
from evorag.memory.store import ExperienceStore
from evorag.memory.index_builder import ExperienceIndexBuilder
from evorag.memory.retriever import ExperienceRetriever
from evorag.memory.validator import ExperienceValidator
from evorag.controller import EvoRAGController
from retriever.retriever import PassageRetriever
from selfrag.generator import SelfRAGGenerator, DEFAULT_GGUF, format_prompt
from selfrag.pipeline import self_rag_answer, _score_candidate
from selfrag.reflection_tokens import (
    RETRIEVE_YES,
    RETRIEVE_NO,
    RETRIEVE_CONTINUE,
    ISREL_RELEVANT,
    ISREL_IRRELEVANT,
    ISSUP_FULL,
    ISSUP_PARTIAL,
    ISSUP_NONE,
    ISUSE_TOKENS,
    critique_score,
)

pytestmark = pytest.mark.model

EXPECTED_REFLECTION_TOKENS = {
    "[No Retrieval]": 32000,
    "[Retrieval]": 32001,
    "[Continue to Use Evidence]": 32002,
    "[Irrelevant]": 32003,
    "[Relevant]": 32004,
    "<paragraph>": 32005,
    "</paragraph>": 32006,
    "[Utility:1]": 32007,
    "[Utility:2]": 32008,
    "[Utility:3]": 32009,
    "[Utility:4]": 32010,
    "[Utility:5]": 32011,
    "[Fully supported]": 32012,
    "[Partially supported]": 32013,
    "[No support / Contradictory]": 32014,
}


class ModelInvocationTracker:
    """Pass-through wrapper around SelfRAGGenerator tracking calls and prompts."""

    def __init__(self, inner: SelfRAGGenerator):
        self.inner = inner
        self.generate_calls = 0
        self.batch_calls = 0
        self.logged_prompts: list[str] = []
        self.logged_batch_paragraphs: list[list[str]] = []

    def generate(self, instruction: str, paragraph: str | None = None, history: str = "") -> str:
        self.generate_calls += 1
        return self.inner.generate(instruction, paragraph, history)

    def generate_batch(self, instruction: str, paragraphs: list[str], history: str = "") -> list[str]:
        self.batch_calls += 1
        self.logged_batch_paragraphs.append(list(paragraphs))
        for p in paragraphs:
            self.logged_prompts.append(format_prompt(instruction, p, history))
        return self.inner.generate_batch(instruction, paragraphs, history)


@pytest.fixture(scope="session")
def real_selfrag_generator():
    """Load the real 7B Self-RAG model once for the entire test session."""
    if not HAS_LLAMA_CPP:
        pytest.skip("llama-cpp-python is not installed")
    model_path = Path(DEFAULT_GGUF)
    if not model_path.exists():
        pytest.skip(f"Actual Self-RAG checkpoint not found at {model_path}")

    # Use Metal offload on Apple silicon (-1 offloads all layers)
    generator = SelfRAGGenerator(str(model_path), n_gpu_layers=-1)
    return generator


@pytest.fixture
def check_all_dependencies():
    """Ensure FAISS, SentenceTransformer, and llama-cpp are available."""
    if not HAS_FAISS:
        pytest.skip("FAISS is not installed")
    if not HAS_ST:
        pytest.skip("sentence-transformers is not installed")
    if not HAS_LLAMA_CPP:
        pytest.skip("llama-cpp-python is not installed")
    if not Path(DEFAULT_GGUF).exists():
        pytest.skip("Self-RAG GGUF checkpoint is missing")


# =====================================================================
# 1. Reflection-Token & Tokenizer Validation (Step 5)
# =====================================================================

class TestSelfRAGTokenizerAndReflectionTokens:
    """Inspects the actual checkpoint's vocabulary, reflection token IDs, and critique scoring."""

    def test_reflection_token_ids_and_exact_decoding(self, check_all_dependencies, real_selfrag_generator):
        """Verify that every reflection token maps to its exact expected token ID and decodes verbatim."""
        llm: Llama = real_selfrag_generator._llm

        assert llm.n_vocab() == 32016, f"Expected vocabulary size 32016 (32000 base + 16 added), got {llm.n_vocab()}"

        for token_str, expected_id in EXPECTED_REFLECTION_TOKENS.items():
            token_bytes = token_str.encode("utf-8")
            token_ids = llm.tokenize(token_bytes, add_bos=False)
            assert token_ids == [expected_id], (
                f"Token string '{token_str}' tokenized to {token_ids}, expected single ID [{expected_id}]"
            )
            decoded = llm.detokenize(token_ids).decode("utf-8", errors="replace")
            assert decoded == token_str, f"Detokenized string '{decoded}' does not match expected '{token_str}'"

    def test_reflection_token_critique_scoring(self):
        """Verify that candidate scoring correctly applies ISREL, ISSUP, and ISUSE weights."""
        passage = {"text": "A test passage", "chunk_hash": "h1"}
        raw_text = "[Relevant] This is supported by evidence. [Fully supported] [Utility:5]"

        candidate = _score_candidate(passage, raw_text)
        assert candidate.isrel == ISREL_RELEVANT
        assert candidate.issup == ISSUP_FULL
        assert candidate.isuse == 5
        # 1.0 (Relevant) + 1.0 (Fully supported) + (5-3)/2.0 (Utility:5 -> 1.0) = 3.0
        assert candidate.score == 3.0
        assert candidate.answer_text == "This is supported by evidence."


# =====================================================================
# 2. Three-Path Real Model Validation (Steps 2, 3, 4)
# =====================================================================

class TestRealSelfRAGThreePaths:
    """Validates Path A (Direct Reuse), Path B (Experience-Assisted), and Path C (Fallback) with real model."""

    @pytest.fixture
    def test_env(self, check_all_dependencies, tmp_path: Path, real_selfrag_generator):
        """Set up isolated SQLite store, FAISS memory index, and controller with real model."""
        db_path = tmp_path / "actual_test_memory.db"
        index_dir = tmp_path / "actual_exp_index"
        store = ExperienceStore(db_path)

        # Experience 1: Section 14 penalty
        exp_sec14 = Experience(
            question="What is the statutory penalty under Section 14?",
            answer="Under Section 14, the statutory penalty for non-compliance is Rs. 500 per day.",
            evidence=[{
                "text": "The statutory penalty under Section 14 is Rs. 500 per day.",
                "source": "compliance_act_2024.pdf",
                "source_type": "document",
                "page": 1,
                "chunk_id": 0,
                "chunk_hash": "hash_sec14_chunk",
                "source_version": "v1_sha256_sec14",
            }],
            source_versions={"compliance_act_2024.pdf": "v1_sha256_sec14"},
            chunk_hashes=["hash_sec14_chunk"],
            verification_status=VerificationStatus.APPROVED,
        )
        store.add_experience(exp_sec14)

        builder = ExperienceIndexBuilder(index_dir)
        builder.rebuild_from_store(store)

        memory_retriever = ExperienceRetriever(index_dir, store=store)
        validator = ExperienceValidator(similarity_threshold=0.85)

        tracker = ModelInvocationTracker(real_selfrag_generator)
        doc_retriever = PassageRetriever("index")

        controller = EvoRAGController(
            store=store,
            memory_retriever=memory_retriever,
            validator=validator,
            generator=tracker,
            retriever=doc_retriever,
            index_builder=builder,
            k=1,
            max_segments=1,
        )

        return {
            "controller": controller,
            "tracker": tracker,
            "store": store,
            "exp_sec14": exp_sec14,
        }

    def test_path_a_direct_experience_reuse_zero_generator_calls(self, test_env):
        """Exact normalized match returns stored answer verbatim with 0 LLM forward passes."""
        controller = test_env["controller"]
        tracker = test_env["tracker"]
        exp_sec14 = test_env["exp_sec14"]

        query = "  what is the statutory penalty under section 14?  "
        result = controller.answer(query)

        # 1. Routing checks
        assert result.response_mode == "direct_reuse"
        assert result.memory_hit is True
        assert result.memory_context_used is False
        assert result.experience_id == exp_sec14.experience_id

        # 2. Verbatim answer check
        assert result.answer == exp_sec14.answer

        # 3. Provenance check
        assert result.source_versions == exp_sec14.source_versions
        assert result.chunk_hashes == exp_sec14.chunk_hashes

        # 4. Strict generator invocation count check (0 calls)
        assert tracker.generate_calls == 0, f"Expected 0 generate calls, got {tracker.generate_calls}"
        assert tracker.batch_calls == 0, f"Expected 0 batch calls, got {tracker.batch_calls}"

    def test_path_b_experience_assisted_both_contexts_reach_model(self, test_env):
        """Related non-identical question passes document and experience passages into real model."""
        controller = test_env["controller"]
        tracker = test_env["tracker"]
        exp_sec14 = test_env["exp_sec14"]

        # Related question (similarity ~0.90, not exact match)
        query = "What is the statutory penalty for non-compliance under Section 14?"
        result = controller.answer(query)

        # 1. Routing checks
        assert result.response_mode == "experience_assisted"
        assert result.memory_hit is False
        assert result.memory_context_used is True
        assert exp_sec14.experience_id in result.memory_experience_ids

        # 2. Verify model was invoked
        assert tracker.batch_calls >= 1, "Model batch generation must be called on Path B"
        assert len(tracker.logged_batch_paragraphs) >= 1

        # 3. Concrete proof: verify BOTH contexts reached the model input
        batch_passages = tracker.logged_batch_paragraphs[0]
        has_doc_passage = any("Preprint" in p or "Amazon" in p or "Act" in p for p in batch_passages)
        has_exp_passage = any("[PRIOR VERIFIED EXPERIENCE" in p for p in batch_passages)

        assert has_doc_passage, "Document context must reach model input"
        assert has_exp_passage, "Experience context must reach model input"

        # 4. Verify model generated a non-empty fresh answer
        assert len(result.answer) > 10
        assert "500" in result.answer or "penalty" in result.answer.lower()

        # 5. Provenance guarantees
        assert "compliance_act_2024.pdf" in result.source_versions
        assert "hash_sec14_chunk" in result.chunk_hashes
        assert not any(h.startswith("mem_") for h in result.chunk_hashes), (
            "Synthetic memory hashes ('mem_...') must never appear in final chunk_hashes"
        )

    def test_path_c_normal_self_rag_fallback(self, test_env):
        """Unrelated query falls back to standard document retrieval and Self-RAG generation."""
        controller = test_env["controller"]
        tracker = test_env["tracker"]

        query = "How large is the Amazon rainforest?"
        result = controller.answer(query)

        # 1. Routing checks
        assert result.response_mode == "normal_self_rag"
        assert result.memory_hit is False
        assert result.memory_context_used is False
        assert result.fallback_reason is not None

        # 2. Verify model was invoked
        assert tracker.generate_calls >= 1 or tracker.batch_calls >= 1

        # 3. Document retrieval check
        assert len(result.document_evidence) > 0
        assert len(result.experience_evidence) == 0

        # 4. Answer quality check
        assert "Amazon" in result.answer or "million" in result.answer


# =====================================================================
# 3. Provenance & Continuous Learning Loop Validation (Steps 6, 7)
# =====================================================================

class TestRealModelProvenanceAndContinuousLearning:
    """Validates provenance extraction, exclusion of synthetic IDs, and approval persistence."""

    def test_approve_and_save_enables_immediate_path_a_reuse(self, check_all_dependencies, tmp_path: Path, real_selfrag_generator):
        """A fresh answer generated via Self-RAG, when approved, becomes reusable via Path A."""
        # Build a temporary provenance-enabled document index over sample.pdf
        doc_index_dir = tmp_path / "provenance_doc_index"
        from index.build_index import build_index
        build_index(["data/sample.pdf"], str(doc_index_dir))
        doc_retriever = PassageRetriever(str(doc_index_dir))

        db_path = tmp_path / "learning_memory.db"
        index_dir = tmp_path / "learning_exp_index"
        store = ExperienceStore(db_path)
        builder = ExperienceIndexBuilder(index_dir)
        builder.rebuild_from_store(store)

        memory_retriever = ExperienceRetriever(index_dir, store=store)
        validator = ExperienceValidator(similarity_threshold=0.85)
        tracker = ModelInvocationTracker(real_selfrag_generator)

        controller = EvoRAGController(
            store=store,
            memory_retriever=memory_retriever,
            validator=validator,
            generator=tracker,
            retriever=doc_retriever,
            index_builder=builder,
            k=1,
            max_segments=1,
        )

        query = "How large is the Amazon rainforest?"

        # Step 1: Initial query executes through Path C (fallback)
        res_c = controller.answer(query)
        assert res_c.response_mode == "normal_self_rag"
        initial_gen_calls = tracker.generate_calls + tracker.batch_calls
        assert initial_gen_calls > 0

        # Step 2: User explicitly approves and saves the result
        exp_id = controller.approve_and_save(res_c, user_approved=True)
        assert exp_id is not None
        assert store.experience_exists(exp_id)

        # Verify saved experience in SQLite preserves original provenance
        saved_exp = store.get_experience(exp_id)
        assert saved_exp is not None
        assert not any(h.startswith("mem_") for h in saved_exp.chunk_hashes)

        # Step 3: Repeating the query routes through Path A (Direct Reuse, 0 new LLM calls)
        res_a = controller.answer(query)
        assert res_a.response_mode == "direct_reuse"
        assert res_a.memory_hit is True
        assert res_a.answer == res_c.answer
        assert res_a.experience_id == exp_id

        # Total generator calls should not have increased
        final_gen_calls = tracker.generate_calls + tracker.batch_calls
        assert final_gen_calls == initial_gen_calls
