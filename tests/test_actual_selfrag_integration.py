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


# =====================================================================
# 4. Combined-Context Path B Validation + Grounding Check (Task 7B.1)
# =====================================================================

class TestPathBCombinedContextAndGrounding:
    """
    Task 7B.1: Validates that Path B passes BOTH document evidence and experience
    evidence inside the SAME Self-RAG generation prompt (<paragraph> block), and
    adds an explicit grounding check for incorrect legal-source attribution.

    Uses data/sample.pdf (Amazon rainforest) and an approved experience about the
    same topic so the retrieved document is semantically relevant to the experience.
    This avoids domain-mismatch artefacts and gives the model a clean grounding
    signal to generate a verifiable answer.
    """

    @pytest.fixture
    def combined_ctx_env(self, check_all_dependencies, tmp_path: Path,
                         real_selfrag_generator):
        """
        Build a provenance-backed document index over sample.pdf, store one
        approved Amazon experience, and wire up the full EvoRAG controller.
        """
        from index.build_index import build_index

        # Build a real document FAISS index from sample.pdf (Amazon content)
        doc_index_dir = tmp_path / "combined_doc_index"
        build_index(["data/sample.pdf"], str(doc_index_dir))
        doc_retriever = PassageRetriever(str(doc_index_dir))

        # Verify at least one doc passage retrieved for a matching query
        sample_passages = doc_retriever.search("Amazon rainforest area", k=1)
        assert len(sample_passages) >= 1, "sample.pdf must yield at least one passage"
        doc_text = sample_passages[0]["text"]
        doc_source = sample_passages[0]["source"]
        doc_source_version = sample_passages[0].get("source_version", "")
        doc_chunk_hash = sample_passages[0].get("chunk_hash", "")

        # Experience: approved prior knowledge about the Amazon
        db_path = tmp_path / "combined_ctx_memory.db"
        index_dir = tmp_path / "combined_ctx_exp_index"
        store = ExperienceStore(db_path)

        exp_amazon = Experience(
            question="How large is the Amazon rainforest?",
            answer="The Amazon rainforest covers approximately 5.5 million square kilometres.",
            evidence=[{
                "text": doc_text,
                "source": doc_source,
                "source_type": "document",
                "page": sample_passages[0].get("page", 1),
                "chunk_id": sample_passages[0].get("chunk_id", 0),
                "chunk_hash": doc_chunk_hash,
                "source_version": doc_source_version,
            }],
            source_versions={doc_source: doc_source_version} if doc_source_version else {},
            chunk_hashes=[doc_chunk_hash] if doc_chunk_hash else [],
            verification_status=VerificationStatus.APPROVED,
        )
        store.add_experience(exp_amazon)

        builder = ExperienceIndexBuilder(index_dir)
        builder.rebuild_from_store(store)
        memory_retriever = ExperienceRetriever(index_dir, store=store)
        validator = ExperienceValidator(similarity_threshold=0.75)

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

        return {
            "controller": controller,
            "tracker": tracker,
            "store": store,
            "exp_amazon": exp_amazon,
            "doc_source": doc_source,
            "doc_chunk_hash": doc_chunk_hash,
            "doc_source_version": doc_source_version,
        }

    def test_path_b_combined_context_in_single_prompt(self, combined_ctx_env):
        """
        Path B must supply both document and experience text inside the SAME
        <paragraph> block of the actual Self-RAG generation prompt.

        Verification approach:
        - Capture the raw paragraph strings passed to generate_batch.
        - Assert that at least one paragraph string contains BOTH document text
          AND the delimited experience marker.
        - Assert the Self-RAG instruction format (### Instruction / ### Response) is
          preserved around the paragraph.
        - Assert the generated answer is not the verbatim stored answer (fresh generation).
        """
        env = combined_ctx_env
        controller = env["controller"]
        tracker = env["tracker"]
        exp_amazon = env["exp_amazon"]

        # Related query that is semantically similar but not an exact match
        query = "What is the size of the Amazon rainforest in square kilometres?"
        result = controller.answer(query)

        # 1. Routing check: must be experience_assisted (not direct_reuse, not normal_self_rag)
        assert result.response_mode == "experience_assisted", (
            f"Expected experience_assisted routing, got {result.response_mode!r}. "
            f"Fallback reason: {result.fallback_reason}"
        )
        assert result.memory_hit is False
        assert result.memory_context_used is True
        assert exp_amazon.experience_id in result.memory_experience_ids

        # 2. Model was actually invoked (not a cache hit)
        assert tracker.batch_calls >= 1, "generate_batch must be called for Path B"

        # 3. The KEY check: both doc and experience text appear in the SAME paragraph string
        all_paragraphs = [p for batch in tracker.logged_batch_paragraphs for p in batch]
        assert len(all_paragraphs) >= 1, "At least one paragraph must reach generate_batch"

        combined_paragraphs = [
            p for p in all_paragraphs
            if "[PRIOR VERIFIED EXPERIENCE" in p and "Amazon" in p
        ]
        assert len(combined_paragraphs) >= 1, (
            "Expected at least one paragraph containing BOTH 'Amazon' (document text) "
            "and '[PRIOR VERIFIED EXPERIENCE' (experience context) in the SAME string.\n"
            f"Paragraphs received:\n" +
            "\n---\n".join(all_paragraphs[:3])
        )

        # 4. The official Self-RAG prompt format is preserved in the full logged prompt
        assert len(tracker.logged_prompts) >= 1
        full_prompt = tracker.logged_prompts[0]
        assert "### Instruction:" in full_prompt
        assert "### Response:" in full_prompt
        assert "<paragraph>" in full_prompt
        assert "</paragraph>" in full_prompt

        # 5. Answer is freshly generated, not the verbatim stored answer
        assert result.answer != exp_amazon.answer, (
            "Path B must generate a fresh answer, not return the stored experience verbatim"
        )
        assert len(result.answer) > 10, "Generated answer must be non-trivial"

    def test_path_b_provenance_preserved_no_synthetic_hashes(self, combined_ctx_env):
        """
        Verify that Path B preserves both document and experience provenance,
        and that synthetic mem_... hashes are excluded from final chunk_hashes.
        """
        env = combined_ctx_env
        controller = env["controller"]
        exp_amazon = env["exp_amazon"]
        doc_chunk_hash = env["doc_chunk_hash"]
        doc_source = env["doc_source"]
        doc_source_version = env["doc_source_version"]

        query = "What is the size of the Amazon rainforest in square kilometres?"
        result = controller.answer(query)

        if result.response_mode != "experience_assisted":
            pytest.skip(
                f"Path B not taken ({result.response_mode}); "
                "provenance test requires experience_assisted routing"
            )

        # No synthetic mem_... hashes in the reported chunk_hashes
        assert not any(h.startswith("mem_") for h in result.chunk_hashes), (
            f"Synthetic 'mem_...' hashes must never appear in chunk_hashes. "
            f"Got: {result.chunk_hashes}"
        )

        # Experience evidence is recorded separately from document evidence
        assert result.memory_context_used is True
        # mem_ev comes from embedded experience passages
        assert len(result.experience_evidence) >= 1 or len(result.memory_experience_ids) >= 1

    def test_path_b_legal_domain_grounding_check(self, check_all_dependencies,
                                                  tmp_path: Path, real_selfrag_generator):
        """
        Grounding audit for the legal-domain Section 14 scenario documented in Task 7B.

        The document index has NO content about any compliance act. The only relevant
        context is the experience passage ("Rs. 500 per day under Section 14").  When
        the model generates an answer attributing the penalty to a specific named act
        (e.g. 'Income Tax Act'), that attribution is NOT supported by any retrieved
        passage — it comes from the model's parametric memory.

        This test:
        - Performs the same Section 14 Path B query used in Task 7B.
        - Captures the actual combined prompt supplied to the model.
        - Runs a grounding check: if the answer names a specific legal act, that act
          name must appear in at least one of the retrieved passage texts.
        - Reports the grounding result honestly: FAIL if the named act is unsupported,
          PASS with a warning note if the answer is suitably generic.

        This test does NOT assert on the specific answer text; it asserts on
        grounding integrity.
        """
        db_path = tmp_path / "legal_grounding.db"
        index_dir = tmp_path / "legal_grounding_exp_index"
        store = ExperienceStore(db_path)

        # Approved experience: Section 14 penalty with synthetic provenance
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
        doc_retriever = PassageRetriever("index")  # production index (no compliance act content)

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

        query = "What is the statutory penalty for non-compliance under Section 14?"
        result = controller.answer(query)

        # Collect all passage texts that actually reached the model
        all_paragraphs = [p for batch in tracker.logged_batch_paragraphs for p in batch]
        all_passage_text = "\n".join(all_paragraphs).lower()

        # Grounding check: if the answer names a specific legal act, it must appear in passages
        # Known hallucination pattern from Task 7B: model writes "Income Tax Act" despite
        # no passage mentioning it.
        LEGAL_ACT_PATTERNS = [
            "income tax act",
            "consumer protection act",
            "companies act",
            "prevention of money laundering",
            "information technology act",
        ]
        answer_lower = result.answer.lower()
        unsupported_attributions = [
            act for act in LEGAL_ACT_PATTERNS
            if act in answer_lower and act not in all_passage_text
        ]

        # Document grounding status: record for the report
        grounding_note = (
            f"Answer: {result.answer!r}\n"
            f"Path taken: {result.response_mode}\n"
            f"All passage text (first 400 chars): {all_passage_text[:400]}\n"
            f"Unsupported legal attributions: {unsupported_attributions}"
        )

        # Hard assertion: hallucinated specific act names are a grounding failure
        assert len(unsupported_attributions) == 0, (
            "GROUNDING FAILURE: The model attributed the answer to a specific legal act "
            "that does not appear in any retrieved passage. This is a hallucination from "
            "the model's parametric memory, not grounded in the supplied evidence.\n\n"
            + grounding_note
        )
