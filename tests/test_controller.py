import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock
import numpy as np
import pytest

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DeterministicTestEmbedder:
    """Produces deterministic normalized 384-dimensional float32 vectors from text hashes."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vecs = []
        for t in texts:
            seed = int(hashlib.sha256(t.strip().lower().encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            v = rng.randn(self.dim).astype("float32")
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                if norm > 0:
                    v /= norm
            vecs.append(v)
        return np.array(vecs, dtype="float32")


class MockIndexFlatIP:
    """In-memory mock for faiss.IndexFlatIP performing true inner product search."""

    def __init__(self, d: int):
        self.d = d
        self.vectors: list[np.ndarray] = []

    def add(self, x: np.ndarray) -> None:
        for row in x:
            self.vectors.append(np.array(row, dtype="float32"))

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def search(self, x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.vectors:
            return np.zeros((len(x), 0), dtype="float32"), np.full((len(x), 0), -1, dtype="int64")

        vecs = np.array(self.vectors, dtype="float32")
        scores = np.dot(x, vecs.T)
        k = min(k, len(self.vectors))
        indices = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, indices, axis=1)
        return top_scores.astype("float32"), indices.astype("int64")


def _mock_write_index(idx, path):
    vecs = getattr(idx, "vectors", [])
    data = {"d": getattr(idx, "d", 384), "vectors": [v.tolist() for v in vecs]}
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def _mock_read_index(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    idx = MockIndexFlatIP(data["d"])
    if data["vectors"]:
        idx.add(np.array(data["vectors"], dtype="float32"))
    return idx


# Mock faiss if not present or bare MagicMock
faiss_mod = sys.modules.get("faiss")
if faiss_mod is None or isinstance(faiss_mod, MagicMock) or not hasattr(faiss_mod, "IndexFlatIP"):
    if faiss_mod is None:
        faiss_mod = MagicMock()
        sys.modules["faiss"] = faiss_mod
    faiss_mod.IndexFlatIP = MockIndexFlatIP
    faiss_mod.write_index = _mock_write_index
    faiss_mod.read_index = _mock_read_index

from evorag.controller import EvoRAGController, MemoryIndexRebuildError
from evorag.memory.schema import Experience, VerificationStatus, normalize_question
from evorag.memory.store import ExperienceStore
from evorag.memory.index_builder import ExperienceIndexBuilder
from evorag.memory.retriever import ExperienceRetriever, ExperienceAugmentedRetriever
from evorag.memory.validator import ExperienceValidator
from evorag.result import EvoRAGResult
from selfrag.pipeline import SelfRAGResult, Candidate, Segment


@pytest.fixture
def test_embedder():
    return DeterministicTestEmbedder(dim=384)


@pytest.fixture
def mock_self_rag_setup(tmp_path: Path, test_embedder):
    """Fixture providing initialized Memory store, index, validator, and mock generator/retriever."""
    db_path = tmp_path / "memory.db"
    index_dir = tmp_path / "memory_index"

    store = ExperienceStore(db_path)
    index_builder = ExperienceIndexBuilder(index_dir=index_dir, embedder=test_embedder)
    retriever = ExperienceRetriever(index_dir=index_dir, store=store, embedder=test_embedder)
    validator = ExperienceValidator(similarity_threshold=0.85)

    mock_generator = MagicMock()
    mock_doc_retriever = MagicMock()

    # Default synthetic passage returned by doc retriever
    mock_passage = {
        "text": "Self-RAG dynamically retrieves passages and scores candidates.",
        "page": 1,
        "source": "selfrag.pdf",
        "source_type": "document",
        "source_version": "v1_doc_hash",
        "chunk_hash": "c_hash_1",
        "chunk_id": 0,
    }
    mock_doc_retriever.search.return_value = [mock_passage]

    # Default synthetic generator outputs
    mock_generator.generate.return_value = "[Retrieval]"
    mock_generator.generate_batch.return_value = [
        "[Relevant]Freshly generated Self-RAG answer response.[Fully supported][Utility:5]"
    ]

    controller = EvoRAGController(
        store=store,
        memory_retriever=retriever,
        validator=validator,
        generator=mock_generator,
        retriever=mock_doc_retriever,
        index_builder=index_builder,
        k=2,
    )

    return {
        "controller": controller,
        "store": store,
        "index_builder": index_builder,
        "retriever": retriever,
        "validator": validator,
        "generator": mock_generator,
        "doc_retriever": mock_doc_retriever,
        "db_path": db_path,
        "index_dir": index_dir,
    }


class TestEvoRAGController:
    """Comprehensive tests for Controller orchestration, Three-Path Hybrid Routing, and Provenance."""

    # -------------------------------------------------------------
    # Path A: Exact Direct Reuse
    # -------------------------------------------------------------

    def test_exact_normalized_match_uses_direct_reuse(self, mock_self_rag_setup):
        """Verify exact normalized match triggers Path A (direct_reuse) without calling Self-RAG."""
        c = mock_self_rag_setup["controller"]
        store = mock_self_rag_setup["store"]
        builder = mock_self_rag_setup["index_builder"]
        generator = mock_self_rag_setup["generator"]

        stored_q = "What is Self-RAG?"
        stored_a = "Self-RAG is an adaptive retrieval-augmented generation framework."
        exp = Experience(
            question=stored_q,
            answer=stored_a,
            evidence=[{"text": "evidence text", "source": "paper.pdf", "chunk_hash": "h1"}],
            source_versions={"paper.pdf": "v1"},
            chunk_hashes=["h1"],
            verification_status=VerificationStatus.APPROVED,
        )
        store.add_experience(exp)
        builder.rebuild_from_store(store)
        c.memory_retriever._load_index()

        # Query with casing and spacing differences (exact normalized match)
        query = "  what is   SELF-RAG? "
        res = c.answer(query)

        assert res.memory_hit is True
        assert res.response_mode == "direct_reuse"
        assert res.memory_context_used is False
        assert res.answer == stored_a
        assert res.experience_id == exp.experience_id
        assert res.similarity_score >= 0.85
        assert res.self_rag_result is None

        # Generator was NOT invoked
        assert generator.generate.call_count == 0
        assert generator.generate_batch.call_count == 0

    def test_semantically_similar_non_identical_question_does_not_use_direct_reuse(
        self, mock_self_rag_setup
    ):
        """Verify high semantic similarity alone never triggers direct reuse if question is not identical."""
        c = mock_self_rag_setup["controller"]
        store = mock_self_rag_setup["store"]
        builder = mock_self_rag_setup["index_builder"]
        # Allow validation threshold to pass in mock embedding test
        c.validator.similarity_threshold = -1.0

        stored_q = "What is Self-RAG?"
        stored_a = "Stored baseline answer"
        exp = Experience(
            question=stored_q,
            answer=stored_a,
            evidence=[{"text": "ev", "source": "p.pdf", "chunk_hash": "h1"}],
            source_versions={"p.pdf": "v1"},
            chunk_hashes=["h1"],
            verification_status=VerificationStatus.APPROVED,
        )
        store.add_experience(exp)
        builder.rebuild_from_store(store)
        c.memory_retriever._load_index()

        query = "Explain the concepts behind Self-RAG"
        res = c.answer(query)

        # Must NOT be direct_reuse
        assert res.memory_hit is False
        assert res.response_mode == "experience_assisted"
        assert res.memory_context_used is True
        assert res.answer != stored_a  # Uses freshly generated answer, not stored answer!

    # -------------------------------------------------------------
    # Path B: Related Experience-Assisted Generation
    # -------------------------------------------------------------

    def test_related_approved_experience_reaches_self_rag_as_context(self, mock_self_rag_setup):
        """Verify related experience is injected as context and Self-RAG generates fresh answer."""
        c = mock_self_rag_setup["controller"]
        store = mock_self_rag_setup["store"]
        builder = mock_self_rag_setup["index_builder"]
        generator = mock_self_rag_setup["generator"]
        doc_retriever = mock_self_rag_setup["doc_retriever"]
        c.validator.similarity_threshold = -1.0

        exp = Experience(
            question="What reflection tokens exist?",
            answer="Retrieve, ISREL, ISSUP, ISUSE.",
            evidence=[{"text": "Tokens are...", "source": "paper.pdf", "chunk_hash": "h2"}],
            source_versions={"paper.pdf": "v1"},
            chunk_hashes=["h2"],
            verification_status=VerificationStatus.APPROVED,
        )
        store.add_experience(exp)
        builder.rebuild_from_store(store)
        c.memory_retriever._load_index()

        query = "Tell me about reflection tokens used in Self-RAG"
        res = c.answer(query)

        assert res.response_mode == "experience_assisted"
        assert res.memory_hit is False
        assert res.memory_context_used is True
        assert len(res.memory_experience_ids) == 1
        assert res.memory_experience_ids[0] == exp.experience_id
        assert "Freshly generated Self-RAG answer" in res.answer

        # Self-RAG was invoked exactly once
        assert generator.generate.call_count == 1
        # Original document retrieval still occurred
        assert doc_retriever.search.call_count >= 1

    def test_memory_derived_evidence_retains_provenance(self, mock_self_rag_setup):
        """Verify augmented retriever embeds experience provenance inside combined doc passages."""
        base_retriever = mock_self_rag_setup["doc_retriever"]
        exp = Experience(
            question="Sample Q",
            answer="Sample A",
            evidence=[{"text": "Prior evidence passage"}],
            source_versions={"doc.pdf": "v1"},
            chunk_hashes=["hash_sample"],
        )
        aug_retriever = ExperienceAugmentedRetriever(base_retriever, [exp])

        passages = aug_retriever.search("test query", k=2)

        # Combined-context mode: only document passages are returned (experience embedded inside)
        doc_passages = [p for p in passages if p.get("source_type") == "document"]
        assert len(doc_passages) == 1
        assert doc_passages[0]["chunk_id"] == 0

        # Experience text is embedded inside the doc passage's text
        assert "[PRIOR VERIFIED EXPERIENCE" in doc_passages[0]["text"]
        assert "Sample Q" in doc_passages[0]["text"]
        assert "Sample A" in doc_passages[0]["text"]

        # Experience provenance is accessible via _embedded_exp_passages
        assert "_embedded_exp_passages" in doc_passages[0]
        embedded = doc_passages[0]["_embedded_exp_passages"]
        assert len(embedded) == 1
        ep = embedded[0]
        assert ep["experience_id"] == exp.experience_id
        assert ep["original_question"] == exp.question
        assert ep["original_answer"] == exp.answer
        assert ep["original_source_versions"] == exp.source_versions
        assert ep["original_chunk_hashes"] == exp.chunk_hashes
        assert ep["source_type"] == "experience_memory"

        # No standalone experience_memory passages in the returned list
        mem_passages = [p for p in passages if p.get("source_type") == "experience_memory"]
        assert len(mem_passages) == 0

    # -------------------------------------------------------------
    # Path C: Normal Self-RAG (Exclusions and Fallback)
    # -------------------------------------------------------------

    def test_no_valid_experience_invokes_normal_self_rag(self, mock_self_rag_setup):
        """Verify empty memory or invalid candidates result in Path C (normal_self_rag)."""
        c = mock_self_rag_setup["controller"]

        res = c.answer("Question with empty memory")

        assert res.response_mode == "normal_self_rag"
        assert res.memory_hit is False
        assert res.memory_context_used is False
        assert res.memory_experiences == []
        assert res.memory_experience_ids == []
        assert len(res.document_evidence) > 0
        assert len(res.experience_evidence) == 0

    def test_stale_source_version_falls_back_to_normal_self_rag(
        self, mock_self_rag_setup, tmp_path: Path
    ):
        """Verify candidate with modified source document is excluded from assisting context."""
        c = mock_self_rag_setup["controller"]
        store = mock_self_rag_setup["store"]
        builder = mock_self_rag_setup["index_builder"]
        c.validator.similarity_threshold = -1.0

        # Document index with v2
        doc_idx_dir = tmp_path / "doc_idx"
        doc_idx_dir.mkdir(parents=True, exist_ok=True)
        (doc_idx_dir / "meta.json").write_text(
            json.dumps({"source_versions": {"paper.pdf": "v2_new_hash"}}),
            encoding="utf-8",
        )
        c.doc_index_dir = doc_idx_dir

        # Experience references v1
        exp = Experience(
            question="Question about paper",
            answer="Answer",
            evidence=[{"text": "ev", "source": "paper.pdf"}],
            source_versions={"paper.pdf": "v1_old_hash"},
            chunk_hashes=["h1"],
        )
        store.add_experience(exp)
        builder.rebuild_from_store(store)
        c.memory_retriever._load_index()

        res = c.answer("Question about paper")

        # Must fall back to normal Self-RAG
        assert res.response_mode == "normal_self_rag"
        assert res.memory_context_used is False
        assert "invalidated by modified source document" in res.fallback_reason

    def test_polarity_mismatch_excluded_from_context(self, mock_self_rag_setup):
        """Verify negation polarity mismatch ('What is covered?' vs 'What is NOT covered?') falls back to normal."""
        c = mock_self_rag_setup["controller"]
        store = mock_self_rag_setup["store"]
        builder = mock_self_rag_setup["index_builder"]
        c.validator.similarity_threshold = -1.0

        exp = Experience(
            question="What is covered?",
            answer="Covered items.",
            evidence=[{"text": "ev", "source": "d.pdf"}],
            source_versions={"d.pdf": "v1"},
            chunk_hashes=["h1"],
        )
        store.add_experience(exp)
        builder.rebuild_from_store(store)
        c.memory_retriever._load_index()

        res = c.answer("What is NOT covered?")

        assert res.response_mode == "normal_self_rag"
        assert res.memory_context_used is False
        assert "Polarity mismatch" in res.fallback_reason

    def test_scope_mismatch_excluded_from_context(self, mock_self_rag_setup):
        """Verify numeric scope mismatch ('Section 10' vs 'Section 14') falls back to normal."""
        c = mock_self_rag_setup["controller"]
        store = mock_self_rag_setup["store"]
        builder = mock_self_rag_setup["index_builder"]
        c.validator.similarity_threshold = -1.0

        exp = Experience(
            question="Section 10 details",
            answer="Section 10 details",
            evidence=[{"text": "ev", "source": "d.pdf"}],
            source_versions={"d.pdf": "v1"},
            chunk_hashes=["h1"],
        )
        store.add_experience(exp)
        builder.rebuild_from_store(store)
        c.memory_retriever._load_index()

        res = c.answer("Section 14 details")

        assert res.response_mode == "normal_self_rag"
        assert res.memory_context_used is False
        assert "Scope mismatch" in res.fallback_reason

    # -------------------------------------------------------------
    # Approval and Saving
    # -------------------------------------------------------------

    def test_explicit_approval_required_to_save(self, mock_self_rag_setup):
        """Verify saving requires explicit user approval."""
        c = mock_self_rag_setup["controller"]
        store = mock_self_rag_setup["store"]

        res = c.answer("Unseen question")
        assert res.response_mode == "normal_self_rag"

        # user_approved = False
        assert c.approve_and_save(res, user_approved=False) is None
        assert store.count() == 0

        # user_approved = True
        saved_id = c.approve_and_save(res, user_approved=True)
        assert saved_id is not None
        assert store.count() == 1

    def test_direct_reuse_memory_hit_answers_not_saved_again(self, mock_self_rag_setup):
        """Verify answers produced via direct_reuse are not duplicated into memory."""
        c = mock_self_rag_setup["controller"]
        store = mock_self_rag_setup["store"]
        builder = mock_self_rag_setup["index_builder"]

        q = "What is 2+2?"
        exp = Experience(
            question=q,
            answer="4",
            evidence=[{"text": "math", "source": "m.pdf"}],
            source_versions={"m.pdf": "v1"},
            chunk_hashes=["h1"],
        )
        store.add_experience(exp)
        builder.rebuild_from_store(store)
        c.memory_retriever._load_index()

        hit_res = c.answer(q)
        assert hit_res.response_mode == "direct_reuse"

        assert c.approve_and_save(hit_res, user_approved=True) is None
        assert store.count() == 1

    def test_index_is_rebuilt_after_approval(self, mock_self_rag_setup):
        """Verify approving newly generated answer rebuilds memory index and enables direct reuse."""
        c = mock_self_rag_setup["controller"]

        q = "Unique test question for learning"
        res1 = c.answer(q)
        assert res1.response_mode == "normal_self_rag"

        saved_id = c.approve_and_save(res1, user_approved=True)
        assert saved_id is not None

        # Repeat same query
        res2 = c.answer(q)
        assert res2.response_mode == "direct_reuse"
        assert res2.memory_hit is True
        assert res2.experience_id == saved_id

    # -------------------------------------------------------------
    # Task 6B - Issue 1: Provenance Extraction Regression Tests
    # -------------------------------------------------------------

    def test_extract_provenance_document_only(self, mock_self_rag_setup):
        """Document-only evidence preserves document chunk hash and source version."""
        c = mock_self_rag_setup["controller"]
        doc_p = {
            "text": "Doc text",
            "source": "doc1.pdf",
            "source_version": "v1_hash",
            "chunk_hash": "chunk_h1",
            "source_type": "document",
        }
        mock_res = MagicMock()
        mock_cand = MagicMock()
        mock_cand.passage = doc_p
        mock_res.segments = [MagicMock(selected=mock_cand)]

        evidence, chunk_hashes, source_versions = c._extract_provenance(mock_res)
        assert len(evidence) == 1
        assert chunk_hashes == ["chunk_h1"]
        assert source_versions == {"doc1.pdf": "v1_hash"}

    def test_extract_provenance_experience_only(self, mock_self_rag_setup):
        """Experience-only evidence contributes original document chunk hashes and versions, dropping synthetic mem_ hash."""
        c = mock_self_rag_setup["controller"]
        mem_p = {
            "text": "Prior experience text",
            "source": "experience_memory",
            "source_type": "experience_memory",
            "chunk_id": "exp_12345",
            "chunk_hash": "mem_12345",
            "original_chunk_hashes": ["orig_chunk_1", "orig_chunk_2"],
            "original_source_versions": {"paper.pdf": "v_paper_1"},
        }
        mock_res = MagicMock()
        mock_cand = MagicMock()
        mock_cand.passage = mem_p
        mock_res.segments = [MagicMock(selected=mock_cand)]

        evidence, chunk_hashes, source_versions = c._extract_provenance(mock_res)
        assert len(evidence) == 1
        assert chunk_hashes == ["orig_chunk_1", "orig_chunk_2"]
        assert "mem_12345" not in chunk_hashes
        assert source_versions == {"paper.pdf": "v_paper_1"}
        assert "experience_memory" not in source_versions

    def test_extract_provenance_mixed_document_and_experience(self, mock_self_rag_setup):
        """Mixed document and experience evidence combines provenance without duplicates."""
        c = mock_self_rag_setup["controller"]
        doc_p = {
            "text": "Doc text",
            "source": "doc1.pdf",
            "source_version": "v1",
            "chunk_hash": "ch_doc_1",
            "source_type": "document",
        }
        mem_p = {
            "text": "Prior experience text",
            "source": "experience_memory",
            "source_type": "experience_memory",
            "chunk_hash": "mem_abc",
            "original_chunk_hashes": ["ch_orig_2"],
            "original_source_versions": {"doc2.pdf": "v2"},
        }
        mock_res = MagicMock()
        cand1 = MagicMock(passage=doc_p)
        cand2 = MagicMock(passage=mem_p)
        mock_res.segments = [MagicMock(selected=cand1), MagicMock(selected=cand2)]

        evidence, chunk_hashes, source_versions = c._extract_provenance(mock_res)
        assert len(evidence) == 2
        assert chunk_hashes == ["ch_doc_1", "ch_orig_2"]
        assert source_versions == {"doc1.pdf": "v1", "doc2.pdf": "v2"}

    def test_extract_provenance_multiple_experiences_same_source(self, mock_self_rag_setup):
        """Multiple experiences referencing the same source map to the correct source version."""
        c = mock_self_rag_setup["controller"]
        mem1 = {
            "text": "Exp 1",
            "source": "experience_memory",
            "source_type": "experience_memory",
            "chunk_hash": "mem_1",
            "original_chunk_hashes": ["ch_common", "ch_exp1"],
            "original_source_versions": {"shared.pdf": "v_shared"},
        }
        mem2 = {
            "text": "Exp 2",
            "source": "experience_memory",
            "source_type": "experience_memory",
            "chunk_hash": "mem_2",
            "original_chunk_hashes": ["ch_common", "ch_exp2"],
            "original_source_versions": {"shared.pdf": "v_shared"},
        }
        mock_res = MagicMock()
        mock_res.segments = [
            MagicMock(selected=MagicMock(passage=mem1)),
            MagicMock(selected=MagicMock(passage=mem2)),
        ]

        evidence, chunk_hashes, source_versions = c._extract_provenance(mock_res)
        assert len(evidence) == 2
        assert chunk_hashes == ["ch_common", "ch_exp1", "ch_exp2"]
        assert source_versions == {"shared.pdf": "v_shared"}

    def test_extract_provenance_missing_or_malformed_fields(self, mock_self_rag_setup):
        """Malformed or missing provenance fields are handled safely without exceptions."""
        c = mock_self_rag_setup["controller"]
        malformed_passages = [
            {"source_type": "experience_memory", "original_chunk_hashes": None, "original_source_versions": "not_a_dict"},
            {"source_type": "document", "source": 123, "source_version": None, "chunk_hash": None},
            "not_a_dict_passage",
            {"text": "empty"},
        ]
        mock_res = MagicMock()
        mock_res.segments = [MagicMock(selected=MagicMock(passage=p)) for p in malformed_passages]

        evidence, chunk_hashes, source_versions = c._extract_provenance(mock_res)
        assert isinstance(chunk_hashes, list)
        assert isinstance(source_versions, dict)
        assert len(chunk_hashes) == 0
        assert len(source_versions) == 0

    def test_approve_and_save_after_experience_assisted_generation(self, mock_self_rag_setup):
        """Ensure approve_and_save succeeds for experience-assisted answers because provenance is preserved."""
        c = mock_self_rag_setup["controller"]
        store = mock_self_rag_setup["store"]
        builder = mock_self_rag_setup["index_builder"]
        c.validator.similarity_threshold = -1.0

        initial_exp = Experience(
            question="What is Self-RAG?",
            answer="Adaptive RAG framework.",
            evidence=[{"text": "text", "source": "paper.pdf", "chunk_hash": "h1"}],
            source_versions={"paper.pdf": "v1"},
            chunk_hashes=["h1"],
        )
        store.add_experience(initial_exp)
        builder.rebuild_from_store(store)
        c.memory_retriever._load_index()

        res = c.answer("Explain how Self-RAG functions")
        assert res.response_mode == "experience_assisted"
        assert res.source_versions

        saved_id = c.approve_and_save(res, user_approved=True)
        assert saved_id is not None
        assert store.count() == 2

    # -------------------------------------------------------------
    # Task 6B - Issue 2: Retrieval Budget Semantics Regression Tests
    # -------------------------------------------------------------

    def test_retrieval_budget_k5_two_experiences(self):
        """k=5 with 2 experiences -> 5 combined passages (experience context embedded inline)."""
        base_retriever = MagicMock()
        base_retriever.search.return_value = [{"text": f"doc {i}", "source": f"d{i}.pdf"} for i in range(5)]
        exps = [
            Experience(question=f"Q{i}", answer=f"A{i}", evidence=[{"text": "ev"}], source_versions={"d.pdf": "v1"}, chunk_hashes=[f"h{i}"])
            for i in range(2)
        ]
        aug = ExperienceAugmentedRetriever(base_retriever, exps)
        res = aug.search("query", k=5)
        # With combined-context mode, experience text is embedded inside doc passages.
        # Document count is preserved (k=5); no separate experience candidates are added.
        assert len(res) == 5
        assert sum(1 for p in res if p.get("source_type") == "document") == 5
        assert sum(1 for p in res if p.get("source_type") == "experience_memory") == 0
        # Every combined passage carries embedded experience metadata
        assert all("_embedded_exp_passages" in p for p in res)
        assert all(len(p["_embedded_exp_passages"]) == 2 for p in res)
        # Experience text is present inside each doc passage's text
        assert all("[PRIOR VERIFIED EXPERIENCE" in p["text"] for p in res)
        base_retriever.search.assert_called_once_with("query", k=5)

    def test_retrieval_budget_k5_zero_experiences(self):
        """k=5 with 0 experiences -> 5 total passages returned."""
        base_retriever = MagicMock()
        base_retriever.search.return_value = [{"text": f"doc {i}", "source": f"d{i}.pdf"} for i in range(5)]
        aug = ExperienceAugmentedRetriever(base_retriever, [])
        res = aug.search("query", k=5)
        assert len(res) == 5
        assert sum(1 for p in res if p.get("source_type") == "document") == 5
        assert sum(1 for p in res if p.get("source_type") == "experience_memory") == 0
        base_retriever.search.assert_called_once_with("query", k=5)

    def test_retrieval_budget_k1_two_experiences(self):
        """k=1 with 2 experiences -> 1 combined passage (experience context embedded inline)."""
        base_retriever = MagicMock()
        base_retriever.search.return_value = [{"text": "doc 0", "source": "d0.pdf"}]
        exps = [
            Experience(question=f"Q{i}", answer=f"A{i}", evidence=[{"text": "ev"}], source_versions={"d.pdf": "v1"}, chunk_hashes=[f"h{i}"])
            for i in range(2)
        ]
        aug = ExperienceAugmentedRetriever(base_retriever, exps)
        res = aug.search("query", k=1)
        # Combined-context mode: 1 doc passage with exp text embedded = 1 total passage
        assert len(res) == 1
        assert sum(1 for p in res if p.get("source_type") == "document") == 1
        assert sum(1 for p in res if p.get("source_type") == "experience_memory") == 0
        # Both experience texts are present inside the single combined passage
        assert len(res[0]["_embedded_exp_passages"]) == 2
        assert "[PRIOR VERIFIED EXPERIENCE" in res[0]["text"]
        base_retriever.search.assert_called_once_with("query", k=1)

    def test_retrieval_budget_preserves_document_passages_without_displacement(self):
        """Document passages are never evicted or displaced by memory passages."""
        base_retriever = MagicMock()
        doc_passages = [{"text": f"original doc passage {i}", "chunk_id": i} for i in range(4)]
        base_retriever.search.return_value = doc_passages
        exps = [Experience(question="Q", answer="A", evidence=[], source_versions={"p.pdf": "v1"}, chunk_hashes=["h"])]
        aug = ExperienceAugmentedRetriever(base_retriever, exps)
        res = aug.search("query", k=4)
        # All 4 document passages are present in the results (embedded with experience text)
        assert len(res) == 4
        for i in range(4):
            assert any(f"original doc passage {i}" in p["text"] for p in res)

    # -------------------------------------------------------------
    # Task 6B - Issue 3: Recoverable SQLite/FAISS Persistence Tests
    # -------------------------------------------------------------

    def test_approve_and_save_index_rebuild_failure_preserves_sqlite_and_raises(self, mock_self_rag_setup):
        """When FAISS rebuild fails, SQLite record is preserved, in-memory index is invalidated, and error is raised."""
        c = mock_self_rag_setup["controller"]
        store = mock_self_rag_setup["store"]
        builder = mock_self_rag_setup["index_builder"]

        res = c.answer("Question for rebuild failure test")
        assert res.response_mode == "normal_self_rag"

        # Mock builder.rebuild_from_store to fail
        builder.rebuild_from_store = MagicMock(side_effect=IOError("Simulated disk error writing FAISS index"))

        with pytest.raises(MemoryIndexRebuildError) as exc_info:
            c.approve_and_save(res, user_approved=True)

        assert "Simulated disk error" in str(exc_info.value)
        assert c.last_index_error is not None
        assert "Simulated disk error" in c.last_index_error

        # SQLite record is safely preserved!
        assert store.count() == 1
        # In-memory retriever was invalidated so it won't query corrupt/out-of-sync index
        assert c.memory_retriever._index is None

    def test_recover_index_restores_faiss_and_retriever(self, mock_self_rag_setup):
        """Calling recover_index() restores the FAISS index and refreshes retriever."""
        c = mock_self_rag_setup["controller"]
        store = mock_self_rag_setup["store"]

        # Insert directly into SQLite (source of truth)
        exp = Experience(
            question="Recoverable question?",
            answer="Recoverable answer.",
            evidence=[{"text": "ev", "source": "d.pdf"}],
            source_versions={"d.pdf": "v1"},
            chunk_hashes=["h1"],
        )
        store.add_experience(exp)

        # Invalidate in-memory retriever
        c.memory_retriever.invalidate_index()
        assert c.memory_retriever._index is None

        # Call recover_index
        meta = c.recover_index()
        assert meta["num_vectors"] == 1
        assert meta["id_map"] == [exp.experience_id]
        assert c.memory_retriever._index is not None
        assert c.last_index_error is None

        # Verify direct reuse now succeeds
        hit_res = c.answer("Recoverable question?")
        assert hit_res.response_mode == "direct_reuse"
        assert hit_res.answer == "Recoverable answer."

    def test_repeated_recovery_avoids_duplicate_records(self, mock_self_rag_setup):
        """Repeated recovery or approve_and_save retries never duplicate SQLite rows."""
        c = mock_self_rag_setup["controller"]
        store = mock_self_rag_setup["store"]

        res = c.answer("Duplicate prevention question")
        assert res.response_mode == "normal_self_rag"

        # Save once
        id1 = c.approve_and_save(res, user_approved=True)
        assert id1 is not None
        assert store.count() == 1

        # Save same result again (e.g. retry after recovery)
        id2 = c.approve_and_save(res, user_approved=True)
        assert id2 == id1
        assert store.count() == 1

        # Call recover_index multiple times
        c.recover_index()
        c.recover_index()
        assert store.count() == 1
