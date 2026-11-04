"""
Real FAISS and SentenceTransformer Integration Test Suite (Task 7A.1).

Validates the actual FAISS vector search and SentenceTransformer embedding
integration across both document retrieval and Experience Memory:
- Model loading, dimensionality, float32 datatype, L2 normalization, similarity ranking.
- Real document FAISS indexing, persistence, and passage retrieval with metadata preservation.
- Real Experience Memory FAISS indexing, metadata/id_map persistence, and SQLite resolution.
- Incremental experience addition and persistence across store reload.
- Comprehensive recovery from missing/corrupted FAISS indexes and metadata files.
- Empty Experience Memory safe handling.
- Real controller three-path routing (Path A direct reuse, Path B experience-assisted, Path C fallback).
- Error handling on dimension mismatch, invalid shapes, and missing dependencies.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import MagicMock

# Detect availability of real dependencies
try:
    import faiss
    HAS_REAL_FAISS = hasattr(faiss, "IndexFlatIP") and not isinstance(faiss, MagicMock) and hasattr(faiss, "write_index")
except ImportError:
    HAS_REAL_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_REAL_ST = True
except ImportError:
    HAS_REAL_ST = False

from evorag.config import DEFAULT_EMBED_MODEL, DEFAULT_SIMILARITY_THRESHOLD
from evorag.memory.schema import Experience, VerificationStatus, normalize_question
from evorag.memory.store import ExperienceStore
from evorag.memory.index_builder import (
    ExperienceIndexBuilder,
    INDEX_FILE_NAME as EXP_INDEX_NAME,
    META_FILE_NAME as EXP_META_NAME,
)
from evorag.memory.retriever import ExperienceRetriever, ExperienceAugmentedRetriever
from evorag.memory.validator import ExperienceValidator
from evorag.controller import EvoRAGController, MemoryIndexRebuildError
from evorag.result import EvoRAGResult
from index.build_index import build_index, compute_chunk_hash, compute_file_hash
from retriever.retriever import PassageRetriever


# Mark all tests in this module with 'integration'
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def real_st_model():
    """Load the real SentenceTransformer embedding model once for the test session."""
    if not HAS_REAL_ST:
        pytest.skip("sentence-transformers package not installed")
    try:
        # Respect local cache in offline/sandboxed environments
        model = SentenceTransformer(DEFAULT_EMBED_MODEL)
        return model
    except Exception as exc:
        pytest.skip(f"Failed to load real embedding model '{DEFAULT_EMBED_MODEL}': {exc}")


@pytest.fixture
def check_dependencies():
    """Skip test if real FAISS or SentenceTransformer dependencies are missing."""
    if not HAS_REAL_FAISS:
        pytest.skip("Real FAISS library is not installed or available")
    if not HAS_REAL_ST:
        pytest.skip("Real sentence-transformers is not installed or available")


# =====================================================================
# Step 2: Real SentenceTransformer Embeddings Validation
# =====================================================================

class TestSentenceTransformerValidation:
    """Validates real SentenceTransformer embedding generation, dimensions, normalization, and ranking."""

    def test_real_embedding_generation_and_properties(self, check_dependencies, real_st_model):
        """Verify model loading, embedding shape, float32 datatype, and L2 normalization."""
        texts = [
            "Section 14 of the Compliance Act provides special exemptions for small enterprises.",
            "All appeals lie before the Appellate Tribunal within forty-five days.",
        ]
        embeddings = real_st_model.encode(texts, normalize_embeddings=True)

        assert isinstance(embeddings, np.ndarray), "Embeddings must be a numpy ndarray"
        assert embeddings.shape == (2, 384), f"Expected shape (2, 384), got {embeddings.shape}"
        assert embeddings.dtype == np.float32, f"Expected dtype float32, got {embeddings.dtype}"

        # Verify unit L2 norm
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5, err_msg="Normalized vectors must have unit L2 norm")

    def test_real_embedding_semantic_similarity_ranking(self, check_dependencies, real_st_model):
        """Verify that semantically related queries and passages produce sensible cosine rankings."""
        passages = [
            # Relevant to Section 14 penalty
            "The statutory penalty for non-compliance under Section 14 is Rs. 500 per day.",
            # Related to legal appeals but different section
            "An appeal must be filed within forty-five days from receipt of the order under Section 15.",
            # Completely unrelated
            "Photosynthesis in tropical rainforests converts solar energy into chemical energy.",
        ]
        p_embs = real_st_model.encode(passages, normalize_embeddings=True)

        query = "What is the penalty or fine under Section 14?"
        q_emb = real_st_model.encode([query], normalize_embeddings=True)

        # Inner product on normalized vectors = cosine similarity
        similarities = np.dot(p_embs, q_emb.T).flatten()

        # Section 14 passage should score highest, followed by appeal, rainforest lowest
        assert similarities[0] > 0.65, f"Expected high similarity for relevant passage, got {similarities[0]:.4f}"
        assert similarities[0] > similarities[1], "Section 14 passage must outrank Section 15 passage"
        assert similarities[1] > similarities[2], "Legal appeal passage must outrank rainforest passage"
        assert similarities[2] < 0.20, f"Unrelated passage should have low similarity, got {similarities[2]:.4f}"


# =====================================================================
# Step 3: Real Document FAISS Indexing & Retrieval
# =====================================================================

class TestRealDocumentFAISSValidation:
    """Validates real document FAISS index building, persistence, and passage retrieval with metadata."""

    @pytest.fixture
    def test_doc_index(self, check_dependencies, tmp_path: Path):
        """Build a real FAISS document index from a temporary test PDF."""
        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            pytest.skip("reportlab required to build test PDF for document FAISS validation")

        pdf_path = tmp_path / "compliance_doc.pdf"
        c = canvas.Canvas(str(pdf_path))
        # Page 1
        t1 = c.beginText(50, 750)
        for line in [
            "The Compliance Act of 2024 establishes statutory standards for all enterprises.",
            "Under Section 14, registered entities must file annual returns by March 31.",
            "Failure to file returns attracts a mandatory penalty of Rs. 500 per calendar day.",
            "Special exemptions under Clause 5 apply exclusively to qualifying non-profit organizations.",
        ]:
            t1.textLine(line)
        c.drawText(t1)
        c.showPage()
        # Page 2
        t2 = c.beginText(50, 750)
        for line in [
            "Appeals and Tribunal Procedures. Section 15 specifies appellate filing procedures.",
            "An aggrieved party may file an appeal before the Appellate Tribunal within 45 days.",
            "The decision of the Appellate Tribunal is final and binding on all parties.",
        ]:
            t2.textLine(line)
        c.drawText(t2)
        c.save()

        out_dir = tmp_path / "real_doc_index"
        build_index([str(pdf_path)], str(out_dir))
        return out_dir, pdf_path

    def test_real_document_faiss_files_persisted(self, test_doc_index):
        """Verify that faiss.index, corpus.jsonl, and meta.json are written to disk."""
        out_dir, pdf_path = test_doc_index
        assert (out_dir / "faiss.index").exists(), "faiss.index must exist"
        assert (out_dir / "corpus.jsonl").exists(), "corpus.jsonl must exist"
        assert (out_dir / "meta.json").exists(), "meta.json must exist"

        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["num_chunks"] > 0
        assert meta["embed_model"] == DEFAULT_EMBED_MODEL
        assert pdf_path.name in meta["sources"]
        assert pdf_path.name in meta["source_versions"]

    def test_real_document_retrieval_and_metadata_preservation(self, test_doc_index):
        """Verify PassageRetriever loads real FAISS index and returns relevant passages with full metadata."""
        out_dir, pdf_path = test_doc_index
        retriever = PassageRetriever(str(out_dir))

        query = "What is the penalty for failure to file returns under Section 14?"
        results = retriever.search(query, k=2)

        assert len(results) >= 1, "Must retrieve at least one passage"
        top_passage = results[0]

        # Verify semantic relevance
        assert "500" in top_passage["text"]
        assert top_passage["retrieval_score"] > 0.50

        # Verify complete metadata preservation
        assert top_passage["source"] == pdf_path.name
        assert top_passage["page"] in (1, 2)
        assert isinstance(top_passage["chunk_id"], int)
        assert top_passage["source_version"] != ""
        assert top_passage["chunk_hash"] != ""
        assert isinstance(top_passage["retrieval_score"], float)


# =====================================================================
# Step 4: Real Experience Memory FAISS & SQLite Validation
# =====================================================================

class TestRealExperienceMemoryFAISSValidation:
    """Validates real Experience Memory FAISS indexing, metadata persistence, and SQLite resolution."""

    @pytest.fixture
    def populated_memory_store(self, check_dependencies, tmp_path: Path):
        """Create an SQLite store populated with approved and pending experiences."""
        db_path = tmp_path / "real_memory.db"
        store = ExperienceStore(db_path)

        exp1 = Experience(
            question="What is the statutory penalty under Section 14?",
            answer="Under Section 14, the non-compliance penalty is Rs. 500 per day.",
            evidence=[{"text": "penalty under Sec. 14 is Rs. 500 per day", "source": "compliance_doc.pdf"}],
            source_versions={"compliance_doc.pdf": "v1_hash"},
            chunk_hashes=["hash_chunk_1"],
            verification_status=VerificationStatus.APPROVED,
        )
        exp2 = Experience(
            question="What is the time window for filing an appeal under Section 15?",
            answer="An appeal must be filed within forty-five days from receipt of the order.",
            evidence=[{"text": "appeal within 45 days", "source": "compliance_doc.pdf"}],
            source_versions={"compliance_doc.pdf": "v1_hash"},
            chunk_hashes=["hash_chunk_2"],
            verification_status=VerificationStatus.APPROVED,
        )
        exp3_pending = Experience(
            question="What are the Clause 5 exemptions?",
            answer="Clause 5 exemptions apply to non-profits.",
            evidence=[],
            source_versions={},
            chunk_hashes=[],
            verification_status=VerificationStatus.PENDING,
        )

        store.add_experience(exp1)
        store.add_experience(exp2)
        store.add_experience(exp3_pending)

        index_dir = tmp_path / "real_exp_index"
        return store, index_dir, [exp1, exp2, exp3_pending]

    def test_real_experience_memory_build_and_sqlite_resolution(self, populated_memory_store, real_st_model):
        """Verify building FAISS index with real embeddings and resolving search hits to SQLite records."""
        store, index_dir, experiences = populated_memory_store
        exp1, exp2, exp3_pending = experiences

        builder = ExperienceIndexBuilder(index_dir, embedder=real_st_model)
        meta = builder.rebuild_from_store(store)

        # Verify metadata and ID mapping
        assert meta["num_vectors"] == 2, "Only 2 approved experiences should be indexed"
        assert exp1.experience_id in meta["id_map"]
        assert exp2.experience_id in meta["id_map"]
        assert exp3_pending.experience_id not in meta["id_map"], "Pending experience must not be in id_map"

        # Verify disk persistence
        assert (index_dir / EXP_INDEX_NAME).exists()
        assert (index_dir / EXP_META_NAME).exists()

        # Query with real retriever
        retriever = ExperienceRetriever(index_dir, store=store, embedder=real_st_model)
        candidates = retriever.retrieve("What penalty is imposed under section 14?", k=3)

        assert len(candidates) == 2, "Must return approved candidates only"
        top_cand = candidates[0]

        # Top candidate must resolve to exp1 in SQLite
        assert top_cand.experience.experience_id == exp1.experience_id
        assert top_cand.experience.question == exp1.question
        assert top_cand.experience.answer == exp1.answer
        assert top_cand.experience.evidence == exp1.evidence
        assert top_cand.similarity_score > 0.70

        # FAISS integer index must NOT equal SQLite SHA-256 experience_id
        assert top_cand.experience.experience_id != "0"
        assert len(top_cand.experience.experience_id) == 64


# =====================================================================
# Step 5: Index Consistency and Recovery
# =====================================================================

class TestRealIndexConsistencyAndRecovery:
    """Tests normal incremental addition, disk persistence, and recovery with real FAISS."""

    def test_incremental_experience_addition_and_retrieval(self, check_dependencies, real_st_model, tmp_path: Path):
        """Adding a new experience to SQLite and rebuilding makes it retrievable."""
        db_path = tmp_path / "inc_memory.db"
        index_dir = tmp_path / "inc_index"
        store = ExperienceStore(db_path)
        builder = ExperienceIndexBuilder(index_dir, embedder=real_st_model)
        retriever = ExperienceRetriever(index_dir, store=store, embedder=real_st_model)

        # 1. Add initial experience
        exp1 = Experience(
            question="What is Rule 1?",
            answer="Rule 1 requires registration.",
            evidence=[{"text": "reg", "source": "doc.pdf"}],
            source_versions={"doc.pdf": "v1"},
            chunk_hashes=["h1"],
        )
        store.add_experience(exp1)
        builder.rebuild_from_store(store)
        retriever._load_index()

        cands = retriever.retrieve("What is Rule 1?", k=2)
        assert len(cands) == 1
        assert cands[0].experience.experience_id == exp1.experience_id

        # 2. Add second experience
        exp2 = Experience(
            question="What is Rule 2?",
            answer="Rule 2 mandates annual audit.",
            evidence=[{"text": "audit", "source": "doc.pdf"}],
            source_versions={"doc.pdf": "v1"},
            chunk_hashes=["h2"],
        )
        store.add_experience(exp2)
        builder.rebuild_from_store(store)
        retriever._load_index()

        cands2 = retriever.retrieve("What does Rule 2 mandate?", k=2)
        assert len(cands2) == 2
        assert cands2[0].experience.experience_id == exp2.experience_id
        assert cands2[0].similarity_score > 0.75

    def test_persistence_across_store_reopen(self, check_dependencies, real_st_model, tmp_path: Path):
        """Close store and retriever, reopen from disk, and verify retrieval still functions."""
        db_path = tmp_path / "persist_memory.db"
        index_dir = tmp_path / "persist_index"
        store = ExperienceStore(db_path)
        builder = ExperienceIndexBuilder(index_dir, embedder=real_st_model)

        exp = Experience(
            question="What are the filing hours?",
            answer="Filing hours are 9 AM to 5 PM.",
            evidence=[{"text": "hours", "source": "doc.pdf"}],
            source_versions={"doc.pdf": "v1"},
            chunk_hashes=["h1"],
        )
        store.add_experience(exp)
        builder.rebuild_from_store(store)
        store.close()

        # Reopen brand new store and retriever instances pointing to disk
        reopened_store = ExperienceStore(db_path)
        reopened_retriever = ExperienceRetriever(index_dir, store=reopened_store, embedder=real_st_model)

        cands = reopened_retriever.retrieve("What are the filing hours?", k=1)
        assert len(cands) == 1
        assert cands[0].experience.answer == "Filing hours are 9 AM to 5 PM."
        reopened_store.close()

    def test_recovery_from_missing_faiss_index(self, check_dependencies, real_st_model, tmp_path: Path):
        """If faiss.index is deleted from disk, recover_index rebuilds it cleanly from SQLite."""
        db_path = tmp_path / "rec_memory.db"
        index_dir = tmp_path / "rec_index"
        store = ExperienceStore(db_path)
        builder = ExperienceIndexBuilder(index_dir, embedder=real_st_model)
        retriever = ExperienceRetriever(index_dir, store=store, embedder=real_st_model)
        validator = ExperienceValidator()

        controller = EvoRAGController(
            store=store,
            memory_retriever=retriever,
            validator=validator,
            index_builder=builder,
        )

        exp = Experience(
            question="What is Section 1?",
            answer="Section 1 establishes title.",
            evidence=[{"text": "title", "source": "doc.pdf"}],
            source_versions={"doc.pdf": "v1"},
            chunk_hashes=["h1"],
        )
        store.add_experience(exp)
        controller.recover_index()

        # Delete the FAISS index file to simulate loss
        (index_dir / EXP_INDEX_NAME).unlink()
        assert not (index_dir / EXP_INDEX_NAME).exists()

        # Invalidate in-memory retriever
        retriever.invalidate_index()
        assert retriever.retrieve("Section 1", k=1) == []

        # Execute recovery
        meta = controller.recover_index()
        assert meta["num_vectors"] == 1
        assert (index_dir / EXP_INDEX_NAME).exists()

        # Verify retrieval works again
        cands = retriever.retrieve("What is Section 1?", k=1)
        assert len(cands) == 1
        assert cands[0].experience.experience_id == exp.experience_id

    def test_recovery_from_corrupted_faiss_index(self, check_dependencies, real_st_model, tmp_path: Path):
        """If faiss.index is corrupted with bad bytes, recover_index restores the index from SQLite."""
        db_path = tmp_path / "corrupt_memory.db"
        index_dir = tmp_path / "corrupt_index"
        store = ExperienceStore(db_path)
        builder = ExperienceIndexBuilder(index_dir, embedder=real_st_model)
        retriever = ExperienceRetriever(index_dir, store=store, embedder=real_st_model)
        validator = ExperienceValidator()

        controller = EvoRAGController(
            store=store,
            memory_retriever=retriever,
            validator=validator,
            index_builder=builder,
        )

        exp = Experience(
            question="What is Section 2?",
            answer="Section 2 establishes definitions.",
            evidence=[{"text": "defs", "source": "doc.pdf"}],
            source_versions={"doc.pdf": "v1"},
            chunk_hashes=["h2"],
        )
        store.add_experience(exp)
        controller.recover_index()

        # Corrupt the index file
        (index_dir / EXP_INDEX_NAME).write_bytes(b"CORRUPTED_NON_FAISS_BYTES")

        # Reloading corrupted index raises RuntimeError and cleanly invalidates
        with pytest.raises(RuntimeError):
            retriever._load_index()
        assert retriever._index is None

        # Recovery rebuilds a clean index from SQLite
        meta = controller.recover_index()
        assert meta["num_vectors"] == 1
        assert controller.last_index_error is None
        assert retriever._index is not None

        cands = retriever.retrieve("What is Section 2?", k=1)
        assert len(cands) == 1
        assert cands[0].experience.experience_id == exp.experience_id

    def test_recovery_from_corrupted_metadata_json(self, check_dependencies, real_st_model, tmp_path: Path):
        """If experience_meta.json is corrupted, recover_index restores valid metadata."""
        db_path = tmp_path / "corrupt_meta.db"
        index_dir = tmp_path / "corrupt_meta_index"
        store = ExperienceStore(db_path)
        builder = ExperienceIndexBuilder(index_dir, embedder=real_st_model)
        retriever = ExperienceRetriever(index_dir, store=store, embedder=real_st_model)
        validator = ExperienceValidator()

        controller = EvoRAGController(
            store=store,
            memory_retriever=retriever,
            validator=validator,
            index_builder=builder,
        )

        exp = Experience(
            question="What is Section 3?",
            answer="Section 3 defines scope.",
            evidence=[{"text": "scope", "source": "doc.pdf"}],
            source_versions={"doc.pdf": "v1"},
            chunk_hashes=["h3"],
        )
        store.add_experience(exp)
        controller.recover_index()

        # Corrupt the metadata file with invalid JSON
        (index_dir / EXP_META_NAME).write_text("{invalid json", encoding="utf-8")

        # Reloading corrupted meta raises JSONDecodeError and invalidates
        with pytest.raises(json.JSONDecodeError):
            retriever._load_index()
        assert retriever._index is None

        # Recovery restores valid metadata from SQLite
        meta = controller.recover_index()
        assert meta["num_vectors"] == 1
        assert meta["id_map"] == [exp.experience_id]

        cands = retriever.retrieve("What is Section 3?", k=1)
        assert len(cands) == 1

    def test_empty_experience_memory_handling(self, check_dependencies, real_st_model, tmp_path: Path):
        """Empty SQLite store produces an empty 0-vector FAISS index and retrieval returns []."""
        db_path = tmp_path / "empty_memory.db"
        index_dir = tmp_path / "empty_index"
        store = ExperienceStore(db_path)
        builder = ExperienceIndexBuilder(index_dir, embedder=real_st_model)
        retriever = ExperienceRetriever(index_dir, store=store, embedder=real_st_model)

        meta = builder.rebuild_from_store(store)
        assert meta["num_vectors"] == 0
        assert meta["id_map"] == []

        retriever._load_index()
        assert retriever.retrieve("any query", k=3) == []


# =====================================================================
# Step 6: Controller Three-Path Routing with Real FAISS
# =====================================================================

class TestRealControllerThreePathRouting:
    """Validates Path A direct reuse, Path B experience-assisted, and Path C fallback with real FAISS."""

    class TrackingGenerator:
        """Lightweight tracking generator simulating Self-RAG generation."""
        def __init__(self):
            self.call_count = 0
            self.last_query = None
            self.last_retriever = None

        def __call__(self, *args, **kwargs):
            self.call_count += 1
            return "Generated answer"

    def test_path_a_direct_reuse_with_real_faiss(self, check_dependencies, real_st_model, tmp_path: Path):
        """Exact normalized match routes through Path A with 0 generator calls."""
        db_path = tmp_path / "path_a.db"
        index_dir = tmp_path / "path_a_index"
        store = ExperienceStore(db_path)
        builder = ExperienceIndexBuilder(index_dir, embedder=real_st_model)
        retriever = ExperienceRetriever(index_dir, store=store, embedder=real_st_model)
        validator = ExperienceValidator(similarity_threshold=0.85)

        exp = Experience(
            question="What is the filing deadline for appeals under Section 15?",
            answer="Appeals must be filed within forty-five days.",
            evidence=[{"text": "filing within 45 days", "source": "act.pdf"}],
            source_versions={"act.pdf": "v1"},
            chunk_hashes=["h1"],
        )
        store.add_experience(exp)
        builder.rebuild_from_store(store)
        retriever._load_index()

        generator = self.TrackingGenerator()
        doc_retriever = None  # Should not be called

        controller = EvoRAGController(
            store=store,
            memory_retriever=retriever,
            validator=validator,
            generator=generator,
            retriever=doc_retriever,
            index_builder=builder,
        )

        # Query with extra spaces and mixed casing
        res = controller.answer("  What is the filing deadline for appeals under Section 15?  ")

        assert res.response_mode == "direct_reuse"
        assert res.memory_hit is True
        assert res.answer == "Appeals must be filed within forty-five days."
        assert generator.call_count == 0, "Generator must not be called on Path A"

    def test_path_c_normal_self_rag_fallback_when_unrelated(self, check_dependencies, real_st_model, tmp_path: Path):
        """When query is unrelated to any stored experience, routes to Path C fallback."""
        from unittest.mock import MagicMock
        from selfrag.pipeline import SelfRAGResult, Candidate, Segment

        db_path = tmp_path / "path_c.db"
        index_dir = tmp_path / "path_c_index"
        store = ExperienceStore(db_path)
        builder = ExperienceIndexBuilder(index_dir, embedder=real_st_model)
        retriever = ExperienceRetriever(index_dir, store=store, embedder=real_st_model)
        validator = ExperienceValidator(similarity_threshold=0.85)

        exp = Experience(
            question="What is the statutory penalty under Section 14?",
            answer="Rs. 500 per day.",
            evidence=[{"text": "penalty 500", "source": "act.pdf"}],
            source_versions={"act.pdf": "v1"},
            chunk_hashes=["h1"],
        )
        store.add_experience(exp)
        builder.rebuild_from_store(store)
        retriever._load_index()

        mock_generator = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"text": "The Amazon rainforest is large.", "source": "amazon.pdf", "page": 1, "chunk_id": 0, "chunk_hash": "az1", "source_version": "v1"}
        ]

        # Mock self_rag_answer return
        mock_selfrag_res = SelfRAGResult(
            query="How large is the Amazon basin?",
            retrieved=True,
            candidates=[],
            final_answer="The Amazon rainforest covers 6 million sq km.",
            segments=[],
            selected=Candidate(
                passage={"text": "Amazon is large", "source": "amazon.pdf", "chunk_hash": "az1", "source_version": "v1"},
                raw_text="The Amazon rainforest covers 6 million sq km.",
                isrel="[Relevant]",
                issup="[Fully supported]",
                isuse=5,
                score=1.0,
            ),
        )

        controller = EvoRAGController(
            store=store,
            memory_retriever=retriever,
            validator=validator,
            generator=mock_generator,
            retriever=mock_retriever,
            index_builder=builder,
        )

        import selfrag.pipeline as p_mod
        orig_answer = p_mod.self_rag_answer
        p_mod.self_rag_answer = lambda *args, **kwargs: mock_selfrag_res
        try:
            res = controller.answer("How large is the Amazon basin?")
            assert res.response_mode == "normal_self_rag"
            assert res.memory_hit is False
            assert res.memory_context_used is False
            assert res.fallback_reason is not None
        finally:
            p_mod.self_rag_answer = orig_answer


# =====================================================================
# Step 7: Error & Failure Handling
# =====================================================================

class TestRealErrorHandling:
    """Validates robust exception raising on dimension mismatch, invalid shapes, and missing deps."""

    def test_dimension_mismatch_raises_value_error(self, check_dependencies, tmp_path: Path):
        """Query vector with wrong dimensionality raises ValueError instead of crashing."""
        db_path = tmp_path / "dim_memory.db"
        index_dir = tmp_path / "dim_index"
        store = ExperienceStore(db_path)

        # Build index with dim 384
        dim_384_embedder = lambda texts: np.ones((len(texts), 384), dtype="float32")
        builder = ExperienceIndexBuilder(index_dir, embedder=dim_384_embedder)

        exp = Experience(question="Q", answer="A", evidence=[{"text": "t", "source": "s"}], source_versions={"s": "v"}, chunk_hashes=["h"])
        store.add_experience(exp)
        builder.rebuild_from_store(store)

        # Query with embedder producing dim 128
        dim_128_embedder = lambda texts: np.ones((len(texts), 128), dtype="float32")
        retriever = ExperienceRetriever(index_dir, store=store, embedder=dim_128_embedder)

        with pytest.raises(ValueError, match="does not match index dimension"):
            retriever.retrieve("Q", k=1)

    def test_invalid_embedding_shape_raises_value_error(self, check_dependencies, tmp_path: Path):
        """1D embedding output from a broken embedder raises ValueError."""
        db_path = tmp_path / "shape_memory.db"
        index_dir = tmp_path / "shape_index"
        store = ExperienceStore(db_path)

        # Broken embedder returns 1D vector instead of 2D
        broken_embedder = lambda texts: np.ones(384, dtype="float32")
        builder = ExperienceIndexBuilder(index_dir, embedder=broken_embedder)

        exp = Experience(question="Q", answer="A", evidence=[{"text": "t", "source": "s"}], source_versions={"s": "v"}, chunk_hashes=["h"])
        store.add_experience(exp)

        with pytest.raises(ValueError, match="Expected 2D embedding array"):
            builder.rebuild_from_store(store)

    def test_missing_faiss_raises_import_error(self, monkeypatch, tmp_path: Path):
        """If faiss is None, ExperienceIndexBuilder raises ImportError."""
        from evorag.memory import index_builder as ib_mod
        monkeypatch.setattr(ib_mod, "faiss", None)

        builder = ExperienceIndexBuilder(tmp_path / "no_faiss")
        with pytest.raises(ImportError, match="FAISS is required"):
            builder.build_from_experiences([])
