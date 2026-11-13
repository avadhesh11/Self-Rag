import hashlib
import json
from pathlib import Path
import sys
from typing import Any
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


# Ensure faiss in sys.modules has functional MockIndexFlatIP, write_index, and read_index
faiss_mod = sys.modules.get("faiss")
if faiss_mod is None or isinstance(faiss_mod, MagicMock) or not hasattr(faiss_mod, "IndexFlatIP"):
    if faiss_mod is None:
        faiss_mod = MagicMock()
        sys.modules["faiss"] = faiss_mod
    faiss_mod.IndexFlatIP = MockIndexFlatIP
    faiss_mod.write_index = _mock_write_index
    faiss_mod.read_index = _mock_read_index

from evorag.config import ExperienceMemoryConfig
from evorag.memory.schema import (
    Experience,
    VerificationStatus,
    generate_experience_id,
    normalize_question,
)
from evorag.memory.store import ExperienceStore
from evorag.memory.index_builder import ExperienceIndexBuilder
from evorag.memory.retriever import ExperienceRetriever
from evorag.memory.validator import ExperienceValidator
from evorag.result import ExperienceCandidate


@pytest.fixture
def test_embedder():
    return DeterministicTestEmbedder(dim=384)


@pytest.fixture
def sample_experience():
    return Experience(
        question="What are the reflection tokens in Self-RAG?",
        answer="The reflection tokens are Retrieve, ISREL, ISSUP, and ISUSE.",
        evidence=[
            {
                "text": "Self-RAG uses four types of reflection tokens...",
                "source": "selfrag_paper.pdf",
                "page": 3,
                "chunk_hash": "c1a2b3",
            }
        ],
        source_versions={"selfrag_paper.pdf": "v1_hash_abc"},
        chunk_hashes=["c1a2b3"],
        verification_status=VerificationStatus.APPROVED,
    )


class TestExperienceStore:
    """Tests for SQLite persistence, parameterized queries, and CRUD operations."""

    def test_sqlite_persistence_across_reopening(self, tmp_path: Path, sample_experience: Experience):
        """Verify data persists when closing and reopening the SQLite database connection."""
        db_path = tmp_path / "test_memory.db"

        # Open and insert
        with ExperienceStore(db_path) as store:
            exp_id = store.add_experience(sample_experience)
            assert exp_id == sample_experience.experience_id

        # Reopen connection and verify
        with ExperienceStore(db_path) as store:
            loaded = store.get_experience(exp_id)
            assert loaded is not None
            assert loaded.experience_id == sample_experience.experience_id
            assert loaded.question == sample_experience.question
            assert loaded.normalized_question == sample_experience.normalized_question
            assert loaded.answer == sample_experience.answer
            assert loaded.evidence == sample_experience.evidence
            assert loaded.source_versions == sample_experience.source_versions
            assert loaded.chunk_hashes == sample_experience.chunk_hashes
            assert loaded.verification_status == VerificationStatus.APPROVED
            assert loaded.created_at == sample_experience.created_at

    def test_duplicate_experience_rejection(self, tmp_path: Path, sample_experience: Experience):
        """Verify duplicate experience_id insertions raise ValueError."""
        db_path = tmp_path / "test_memory.db"
        with ExperienceStore(db_path) as store:
            store.add_experience(sample_experience)
            with pytest.raises(ValueError, match="already exists"):
                store.add_experience(sample_experience)

    def test_parameterized_query_safety(self, tmp_path: Path):
        """Verify strings containing SQL injection payloads are stored and retrieved safely."""
        db_path = tmp_path / "test_memory.db"
        injection_text = "What is Self-RAG?'; DROP TABLE experiences; --"
        malicious_exp = Experience(
            question=injection_text,
            answer="Safe answer'; DELETE FROM experiences; --",
            evidence=[{"text": "injection test ' OR '1'='1"}],
            source_versions={"doc.pdf": "hash'; --"},
            chunk_hashes=["hash1"],
        )

        with ExperienceStore(db_path) as store:
            store.add_experience(malicious_exp)
            loaded = store.get_experience(malicious_exp.experience_id)
            assert loaded is not None
            assert loaded.question == injection_text
            # Verify table still exists and row count is 1
            assert store.count() == 1

    def test_verification_status_filtering(self, tmp_path: Path):
        """Verify listing and counting experiences respects verification status filter."""
        db_path = tmp_path / "test_memory.db"
        with ExperienceStore(db_path) as store:
            e1 = Experience(question="Q1", answer="A1", verification_status=VerificationStatus.APPROVED)
            e2 = Experience(question="Q2", answer="A2", verification_status=VerificationStatus.PENDING)
            e3 = Experience(question="Q3", answer="A3", verification_status=VerificationStatus.REJECTED)

            store.add_experience(e1)
            store.add_experience(e2)
            store.add_experience(e3)

            assert store.count() == 3
            assert store.count(status=VerificationStatus.APPROVED) == 1
            assert store.count(status=VerificationStatus.PENDING) == 1
            assert store.count(status=VerificationStatus.REJECTED) == 1

            approved_list = store.list_experiences(status=VerificationStatus.APPROVED)
            assert len(approved_list) == 1
            assert approved_list[0].question == "Q1"

            all_list = store.list_experiences()
            assert len(all_list) == 3

    def test_update_and_deletion(self, tmp_path: Path, sample_experience: Experience):
        """Verify updating verification status and deleting experiences."""
        db_path = tmp_path / "test_memory.db"
        with ExperienceStore(db_path) as store:
            sample_experience.verification_status = VerificationStatus.PENDING
            store.add_experience(sample_experience)

            # Update status
            success = store.update_verification_status(sample_experience.experience_id, VerificationStatus.APPROVED)
            assert success is True
            updated = store.get_experience(sample_experience.experience_id)
            assert updated.verification_status == VerificationStatus.APPROVED

            # Delete
            del_success = store.delete_experience(sample_experience.experience_id)
            assert del_success is True
            assert store.get_experience(sample_experience.experience_id) is None
            assert store.count() == 0


class TestExperienceIndexingAndRetrieval:
    """Tests for FAISS index building, ID mapping, semantic retrieval, and rebuilding."""

    def test_empty_faiss_index_handling(self, tmp_path: Path, test_embedder):
        """Verify building and querying an empty FAISS index operates safely."""
        index_dir = tmp_path / "empty_index"
        db_path = tmp_path / "empty.db"

        builder = ExperienceIndexBuilder(index_dir=index_dir, embedder=test_embedder)
        meta = builder.build_from_experiences([])

        assert meta["num_vectors"] == 0
        assert meta["id_map"] == []

        with ExperienceStore(db_path) as store:
            retriever = ExperienceRetriever(index_dir=index_dir, store=store, embedder=test_embedder)
            candidates = retriever.retrieve("Any question?")
            assert candidates == []

    def test_semantic_retrieval_and_id_mapping(self, tmp_path: Path, test_embedder):
        """Verify semantic retrieval ranks closest question and resolves experience via ID map."""
        db_path = tmp_path / "test.db"
        index_dir = tmp_path / "test_index"

        with ExperienceStore(db_path) as store:
            e1 = Experience(
                question="How does Self-RAG decide whether to retrieve external passages?",
                answer="Self-RAG uses a retrieve decision token evaluated during initial generation.",
                evidence=[{"text": "evidence 1"}],
                source_versions={"doc.pdf": "v1"},
                verification_status=VerificationStatus.APPROVED,
            )
            e2 = Experience(
                question="What is the square root of 144?",
                answer="12",
                evidence=[{"text": "evidence 2"}],
                source_versions={"math.pdf": "v1"},
                verification_status=VerificationStatus.APPROVED,
            )
            store.add_experience(e1)
            store.add_experience(e2)

            builder = ExperienceIndexBuilder(index_dir=index_dir, embedder=test_embedder)
            builder.build_from_experiences([e1, e2])

            retriever = ExperienceRetriever(index_dir=index_dir, store=store, embedder=test_embedder)

            # Query identical to e1 should return e1 with high similarity (~1.0)
            hits = retriever.retrieve("How does Self-RAG decide whether to retrieve external passages?", k=2)
            assert len(hits) == 2
            top_hit = hits[0]
            assert top_hit.experience.experience_id == e1.experience_id
            assert top_hit.similarity_score == pytest.approx(1.0, abs=1e-3)

    def test_unapproved_experiences_excluded_from_retrieval(self, tmp_path: Path, test_embedder):
        """Verify unapproved experiences are not indexed or returned by retriever."""
        db_path = tmp_path / "test.db"
        index_dir = tmp_path / "test_index"

        with ExperienceStore(db_path) as store:
            e_approved = Experience(
                question="Approved question",
                answer="Approved answer",
                evidence=[{"text": "ev"}],
                source_versions={"d.pdf": "v1"},
                verification_status=VerificationStatus.APPROVED,
            )
            e_pending = Experience(
                question="Pending question",
                answer="Pending answer",
                evidence=[{"text": "ev"}],
                source_versions={"d.pdf": "v1"},
                verification_status=VerificationStatus.PENDING,
            )
            store.add_experience(e_approved)
            store.add_experience(e_pending)

            builder = ExperienceIndexBuilder(index_dir=index_dir, embedder=test_embedder)
            builder.rebuild_from_store(store, status=VerificationStatus.APPROVED)

            retriever = ExperienceRetriever(index_dir=index_dir, store=store, embedder=test_embedder)
            results = retriever.retrieve("Pending question", k=5)

            # Should only contain approved experience
            returned_ids = [c.experience.experience_id for c in results]
            assert e_approved.experience_id in returned_ids
            assert e_pending.experience_id not in returned_ids

    def test_faiss_mapping_persistence_and_rebuild(self, tmp_path: Path, test_embedder):
        """Verify reloading persisted index and rebuilding from SQLite store."""
        db_path = tmp_path / "test.db"
        index_dir = tmp_path / "test_index"

        with ExperienceStore(db_path) as store:
            e1 = Experience(question="Initial Q", answer="Initial A", evidence=[{"t": 1}], source_versions={"d": "1"})
            store.add_experience(e1)

            builder = ExperienceIndexBuilder(index_dir=index_dir, embedder=test_embedder)
            builder.rebuild_from_store(store)

            # Reopen retriever
            retriever1 = ExperienceRetriever(index_dir=index_dir, store=store, embedder=test_embedder)
            assert len(retriever1._id_map) == 1
            assert retriever1._id_map[0] == e1.experience_id

            # Add second experience and rebuild
            e2 = Experience(question="Second Q", answer="Second A", evidence=[{"t": 2}], source_versions={"d": "1"})
            store.add_experience(e2)
            builder.rebuild_from_store(store)

            # Reopen and check updated index
            retriever2 = ExperienceRetriever(index_dir=index_dir, store=store, embedder=test_embedder)
            assert len(retriever2._id_map) == 2
            assert e2.experience_id in retriever2._id_map


class TestExperienceValidator:
    """Tests for conservative validation criteria and source version invalidation."""

    def test_validation_passes_for_clean_candidate(self, sample_experience: Experience):
        """Verify approved experience with valid evidence, answer, and high similarity passes."""
        validator = ExperienceValidator(similarity_threshold=0.85)
        candidate = ExperienceCandidate(experience=sample_experience, similarity_score=0.92)
        res = validator.validate(candidate, current_source_versions={"selfrag_paper.pdf": "v1_hash_abc"})

        assert res.is_valid is True
        assert "passed all validation checks" in res.reason

    def test_rejection_low_similarity(self, sample_experience: Experience):
        """Verify candidate with similarity score below threshold is rejected."""
        validator = ExperienceValidator(similarity_threshold=0.85)
        candidate = ExperienceCandidate(experience=sample_experience, similarity_score=0.74)
        res = validator.validate(candidate)

        assert res.is_valid is False
        assert "below required threshold" in res.reason

    def test_rejection_unapproved_status(self, sample_experience: Experience):
        """Verify candidate with pending or rejected verification status is rejected."""
        validator = ExperienceValidator(similarity_threshold=0.80)
        sample_experience.verification_status = VerificationStatus.PENDING
        candidate = ExperienceCandidate(experience=sample_experience, similarity_score=0.95)
        res = validator.validate(candidate)

        assert res.is_valid is False
        assert "not approved" in res.reason

    def test_rejection_empty_answer(self, sample_experience: Experience):
        """Verify experience with empty or whitespace answer is rejected."""
        validator = ExperienceValidator(similarity_threshold=0.80)
        sample_experience.answer = "   "
        candidate = ExperienceCandidate(experience=sample_experience, similarity_score=0.95)
        res = validator.validate(candidate)

        assert res.is_valid is False
        assert "empty or whitespace answer" in res.reason

    def test_rejection_missing_evidence_or_source_versions(self, sample_experience: Experience):
        """Verify experience without evidence or source version metadata is rejected."""
        validator = ExperienceValidator(similarity_threshold=0.80)

        # Missing evidence
        sample_experience.evidence = []
        candidate1 = ExperienceCandidate(experience=sample_experience, similarity_score=0.95)
        res1 = validator.validate(candidate1)
        assert res1.is_valid is False
        assert "lacks supporting evidence" in res1.reason

        # Missing source versions
        sample_experience.evidence = [{"text": "passage"}]
        sample_experience.source_versions = {}
        candidate2 = ExperienceCandidate(experience=sample_experience, similarity_score=0.95)
        res2 = validator.validate(candidate2)
        assert res2.is_valid is False
        assert "lacks source version" in res2.reason

    def test_source_version_invalidation(self, sample_experience: Experience):
        """Verify candidate is invalidated when a source file's version has changed."""
        validator = ExperienceValidator(similarity_threshold=0.85)
        candidate = ExperienceCandidate(experience=sample_experience, similarity_score=0.95)

        # Matching source version -> Valid
        res_valid = validator.validate(
            candidate,
            current_source_versions={"selfrag_paper.pdf": "v1_hash_abc"},
        )
        assert res_valid.is_valid is True

        # Modified source version in index -> Invalidated
        res_invalid = validator.validate(
            candidate,
            current_source_versions={"selfrag_paper.pdf": "v2_modified_hash_xyz"},
        )
        assert res_invalid.is_valid is False
        assert "invalidated by modified source document" in res_invalid.reason

    def test_validate_with_doc_index_directory(self, tmp_path: Path, sample_experience: Experience):
        """Verify validate_with_doc_index reads meta.json from document index directory."""
        doc_index_dir = tmp_path / "doc_index"
        doc_index_dir.mkdir(parents=True, exist_ok=True)

        # Create meta.json with modified source version
        meta_data = {
            "sources": ["selfrag_paper.pdf"],
            "source_versions": {"selfrag_paper.pdf": "different_version_hash"},
        }
        (doc_index_dir / "meta.json").write_text(json.dumps(meta_data), encoding="utf-8")

        validator = ExperienceValidator(similarity_threshold=0.85)
        candidate = ExperienceCandidate(experience=sample_experience, similarity_score=0.95)
        res = validator.validate_with_doc_index(candidate, doc_index_dir)

        assert res.is_valid is False
        assert "invalidated by modified source document" in res.reason
