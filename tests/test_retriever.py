import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock heavy optional dependencies if not installed
for mod in ["faiss", "sentence_transformers"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from retriever.retriever import PassageRetriever
from selfrag.pipeline import _retrieve_new


class TestPassageRetriever:
    """Tests verifying PassageRetriever loads metadata, corpus, and returns provenance fields."""

    @pytest.fixture
    def mock_index_dir(self, tmp_path: Path):
        """Create a mock index directory containing meta.json, corpus.jsonl, and a dummy faiss.index."""
        index_dir = tmp_path / "test_index"
        index_dir.mkdir(parents=True, exist_ok=True)

        meta_data = {
            "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
            "num_chunks": 2,
            "target_words": 120,
            "overlap_words": 30,
            "sources": ["sample_doc.pdf"],
            "source_versions": {
                "sample_doc.pdf": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            },
        }
        (index_dir / "meta.json").write_text(json.dumps(meta_data, indent=2))

        corpus_records = [
            {
                "text": "Self-RAG is a retrieval-augmented generation framework.",
                "page": 1,
                "source": "sample_doc.pdf",
                "source_version": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "chunk_hash": "a1b2c3d4e5f60718293a4b5c6d7e8f901234567890abcdef1234567890abcdef",
                "chunk_id": 0,
            },
            {
                "text": "It uses reflection tokens to critique generated responses.",
                "page": 2,
                "source": "sample_doc.pdf",
                "source_version": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "chunk_hash": "b2c3d4e5f6a10718293a4b5c6d7e8f901234567890abcdef1234567890abcdef",
                "chunk_id": 1,
            },
        ]
        with open(index_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
            for rec in corpus_records:
                f.write(json.dumps(rec) + "\n")

        # Create dummy faiss.index file
        (index_dir / "faiss.index").write_bytes(b"FAISS_DUMMY_BINARY")

        return index_dir

    def test_init_loads_metadata_and_corpus(self, mock_index_dir: Path, monkeypatch):
        """Verify PassageRetriever initialization correctly parses metadata and corpus records."""
        from retriever import retriever as r_mod

        # Mock SentenceTransformer and faiss.read_index
        mock_model = MagicMock()
        monkeypatch.setattr(r_mod, "SentenceTransformer", lambda model_name: mock_model)
        mock_index = MagicMock()
        monkeypatch.setattr(r_mod.faiss, "read_index", lambda path: mock_index)

        retriever = PassageRetriever(str(mock_index_dir))

        assert retriever.embed_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert len(retriever.corpus) == 2
        assert retriever.corpus[0]["source"] == "sample_doc.pdf"
        assert retriever.corpus[0]["chunk_hash"] == "a1b2c3d4e5f60718293a4b5c6d7e8f901234567890abcdef1234567890abcdef"
        assert retriever.corpus[0]["source_version"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_search_returns_provenance_and_score(self, mock_index_dir: Path, monkeypatch):
        """Verify search() returns all chunk fields including chunk_hash, source_version, and retrieval_score."""
        from retriever import retriever as r_mod

        # Mock model.encode to return dummy embeddings
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2]]
        monkeypatch.setattr(r_mod, "SentenceTransformer", lambda model_name: mock_model)

        # Mock faiss index.search: returns scores and indices
        mock_index = MagicMock()
        mock_index.search.return_value = ([[0.88, 0.72]], [[0, 1]])
        monkeypatch.setattr(r_mod.faiss, "read_index", lambda path: mock_index)

        retriever = PassageRetriever(str(mock_index_dir))
        results = retriever.search("What is Self-RAG?", k=2)

        assert len(results) == 2

        # First result checks
        r0 = results[0]
        assert r0["chunk_id"] == 0
        assert r0["page"] == 1
        assert r0["source"] == "sample_doc.pdf"
        assert r0["source_version"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert r0["chunk_hash"] == "a1b2c3d4e5f60718293a4b5c6d7e8f901234567890abcdef1234567890abcdef"
        assert r0["retrieval_score"] == pytest.approx(0.88)
        assert "Self-RAG is a retrieval-augmented generation framework." in r0["text"]

        # Second result checks
        r1 = results[1]
        assert r1["chunk_id"] == 1
        assert r1["chunk_hash"] == "b2c3d4e5f6a10718293a4b5c6d7e8f901234567890abcdef1234567890abcdef"
        assert r1["retrieval_score"] == pytest.approx(0.72)

    def test_pipeline_retrieve_new_compatibility(self, mock_index_dir: Path, monkeypatch):
        """Verify _retrieve_new in selfrag/pipeline works with retrieved passages containing provenance fields."""
        from retriever import retriever as r_mod

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2]]
        monkeypatch.setattr(r_mod, "SentenceTransformer", lambda model_name: mock_model)

        mock_index = MagicMock()
        mock_index.search.return_value = ([[0.88, 0.72]], [[0, 1]])
        monkeypatch.setattr(r_mod.faiss, "read_index", lambda path: mock_index)

        retriever = PassageRetriever(str(mock_index_dir))

        # When chunk_id 0 is already used, _retrieve_new should filter it out and return only chunk 1
        used_ids = {0}
        new_passages = _retrieve_new(retriever, "query", k=1, used_ids=used_ids)

        assert len(new_passages) == 1
        assert new_passages[0]["chunk_id"] == 1
        assert "chunk_hash" in new_passages[0]
        assert "source_version" in new_passages[0]

