import hashlib
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

from index.build_index import (
    compute_file_hash,
    normalize_chunk_text,
    compute_chunk_hash,
    merge_to_target_size,
)


class TestProvenanceAndChunkHashing:
    """Tests for deterministic chunk hashes, source versioning, and provenance tracking."""

    def test_deterministic_chunk_hash(self):
        """Verify that repeated hashing of the same source and text produces identical hashes."""
        source_id = "sample.pdf"
        text = "Self-RAG improves quality and factuality of LLM generation."
        hash1 = compute_chunk_hash(source_id, text)
        hash2 = compute_chunk_hash(source_id, text)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        # Verify it matches direct hashlib computation
        expected = hashlib.sha256(f"{source_id}:{text}".encode("utf-8")).hexdigest()
        assert hash1 == expected

    def test_source_version_changes_on_file_modification(self, tmp_path: Path):
        """Verify source_version changes when file content is modified."""
        pdf_file = tmp_path / "document.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 Initial document content")
        initial_hash = compute_file_hash(pdf_file)

        # Re-reading unchanged file produces identical hash
        assert compute_file_hash(pdf_file) == initial_hash

        # Modify file bytes
        pdf_file.write_bytes(b"%PDF-1.4 Modified document content with changes")
        modified_hash = compute_file_hash(pdf_file)

        assert initial_hash != modified_hash
        assert len(initial_hash) == 64
        assert len(modified_hash) == 64

    def test_different_sources_produce_different_chunk_hashes(self):
        """Verify different source documents do not share chunk hash even if text is identical."""
        text = "Identical text passage across different source files."
        hash_doc_a = compute_chunk_hash("doc_a.pdf", text)
        hash_doc_b = compute_chunk_hash("doc_b.pdf", text)
        assert hash_doc_a != hash_doc_b

    def test_different_text_produces_different_chunk_hashes(self):
        """Verify changing chunk text changes the chunk hash."""
        source_id = "document.pdf"
        hash_text_1 = compute_chunk_hash(source_id, "Passage content version 1.")
        hash_text_2 = compute_chunk_hash(source_id, "Passage content version 2.")
        assert hash_text_1 != hash_text_2

    def test_text_normalization_whitespace(self):
        """Verify leading, trailing, and internal whitespace variations normalize to identical hashes."""
        source_id = "document.pdf"
        text_clean = "This is a normalized sentence with stable spacing."
        text_with_spaces = "   This   is  a   normalized sentence   with   stable   spacing.   "
        text_with_tabs_newlines = "This is\ta normalized\nsentence\n\twith stable spacing.\n"

        norm_clean = normalize_chunk_text(text_clean)
        norm_spaces = normalize_chunk_text(text_with_spaces)
        norm_tabs_newlines = normalize_chunk_text(text_with_tabs_newlines)

        assert norm_clean == norm_spaces == norm_tabs_newlines

        hash_clean = compute_chunk_hash(source_id, text_clean)
        hash_spaces = compute_chunk_hash(source_id, text_with_spaces)
        hash_tabs_newlines = compute_chunk_hash(source_id, text_with_tabs_newlines)

        assert hash_clean == hash_spaces == hash_tabs_newlines

    def test_text_normalization_unicode_nfkc(self):
        """Verify Unicode decomposed vs precomposed representations normalize to identical hashes."""
        source_id = "document.pdf"
        # 'é' as single precomposed code point (\u00e9) vs 'e' + combining acute accent (\u0301)
        text_precomposed = "Caf\u00e9 knowledge base"
        text_decomposed = "Cafe\u0301 knowledge base"

        assert text_precomposed != text_decomposed  # raw strings differ

        norm_pre = normalize_chunk_text(text_precomposed)
        norm_dec = normalize_chunk_text(text_decomposed)
        assert norm_pre == norm_dec

        hash_pre = compute_chunk_hash(source_id, text_precomposed)
        hash_dec = compute_chunk_hash(source_id, text_decomposed)
        assert hash_pre == hash_dec

    def test_chunk_metadata_schema_preservation(self, tmp_path: Path):
        """Verify that all existing chunk metadata fields and new provenance fields are present."""
        sample_pdf = tmp_path / "test.pdf"
        sample_pdf.write_bytes(b"%PDF-1.4 Dummy PDF Content for metadata test")

        source_version = compute_file_hash(sample_pdf)
        paragraphs = [
            {"text": "First sentence of paragraph.", "page": 1},
            {"text": "Second sentence of paragraph.", "page": 1},
        ]
        chunks = merge_to_target_size(paragraphs, target_words=50, overlap_words=10)

        # Simulate the tagging performed by build_index
        for i, c in enumerate(chunks):
            c["source"] = sample_pdf.name
            c["source_version"] = source_version
            c["chunk_hash"] = compute_chunk_hash(sample_pdf.name, c["text"])
            c["chunk_id"] = i

        assert len(chunks) == 1
        chunk = chunks[0]

        # Check existing fields preserved
        assert "text" in chunk
        assert "page" in chunk
        assert "source" in chunk
        assert "chunk_id" in chunk
        assert isinstance(chunk["chunk_id"], int)
        assert chunk["page"] == 1
        assert chunk["source"] == "test.pdf"

        # Check new provenance fields
        assert "source_version" in chunk
        assert "chunk_hash" in chunk
        assert chunk["source_version"] == source_version
        assert chunk["chunk_hash"] == compute_chunk_hash("test.pdf", chunk["text"])

    def test_build_index_pipeline_provenance(self, tmp_path: Path, monkeypatch):
        """Verify build_index end-to-end writes chunk_hash and source_versions to corpus and meta."""
        from index import build_index as bi_mod

        # Create dummy PDF files
        pdf_file = tmp_path / "sample.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 Dummy PDF Content")
        out_dir = tmp_path / "index_out"

        # Mock extract_paragraphs to return synthetic sentences
        mock_paragraphs = [
            {"text": "Sentence 1 of passage.", "page": 1},
            {"text": "Sentence 2 of passage.", "page": 1},
        ]
        monkeypatch.setattr(bi_mod, "extract_paragraphs", lambda p: mock_paragraphs)

        # Mock SentenceTransformer and FAISS index
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        monkeypatch.setattr(bi_mod, "SentenceTransformer", lambda m: mock_model)

        mock_faiss_idx = MagicMock()
        monkeypatch.setattr(bi_mod.faiss, "IndexFlatIP", lambda d: mock_faiss_idx)
        monkeypatch.setattr(bi_mod.faiss, "write_index", lambda idx, path: None)

        chunks = bi_mod.build_index([str(pdf_file)], str(out_dir))

        assert len(chunks) == 1
        assert "chunk_hash" in chunks[0]
        assert "source_version" in chunks[0]
        assert chunks[0]["source"] == "sample.pdf"
        assert chunks[0]["chunk_id"] == 0

        # Verify corpus.jsonl content
        corpus_path = out_dir / "corpus.jsonl"
        assert corpus_path.exists()
        saved_lines = [json.loads(line) for line in corpus_path.read_text().strip().split("\n")]
        assert len(saved_lines) == 1
        assert saved_lines[0]["chunk_hash"] == chunks[0]["chunk_hash"]
        assert saved_lines[0]["source_version"] == chunks[0]["source_version"]
        assert saved_lines[0]["chunk_id"] == 0

        # Verify meta.json content
        meta_path = out_dir / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["sources"] == ["sample.pdf"]
        assert "source_versions" in meta
        assert meta["source_versions"]["sample.pdf"] == compute_file_hash(pdf_file)

