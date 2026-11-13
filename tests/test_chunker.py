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
    _SENTENCE_SPLIT_RE,
    clean_pdf_text,
    merge_to_target_size,
    _get_overlap_buffer,
    split_sentences,
)


class TestSentenceSplitter:
    """Regression tests for sentence splitting and abbreviation handling."""

    def test_abbreviation_references_intact(self):
        """Verify references (Sec. 10, Art. 14, No. 5, Rs. 500) remain intact."""
        # Individual reference checks
        sents_sec = split_sentences("Refer to Sec. 10 for definitions. Next sentence follows.")
        assert len(sents_sec) == 2
        assert sents_sec[0] == "Refer to Sec. 10 for definitions."
        assert sents_sec[1] == "Next sentence follows."

        sents_art = split_sentences("Under Art. 14, equality is guaranteed. Second sentence.")
        assert len(sents_art) == 2
        assert sents_art[0] == "Under Art. 14, equality is guaranteed."
        assert sents_art[1] == "Second sentence."

        sents_no = split_sentences("Please inspect Order No. 5 before shipment. It contains items.")
        assert len(sents_no) == 2
        assert sents_no[0] == "Please inspect Order No. 5 before shipment."
        assert sents_no[1] == "It contains items."

        sents_rs = split_sentences("The application fee is Rs. 500 only. Late fees apply afterwards.")
        assert len(sents_rs) == 2
        assert sents_rs[0] == "The application fee is Rs. 500 only."
        assert sents_rs[1] == "Late fees apply afterwards."

    def test_combined_references_in_single_sentence(self):
        """Verify multiple references within a single sentence do not cause incorrect splits."""
        text = (
            "According to Sec. 10 and Art. 14 of Act No. 5, the penalty is Rs. 500. "
            "This is the next sentence."
        )
        sents = split_sentences(text)
        assert len(sents) == 2
        assert sents[0] == "According to Sec. 10 and Art. 14 of Act No. 5, the penalty is Rs. 500."
        assert sents[1] == "This is the next sentence."

    def test_lowercase_abbreviations(self):
        """Verify lowercase variants (sec. 10, art. 14, no. 5, rs. 500) remain intact."""
        text = "See sec. 10 and art. 14 or no. 5 with rs. 500. End of text."
        sents = split_sentences(text)
        assert len(sents) == 2
        assert sents[0] == "See sec. 10 and art. 14 or no. 5 with rs. 500."
        assert sents[1] == "End of text."

    def test_ordinary_sentence_boundaries(self):
        """Verify standard delimiters (. ! ?) followed by uppercase or digit split correctly."""
        text = "Sentence one. Sentence two! Sentence three? 4th sentence starts with a digit."
        sents = split_sentences(text)
        assert sents == [
            "Sentence one.",
            "Sentence two!",
            "Sentence three?",
            "4th sentence starts with a digit.",
        ]

    def test_paragraphs_with_newlines(self):
        """Verify text with newlines/paragraph breaks splits cleanly."""
        text = "Paragraph one ending here.\nParagraph two starting here.\n\nParagraph three begins now!"
        sents = split_sentences(text)
        assert len(sents) == 3
        assert sents[0] == "Paragraph one ending here."
        assert sents[1] == "Paragraph two starting here."
        assert sents[2] == "Paragraph three begins now!"

    def test_empty_and_whitespace_text(self):
        """Verify empty and whitespace strings return empty list."""
        assert split_sentences("") == []
        assert split_sentences("   ") == []
        assert split_sentences("\n\t  \r\n") == []

    def test_text_without_punctuation(self):
        """Verify text without punctuation is not fragmented."""
        text = "This is a continuous stream of text without any punctuation symbols"
        sents = split_sentences(text)
        assert sents == [text]

    def test_sentence_boundary_immediately_after_reference(self):
        """Verify a real sentence ending right after a reference digit still splits."""
        text = "The amount due is Rs. 500. Next clause applies."
        sents = split_sentences(text)
        assert len(sents) == 2
        assert sents[0] == "The amount due is Rs. 500."
        assert sents[1] == "Next clause applies."


class TestChunkConstruction:
    """Tests for chunk merging and overlap behavior in merge_to_target_size."""

    def test_merge_small_sentences_into_target_size(self):
        """Verify sentences on the same page are merged up to target size."""
        paragraphs = [
            {"text": "Short sentence 1.", "page": 1},
            {"text": "Short sentence 2.", "page": 1},
            {"text": "Short sentence 3.", "page": 1},
        ]
        # Target size 10 words, so all 3 sentences fit into 1 chunk
        chunks = merge_to_target_size(paragraphs, target_words=10, overlap_words=2)
        assert len(chunks) == 1
        assert chunks[0]["page"] == 1
        assert chunks[0]["text"] == "Short sentence 1. Short sentence 2. Short sentence 3."

    def test_merge_page_boundary_flushes_without_overlap(self):
        """Verify changing pages flushes the buffer and does not carry overlap across pages."""
        paragraphs = [
            {"text": "Page one sentence.", "page": 1},
            {"text": "Page two sentence.", "page": 2},
        ]
        chunks = merge_to_target_size(paragraphs, target_words=50, overlap_words=10)
        assert len(chunks) == 2
        assert chunks[0] == {"text": "Page one sentence.", "page": 1}
        assert chunks[1] == {"text": "Page two sentence.", "page": 2}

    def test_merge_same_page_overlap_buffer(self):
        """Verify overlap buffer retains trailing sentences on the same page."""
        # Create sentences with known word counts
        s1 = "Alpha " * 10  # 10 words
        s2 = "Beta " * 10   # 10 words
        s3 = "Gamma " * 10  # 10 words
        paragraphs = [
            {"text": s1.strip(), "page": 1},
            {"text": s2.strip(), "page": 1},
            {"text": s3.strip(), "page": 1},
        ]
        # Target words = 15 -> s1 + s2 = 20 words (>= 15), flushes first chunk.
        # Overlap words = 10 -> retains s2 (10 words) for next chunk.
        # Next, s2 + s3 = 20 words (>= 15), flushes second chunk and retains s3.
        # Loop ends, flushing the remaining trailing overlap buffer (s3).
        chunks = merge_to_target_size(paragraphs, target_words=15, overlap_words=10)
        assert len(chunks) == 3
        assert chunks[0]["text"] == f"{s1.strip()} {s2.strip()}"
        assert chunks[1]["text"] == f"{s2.strip()} {s3.strip()}"
        assert chunks[2]["text"] == s3.strip()

    def test_clean_pdf_text_artifacts(self):
        """Verify clean_pdf_text resolves common OCR/ligature/hyphenation artifacts."""
        assert clean_pdf_text("SELF -RAG") == "SELF-RAG"
        assert clean_pdf_text("Gen- eration") == "Generation"
        assert clean_pdf_text("") == ""
        assert clean_pdf_text("Multiple   spaces   between   words") == "Multiple spaces between words"
