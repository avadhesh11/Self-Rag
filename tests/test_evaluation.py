import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Ensure the project root (containing 'evaluation', 'retriever', etc.) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock optional heavy dependencies if not installed in the current environment
for mod in ["faiss", "sentence_transformers"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from evaluation.evaluate import load_questions


SAMPLE_QUESTIONS_DATA = [
    {
        "category": "in_knowledge",
        "description": "Questions with answers present in knowledge base.",
        "questions": [
            "What are the four types of reflection tokens used in Self-RAG?",
            "How does Self-RAG decide whether to retrieve external passages?",
        ],
    },
    {
        "category": "no_retrieval",
        "description": "Questions not needing retrieval.",
        "questions": [
            "What is 2 + 2?",
            "Write a short haiku about AI: \u201cTesting unicode\u201d \u2705",
        ],
    },
]


def test_load_questions_utf8_bom(tmp_path: Path):
    """Verify that load_questions successfully loads a JSON file containing a UTF-8 BOM."""
    file_path = tmp_path / "questions_bom.json"
    raw_json = json.dumps(SAMPLE_QUESTIONS_DATA, ensure_ascii=False)
    # Write using utf-8-sig which explicitly includes the UTF-8 BOM (\xef\xbb\xbf)
    file_path.write_text(raw_json, encoding="utf-8-sig")

    # Confirm that the file actually starts with the 3-byte UTF-8 BOM
    raw_bytes = file_path.read_bytes()
    assert raw_bytes.startswith(b"\xef\xbb\xbf"), "Test file must contain UTF-8 BOM"

    # Verify loading without category filter
    questions = load_questions(file_path)
    assert len(questions) == 4
    assert questions[0] == {
        "category": "in_knowledge",
        "question": "What are the four types of reflection tokens used in Self-RAG?",
    }
    assert questions[1] == {
        "category": "in_knowledge",
        "question": "How does Self-RAG decide whether to retrieve external passages?",
    }
    assert questions[2] == {
        "category": "no_retrieval",
        "question": "What is 2 + 2?",
    }
    assert questions[3] == {
        "category": "no_retrieval",
        "question": "Write a short haiku about AI: \u201cTesting unicode\u201d \u2705",
    }

    # Verify category filtering works with BOM files
    filtered = load_questions(file_path, category="in_knowledge")
    assert len(filtered) == 2
    assert all(item["category"] == "in_knowledge" for item in filtered)

    filtered_none = load_questions(file_path, category="non_existent")
    assert filtered_none == []


def test_load_questions_utf8_no_bom(tmp_path: Path):
    """Verify that load_questions successfully loads a standard UTF-8 JSON file without BOM."""
    file_path = tmp_path / "questions_no_bom.json"
    raw_json = json.dumps(SAMPLE_QUESTIONS_DATA, ensure_ascii=False)
    # Write using standard utf-8 without BOM
    file_path.write_text(raw_json, encoding="utf-8")

    # Confirm that the file does NOT start with the UTF-8 BOM
    raw_bytes = file_path.read_bytes()
    assert not raw_bytes.startswith(b"\xef\xbb\xbf"), "Test file must not contain UTF-8 BOM"

    # Verify loading without category filter
    questions = load_questions(file_path)
    assert len(questions) == 4
    assert questions[0] == {
        "category": "in_knowledge",
        "question": "What are the four types of reflection tokens used in Self-RAG?",
    }
    assert questions[1] == {
        "category": "in_knowledge",
        "question": "How does Self-RAG decide whether to retrieve external passages?",
    }
    assert questions[2] == {
        "category": "no_retrieval",
        "question": "What is 2 + 2?",
    }
    assert questions[3] == {
        "category": "no_retrieval",
        "question": "Write a short haiku about AI: \u201cTesting unicode\u201d \u2705",
    }

    # Verify category filtering works with regular UTF-8 files
    filtered = load_questions(file_path, category="no_retrieval")
    assert len(filtered) == 2
    assert all(item["category"] == "no_retrieval" for item in filtered)

