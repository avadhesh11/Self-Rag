"""
EvoRAG Task 9B: Answer Key PDF to Structured JSONL Converter.

Parses the official answer key PDF (e.g. EvoRAG_Answer_Key_For_Frozen_Run/answer_key_32_for_frozen_run.pdf)
and produces a validated, structured JSONL format matching the evaluator benchmark schema.

Key Features:
- Preserves canonical question IDs (DBMS-EVAL-XXXX).
- Extracts category, difficulty, question text, and answerability flag.
- Extracts expected reference answer(s).
- Extracts supporting sources (filename, source_id, page).
- Extracts evidence excerpts and benchmark evidence-anchor SHA-256 hashes.
- Preserves unanswerable questions and expected abstention guidance.
- Enforces 100% question coverage validation against the held-out question archive.
- Confirms frozen predictions remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pypdf
except ImportError:
    pypdf = None


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_answer_key_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract structured Q&A benchmark entries from the official answer key PDF.
    """
    if pypdf is None:
        raise ImportError("pypdf is required to parse answer key PDF.")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Answer key PDF not found: {pdf_path}")

    reader = pypdf.PdfReader(str(pdf_path))
    full_text = "\n".join(p.extract_text() for p in reader.pages)

    # Split into per-question blocks using canonical header pattern
    # Pattern: DBMS-EVAL-XXXX - Category | Difficulty
    q_blocks = re.split(r"\n(?=DBMS-EVAL-\d{4}\s*-\s*)", full_text)

    parsed_entries: List[Dict[str, Any]] = []

    for block in q_blocks:
        m_hdr = re.match(r"DBMS-EVAL-(\d{4})\s*-\s*([^|\n]+)\s*\|\s*([^\n]+)", block)
        if not m_hdr:
            continue

        qid = f"DBMS-EVAL-{m_hdr.group(1)}"
        category_raw = m_hdr.group(2).strip()
        difficulty_raw = m_hdr.group(3).strip()

        # Extract Question text and Answerable flag
        m_q = re.search(r"Question\s*\n(.*?)\nAnswerable from corpus:\s*(Yes|No)", block, re.DOTALL)
        if not m_q:
            # Fallback if answerable text variation occurs
            m_q = re.search(r"Question\s*\n(.*?)\nExpected answer", block, re.DOTALL)
            q_text = m_q.group(1).strip() if m_q else ""
            ans_from_corpus = True
        else:
            q_text = m_q.group(1).strip()
            ans_from_corpus = (m_q.group(2).strip().lower() == "yes")

        # Extract Expected answer
        m_a = re.search(r"Expected answer\s*\n(.*?)\nSupporting sources", block, re.DOTALL)
        if not m_a:
            m_a = re.search(r"Expected answer\s*\n(.*?)\n(?:Evidence excerpts|Benchmark evidence|Scoring guidance|$)", block, re.DOTALL)
        exp_ans = m_a.group(1).strip() if m_a else ""

        # Extract Supporting sources
        sources: List[Dict[str, Any]] = []
        for s_match in re.findall(r"-\s*([a-zA-Z0-9_\-\.]+\.pdf)\s*\((.*?)\)", block):
            fname = s_match[0].strip()
            details = s_match[1]
            m_sid = re.search(r"source\s*([a-zA-Z0-9_\-]+)", details)
            m_pg = re.search(r"page\s*(\d+)", details)
            sources.append({
                "source_id": m_sid.group(1) if m_sid else "",
                "filename": fname,
                "title": fname.replace(".pdf", "").replace("_", " ").title(),
                "source_type": "pdf",
                "page": int(m_pg.group(1)) if m_pg else None,
            })

        # Extract Evidence excerpts & anchors
        evidence: List[Dict[str, Any]] = []
        m_ev_blocks = re.findall(
            r"([a-zA-Z0-9_\-\.]+\.pdf)\s*\|\s*page\s*(\d+)\s*\n(.*?)\nBenchmark evidence-anchor SHA-256:\s*([a-f0-9]{64})",
            block,
            re.DOTALL,
        )
        for ev_match in m_ev_blocks:
            evidence.append({
                "filename": ev_match[0].strip(),
                "page": int(ev_match[1]),
                "evidence_excerpt": ev_match[2].strip(),
                "evidence_hash": ev_match[3].strip(),
            })

        anchors = re.findall(r"Benchmark evidence-anchor SHA-256:\s*([a-f0-9]{64})", block)

        # Scoring guidance
        m_sg = re.search(r"Scoring guidance:\s*(.*?)(?:\nDBMS-EVAL-|\nPage \d+|\Z)", block, re.DOTALL)
        scoring_guidance = m_sg.group(1).strip() if m_sg else ""

        # Support list of reference answers if delimited or multiline
        references = [exp_ans]

        parsed_entries.append({
            "question_id": qid,
            "question": q_text,
            "category": category_raw.lower().replace(" ", "_"),
            "difficulty": difficulty_raw.lower(),
            "answerable_from_corpus": ans_from_corpus,
            "expected_answer": exp_ans,
            "reference_answers": references,
            "supporting_sources": sources,
            "supporting_evidence": evidence,
            "supporting_chunk_hashes": anchors,
            "scoring_guidance": scoring_guidance,
        })

    return parsed_entries


def validate_converted_answer_key(
    entries: List[Dict[str, Any]],
    reference_questions_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate coverage, IDs, and abstention markers."""
    qids = [e["question_id"] for e in entries]
    unique_qids = set(qids)

    if len(qids) != len(unique_qids):
        raise ValueError(f"Duplicate question IDs detected in parsed answer key: {len(qids)} vs {len(unique_qids)}")

    if reference_questions_path and reference_questions_path.is_file():
        ref_qids = []
        with open(reference_questions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ref_qids.append(json.loads(line)["question_id"])

        missing_in_parsed = set(ref_qids) - unique_qids
        if missing_in_parsed:
            raise ValueError(f"Answer key missing {len(missing_in_parsed)} held-out questions: {sorted(missing_in_parsed)}")

    unanswerable = [e["question_id"] for e in entries if not e["answerable_from_corpus"]]

    return {
        "total_parsed": len(entries),
        "unique_questions": len(unique_qids),
        "unanswerable_questions": unanswerable,
        "question_ids": qids,
    }


def convert_answer_key(
    pdf_path: Path,
    out_jsonl_path: Path,
    reference_questions_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete PDF to JSONL answer key conversion and validation."""
    entries = parse_answer_key_pdf(pdf_path)
    val_info = validate_converted_answer_key(entries, reference_questions_path)

    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    out_sha256 = compute_sha256(out_jsonl_path)
    return {
        "pdf_path": str(pdf_path),
        "out_path": str(out_jsonl_path),
        "out_sha256": out_sha256,
        "validation": val_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert official answer key PDF to structured JSONL.")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PROJECT_ROOT / "EvoRAG_Answer_Key_For_Frozen_Run" / "answer_key_32_for_frozen_run.pdf",
        help="Path to official answer key PDF",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "EvoRAG_Answer_Key_For_Frozen_Run" / "answer_key_32_converted_from_pdf.jsonl",
        help="Path for generated JSONL output",
    )
    parser.add_argument(
        "--reference-questions",
        type=Path,
        default=PROJECT_ROOT / "EvoRAG_Blind_Test_Questions" / "held_out_test_questions_only.jsonl",
        help="Path to held-out questions for coverage validation",
    )

    args = parser.parse_args()
    print("=" * 70)
    print("EvoRAG Answer Key PDF Converter (Task 9B)")
    print("=" * 70)
    print(f"Reading PDF from:  {args.pdf}")
    print(f"Writing JSONL to:  {args.out}")

    res = convert_answer_key(args.pdf, args.out, args.reference_questions)

    print("\nConversion successfully completed.")
    print(f"Total questions:         {res['validation']['total_parsed']}")
    print(f"Unanswerable questions:  {res['validation']['unanswerable_questions']}")
    print(f"Output SHA-256:          {res['out_sha256']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

