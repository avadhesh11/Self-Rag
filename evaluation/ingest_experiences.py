"""
EvoRAG Task 9B: Verified Experience Memory Ingestion & Index Builder.

Ingests verified DBMS training experiences into Experience Memory:
- Enforces strict contamination safeguards (rejects held-out questions/IDs).
- Requires explicit approval (VerificationStatus.APPROVED).
- Preserves structured evidence and source provenance (chunk hashes, source versions).
- Rejects experiences lacking evidence or source tracking.
- Validates source version freshness against active corpus metadata.
- Handles duplicates deterministically via SHA-256 experience IDs.
- Safely rebuilds the FAISS vector index and explicit ID map.
- Reports comprehensive ingestion metrics and rejection reasons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evorag.config import (
    DEFAULT_DB_PATH,
    DEFAULT_INDEX_DIR,
    DEFAULT_EMBED_MODEL,
)
from evorag.memory.schema import (
    Experience,
    VerificationStatus,
    generate_experience_id,
    normalize_question,
)
from evorag.memory.store import ExperienceStore
from evorag.memory.index_builder import ExperienceIndexBuilder


def load_held_out_guard(held_out_path: Optional[Path]) -> Tuple[Set[str], Set[str]]:
    """Load held-out question IDs and normalized question strings to prevent contamination."""
    held_out_ids: Set[str] = set()
    held_out_texts: Set[str] = set()

    if held_out_path and held_out_path.is_file():
        with open(held_out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                qid = str(item.get("question_id", "")).strip()
                qtext = str(item.get("question", "")).strip()
                if qid:
                    held_out_ids.add(qid)
                if qtext:
                    held_out_texts.add(normalize_question(qtext))

    return held_out_ids, held_out_texts


def load_corpus_source_versions(meta_path: Optional[Path]) -> Dict[str, str]:
    """Load source document versions (file byte hashes) from index metadata."""
    if not meta_path or not meta_path.is_file():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        # Check source_versions dictionary in meta.json
        if "source_versions" in data and isinstance(data["source_versions"], dict):
            return data["source_versions"]
        if "files" in data and isinstance(data["files"], dict):
            # mapping of filename -> sha256
            return {k: v.get("sha256", "") if isinstance(v, dict) else str(v) for k, v in data["files"].items()}
    except Exception:
        pass
    return {}


def ingest_experiences(
    input_path: Path,
    db_path: Path = DEFAULT_DB_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    corpus_meta_path: Optional[Path] = None,
    held_out_path: Optional[Path] = None,
    rebuild_index: bool = True,
) -> Dict[str, Any]:
    """
    Ingest verified experiences from input JSONL into SQLite and rebuild the FAISS index.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input experiences file not found: {input_path}")

    held_out_ids, held_out_texts = load_held_out_guard(held_out_path)
    corpus_versions = load_corpus_source_versions(corpus_meta_path)

    store = ExperienceStore(db_path)
    total_evaluated = 0
    ingested_count = 0
    rejected_records: List[Dict[str, Any]] = []
    rejection_summary: Dict[str, int] = {}

    def reject(item: Dict[str, Any], reason: str):
        rejection_summary[reason] = rejection_summary.get(reason, 0) + 1
        rejected_records.append({
            "question_id": item.get("question_id", "unknown"),
            "question": item.get("question", "")[:60],
            "reason": reason,
        })

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_evaluated += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                reject({"question_id": f"line_{line_num}"}, f"MALFORMED_JSON: {e}")
                continue

            qid = str(item.get("question_id") or "").strip()
            qtext = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()

            # 1. Contamination check
            if qid in held_out_ids or qid.startswith("DBMS-EVAL-"):
                reject(item, "CONTAMINATION: Matches held-out evaluation question ID")
                continue

            norm_q = normalize_question(qtext)
            if norm_q in held_out_texts:
                reject(item, "CONTAMINATION: Exact match with held-out test question")
                continue

            # 2. Non-empty question and answer
            if not qtext:
                reject(item, "EMPTY_QUESTION: Question text is missing or whitespace")
                continue
            if not answer:
                reject(item, "EMPTY_ANSWER: Answer text is missing or whitespace")
                continue

            # 3. Explicit approval check
            status = str(item.get("verification_status") or "").strip().lower()
            if status != VerificationStatus.APPROVED:
                reject(item, f"NOT_APPROVED: Verification status is '{status}', expected '{VerificationStatus.APPROVED}'")
                continue

            # 4. Usable evidence check
            evidence = item.get("evidence")
            if not evidence or not isinstance(evidence, list):
                reject(item, "MISSING_EVIDENCE: Experience lacks supporting evidence list")
                continue

            valid_evidence_passages = []
            for ev in evidence:
                if isinstance(ev, dict) and ev.get("text") and ev.get("source"):
                    valid_evidence_passages.append(ev)
            if not valid_evidence_passages:
                reject(item, "MISSING_EVIDENCE: No valid evidence passages with text and source found")
                continue

            # 5. Source version tracking check
            source_versions = item.get("source_versions")
            if not source_versions or not isinstance(source_versions, dict):
                # Try inferring from evidence passages
                inferred = {}
                for ev in valid_evidence_passages:
                    src = ev.get("source")
                    ver = ev.get("source_version")
                    if src and ver:
                        inferred[src] = str(ver)
                if not inferred:
                    reject(item, "MISSING_SOURCE_VERSIONS: Experience lacks source_versions tracking metadata")
                    continue
                source_versions = inferred

            # 6. Freshness check against active corpus (if metadata available)
            stale_detected = False
            for src_name, stored_ver in source_versions.items():
                if corpus_versions and src_name in corpus_versions:
                    active_ver = corpus_versions[src_name]
                    if stored_ver != active_ver:
                        reject(item, f"STALE_SOURCE_VERSION: {src_name} version mismatch ({stored_ver[:8]} vs {active_ver[:8]})")
                        stale_detected = True
                        break
            if stale_detected:
                continue

            # 7. Chunk hashes check
            chunk_hashes = item.get("chunk_hashes")
            if not chunk_hashes or not isinstance(chunk_hashes, list):
                chunk_hashes = [ev.get("chunk_hash") for ev in valid_evidence_passages if ev.get("chunk_hash")]

            # 8. Deterministic experience construction and persistence
            exp = Experience(
                question=qtext,
                answer=answer,
                normalized_question=norm_q,
                evidence=valid_evidence_passages,
                source_versions=source_versions,
                chunk_hashes=chunk_hashes,
                verification_status=VerificationStatus.APPROVED,
            )

            if store.experience_exists(exp.experience_id):
                store.update_verification_status(exp.experience_id, exp.verification_status)
            else:
                store.add_experience(exp)
            ingested_count += 1

    # Rebuild FAISS index
    index_rebuild_success = False
    num_vectors = 0
    if rebuild_index:
        approved_exps = store.list_experiences(status=VerificationStatus.APPROVED)
        builder = ExperienceIndexBuilder(index_dir=index_dir)
        meta = builder.build_from_experiences(approved_exps)
        index_rebuild_success = True
        num_vectors = meta.get("num_vectors", len(approved_exps))

    return {
        "input_path": str(input_path),
        "db_path": str(db_path),
        "index_dir": str(index_dir),
        "total_evaluated": total_evaluated,
        "ingested_count": ingested_count,
        "rejected_count": len(rejected_records),
        "rejection_summary": rejection_summary,
        "rejected_records": rejected_records,
        "total_approved_in_db": store.count(status=VerificationStatus.APPROVED),
        "index_rebuild_success": index_rebuild_success,
        "num_index_vectors": num_vectors,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest verified DBMS experiences into EvoRAG Experience Memory.")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=PROJECT_ROOT / "data" / "dbms_training_experiences.jsonl",
        help="Path to training experiences JSONL",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "experience_memory.db",
        help="Path to SQLite experience database",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory to save experience FAISS index and metadata",
    )
    parser.add_argument(
        "--corpus-meta",
        type=Path,
        default=PROJECT_ROOT / "index" / "dbms_sql_v1" / "meta.json",
        help="Path to corpus meta.json for source freshness validation",
    )
    parser.add_argument(
        "--held-out-questions",
        type=Path,
        default=PROJECT_ROOT / "EvoRAG_Blind_Test_Questions" / "held_out_test_questions_only.jsonl",
        help="Path to held-out questions to enforce zero-contamination safeguards",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip FAISS index rebuilding",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EvoRAG Verified Experience Memory Ingestion (Task 9B)")
    print("=" * 70)
    print(f"Reading experiences from: {args.input}")
    print(f"Experience SQLite Store:  {args.db_path}")
    print(f"FAISS Index Directory:    {args.index_dir}")
    print(f"Held-Out Contamination:   {args.held_out_questions}")

    res = ingest_experiences(
        input_path=args.input,
        db_path=args.db_path,
        index_dir=args.index_dir,
        corpus_meta_path=args.corpus_meta,
        held_out_path=args.held_out_questions,
        rebuild_index=not args.no_rebuild,
    )

    print("\nIngestion completed successfully.")
    print(f"Total evaluated:          {res['total_evaluated']}")
    print(f"Successfully ingested:    {res['ingested_count']}")
    print(f"Rejected records:         {res['rejected_count']}")
    if res["rejection_summary"]:
        print(f"Rejection breakdown:      {res['rejection_summary']}")
    print(f"Total approved in DB:     {res['total_approved_in_db']}")
    print(f"FAISS index vectors:      {res['num_index_vectors']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
