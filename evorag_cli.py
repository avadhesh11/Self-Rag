"""EvoRAG Terminal CLI: Experience Memory + Self-RAG.

Usage:
    # Query EvoRAG (checks Experience Memory first, falls back to Self-RAG on miss):
    python evorag_cli.py "What are the four reflection tokens in Self-RAG?"

    # Custom threshold and paths:
    python evorag_cli.py --threshold 0.88 --index index/ "Can I apply for the program?"
"""
import argparse
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from evorag.config import (
    DEFAULT_DB_PATH,
    DEFAULT_INDEX_DIR,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
)
from evorag.memory.store import ExperienceStore
from evorag.memory.retriever import ExperienceRetriever
from evorag.memory.validator import ExperienceValidator
from evorag.memory.index_builder import ExperienceIndexBuilder
from evorag.controller import EvoRAGController, MemoryIndexRebuildError


DIVIDER = "=" * 65


def print_evorag_result(result) -> None:
    """Display clearly formatted outcome distinguishing Path A, Path B, and Path C."""
    print("\n" + DIVIDER)
    print("                    EVORAG EXECUTION RESULT")
    print(DIVIDER)
    print(f"Query: {result.question}\n")

    if result.response_mode == "direct_reuse":
        print("[SOURCE: EXPERIENCE MEMORY (PATH A: EXACT DIRECT REUSE)]")
        print(f"  Response Mode    : {result.response_mode}")
        print(f"  Experience ID    : {result.experience_id}")
        print(f"  Similarity Score : {result.similarity_score:.4f}" if result.similarity_score is not None else "")
        print(f"  Evidence Passages: {len(result.evidence)}")
        print(f"\nStored Answer:\n{result.answer}")
    elif result.response_mode == "experience_assisted":
        print("[SOURCE: EXPERIENCE-ASSISTED SELF-RAG (PATH B: HYBRID CONTEXT INJECTED)]")
        print(f"  Response Mode    : {result.response_mode}")
        print(f"  Memory Context   : Used ({len(result.memory_experience_ids)} experience(s): {result.memory_experience_ids})")
        print(f"  Top Candidate Sim: {result.similarity_score:.4f}" if result.similarity_score is not None else "")
        print(f"  Doc Evidence     : {len(result.document_evidence)} passages")
        print(f"  Memory Evidence  : {len(result.experience_evidence)} passages")
        print(f"\nGenerated Answer:\n{result.answer}")
    else:
        print("[SOURCE: NORMAL SELF-RAG (PATH C: NO REUSABLE MEMORY)]")
        print(f"  Response Mode    : {result.response_mode}")
        if result.fallback_reason:
            print(f"  Fallback Reason  : {result.fallback_reason}")
        print(f"  Doc Evidence     : {len(result.document_evidence)} passages")
        print(f"\nGenerated Answer:\n{result.answer}")
    print(DIVIDER + "\n")


def main():
    parser = argparse.ArgumentParser(description="EvoRAG: Experience Memory + Self-RAG CLI")
    parser.add_argument("query", nargs="?", default=None, help="Question to answer")
    parser.add_argument("--index", default=os.environ.get("INDEX_DIR", "index"),
                        help="Document FAISS index directory")
    parser.add_argument("--memory-db", default=DEFAULT_DB_PATH,
                        help="Path to SQLite Experience Memory database")
    parser.add_argument("--memory-index", default=DEFAULT_INDEX_DIR,
                        help="Path to FAISS Experience Memory index directory")
    parser.add_argument("--model",
                        default=os.environ.get("SELF_RAG_MODEL_PATH", "models/selfrag_llama2_7b.Q4_K_M.gguf"),
                        help="Path to Self-RAG GGUF model")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                        help="Similarity threshold for Experience Memory validation")
    parser.add_argument("--k", type=int, default=5, help="Passages to retrieve on miss")
    parser.add_argument("--max-segments", type=int, default=3, help="Max Self-RAG segments")
    parser.add_argument("--approve", action="store_true",
                        help="Automatically approve and save Self-RAG generated answers to memory")

    args = parser.parse_args()

    query = args.query or input("Enter question: ").strip()
    if not query:
        print("Empty question provided. Exiting.")
        return

    # 1. Initialize Experience Memory Subsystem
    store = ExperienceStore(args.memory_db)
    index_builder = ExperienceIndexBuilder(index_dir=args.memory_index)
    memory_retriever = ExperienceRetriever(index_dir=args.memory_index, store=store)
    validator = ExperienceValidator(similarity_threshold=args.threshold)

    # 2. Lazily initialize document retriever and Self-RAG generator if index exists
    doc_retriever = None
    if Path(args.index).exists() and (Path(args.index) / "meta.json").exists():
        from retriever.retriever import PassageRetriever
        doc_retriever = PassageRetriever(args.index)

    self_generator = None
    if os.environ.get("SELFRAG_BACKEND") == "vllm" or Path(args.model).exists():
        from selfrag.generator import SelfRAGGenerator
        self_generator = SelfRAGGenerator(args.model)

    controller = EvoRAGController(
        store=store,
        memory_retriever=memory_retriever,
        validator=validator,
        generator=self_generator,
        retriever=doc_retriever,
        doc_index_dir=args.index,
        index_builder=index_builder,
        k=args.k,
        max_segments=args.max_segments,
    )

    try:
        try:
            result = controller.answer(query)
        except ValueError as e:
            print(f"[CONFIGURATION ERROR] {e}")
            return

        print_evorag_result(result)

        # Offer approval and persistence if Self-RAG generated the answer
        if not result.memory_hit and result.answer:
            should_save = args.approve
            if not should_save and sys.stdin.isatty():
                ans = input("Approve and save this answer to Experience Memory? [y/N]: ").strip().lower()
                should_save = (ans == "y" or ans == "yes")

            if should_save:
                try:
                    exp_id = controller.approve_and_save(result, user_approved=True)
                    if exp_id:
                        print(f"[SUCCESS] Approved and saved to Experience Memory. Experience ID: {exp_id}")
                    else:
                        print("[WARNING] Could not save answer (lacked sufficient evidence or provenance).")
                except MemoryIndexRebuildError as e:
                    print(f"[WARNING] Experience saved to SQLite ({e.experience_id}), but index rebuild failed: {e}")
                    print("Run controller.recover_index() to restore FAISS index.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
