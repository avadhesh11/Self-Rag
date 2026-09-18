"""
Evaluation CLI: run questions from evaluation/questions.json through
Normal RAG and/or Self-RAG and print (or save) a comparison table.

Usage:
    # Compare both pipelines on all questions:
    python evaluation/evaluate.py --mode both

    # Normal RAG only (no Self-RAG model needed):
    python evaluation/evaluate.py --mode normal

    # Specific category only:
    python evaluation/evaluate.py --mode both --category in_knowledge

    # Save results as JSON:
    python evaluation/evaluate.py --mode both --out results.json

Requirements:
    - GEMINI_API_KEY in .env (for Normal RAG / both modes)
    - Self-RAG model at models/selfrag_llama2_7b.Q4_K_M.gguf
      (for Self-RAG / both modes, same as demo.py)
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Allow running as `python evaluation/evaluate.py` from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from retriever.retriever import PassageRetriever
from evaluation.metrics import (
    metrics_from_normal,
    metrics_from_self_rag,
    print_metrics_table,
)
from comparison.runner import run_rag


def load_questions(path: Path, category: str | None = None) -> list[dict]:
    """Return flat list of {category, question} dicts from questions.json."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    for group in data:
        if category and group["category"] != category:
            continue
        for q in group["questions"]:
            rows.append({"category": group["category"], "question": q})
    return rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate Normal RAG vs Self-RAG")
    ap.add_argument("--mode", choices=["normal", "self", "both"], default="both")
    ap.add_argument("--category", default=None,
                    help="Run only this question category (e.g. in_knowledge)")
    ap.add_argument("--index", default="index",
                    help="FAISS index directory (same as demo.py)")
    ap.add_argument("--model",
                    default=os.environ.get(
                        "SELF_RAG_MODEL_PATH",
                        "models/selfrag_llama2_7b.Q4_K_M.gguf"
                    ),
                    help="Self-RAG model path (ignored for --mode normal)")
    ap.add_argument("--k", type=int, default=5, help="Number of passages to retrieve")
    ap.add_argument("--out", default=None,
                    help="Optional path to save results as JSON")
    ap.add_argument("--download-dir", default=None, help="vLLM backend only")
    args = ap.parse_args()

    questions_path = Path(__file__).parent / "questions.json"
    questions = load_questions(questions_path, args.category)
    if not questions:
        print(f"No questions found for category: {args.category}")
        sys.exit(1)

    print(f"Loading retriever from {args.index} ...")
    retriever = PassageRetriever(args.index)

    normal_generator = None
    self_generator = None

    if args.mode in ("normal", "both"):
        from normal_rag.generator import GeminiGenerator
        print("Initialising Gemini generator ...")
        normal_generator = GeminiGenerator()

    if args.mode in ("self", "both"):
        from selfrag.generator import SelfRAGGenerator
        print(f"Loading Self-RAG generator ({args.model}) ...")
        self_generator = SelfRAGGenerator(args.model, download_dir=args.download_dir)

    all_results = []

    for i, row in enumerate(questions, start=1):
        query = row["question"]
        cat   = row["category"]
        print(f"\n[{i}/{len(questions)}] [{cat}] {query}")
        print("-" * 60)

        result = run_rag(
            query,
            args.mode,
            normal_generator=normal_generator,
            self_generator=self_generator,
            retriever=retriever,
            k=args.k,
        )

        n_metrics = metrics_from_normal(result.normal)   if result.normal   else None
        s_metrics = metrics_from_self_rag(result.self_rag, result.self_rag_latency) \
                    if result.self_rag else None

        print_metrics_table(n_metrics, s_metrics)

        if result.normal:
            print(f"\nNormal RAG answer:\n{result.normal.answer}")
        if result.self_rag:
            print(f"\nSelf-RAG answer:\n{result.self_rag.final_answer}")

        all_results.append({
            "category": cat,
            "question": query,
            "normal_rag": n_metrics,
            "self_rag":   s_metrics,
            "normal_rag_answer": result.normal.answer if result.normal else None,
            "self_rag_answer":   result.self_rag.final_answer if result.self_rag else None,
        })

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
