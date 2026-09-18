"""
RAG comparison demo: Normal RAG (Gemini) vs Self-RAG (selfrag_llama2_7b).

--- Mode 2: Self-RAG (original behaviour, unchanged) ---
    python demo.py "How large is the Amazon rainforest?"
    python demo.py "What is 2+2?"                      # should skip retrieval
    python demo.py --model path/to/custom.gguf "..."
    SELFRAG_BACKEND=vllm python demo.py --model selfrag/selfrag_llama2_7b "..."

--- Mode 1: Normal RAG only (Gemini, no Self-RAG model needed) ---
    python demo.py --mode normal "How large is the Amazon rainforest?"

--- Mode 3: Compare Both ---
    python demo.py --mode both "How large is the Amazon rainforest?"

GEMINI_API_KEY must be set in .env (or the environment) for modes 1 and 3.

Self-RAG prints a full trace: retrieval decision, per-passage ISREL/ISSUP/ISUSE,
which candidate was selected, and the final cleaned answer.
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # loads .env so GEMINI_API_KEY is available

from retriever.retriever import PassageRetriever
from selfrag.generator import SelfRAGGenerator
from selfrag.pipeline import self_rag_answer
from comparison.runner import run_rag, print_normal_rag, print_self_rag, print_comparison


def print_trace(result):
    print(f"\nQuery: {result.query}")
    if result.retrieved:
        print(f"[Retrieve] decision: Yes ({result.decision})")
    elif result.decision is None:
        print("[Retrieve] decision: (none emitted -- model answered directly)")
    else:
        print(f"[Retrieve] decision: No ({result.decision})")

    if not result.retrieved:
        print(f"\nFinal answer (no retrieval needed):\n{result.final_answer}")
        return

    for c in result.candidates:
        tag = "-> SELECTED" if c is result.selected else ""
        print(f"\n  Passage (page {c.passage.get('page')}): "
              f"{c.passage['text'][:100]}...")
        print(f"    ISREL={c.isrel}  ISSUP={c.issup}  ISUSE={c.isuse}  "
              f"score={c.score:.2f} {tag}")

    print(f"\nFinal answer:\n{result.final_answer}")


def _pick_mode_interactive() -> str:
    """Show a mode-selection menu and return the chosen mode string."""
    print("\n" + "=" * 40)
    print("        RAG COMPARISON DEMO")
    print("=" * 40)
    print("  1. Normal RAG  (Gemini)")
    print("  2. Self-RAG    (selfrag_llama2_7b)")
    print("  3. Compare Both")
    print("=" * 40)
    choice = input("Select mode [1/2/3] (default 2): ").strip() or "2"
    return {"1": "normal", "2": "self", "3": "both"}.get(choice, "self")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default=None,
                    help="Question to answer (prompted interactively if omitted)")
    ap.add_argument("--index", default=os.environ.get("INDEX_DIR", "index"),
                    help="Directory with faiss.index/corpus.jsonl")
    ap.add_argument("--model",
                    default=os.environ.get(
                        "SELF_RAG_MODEL_PATH",
                        "models/selfrag_llama2_7b.Q4_K_M.gguf"
                    ),
                    help="Path to a .gguf (llamacpp backend) or a HF model id (vllm backend)")
    ap.add_argument("--download-dir", default=None, help="vLLM backend only")
    ap.add_argument("--k", type=int, default=5, help="Number of passages to retrieve")
    ap.add_argument("--mode", default=None,
                    choices=["normal", "self", "both"],
                    help="RAG mode: normal | self | both  (interactive menu if omitted)")
    args = ap.parse_args()

    # Determine mode -- CLI flag wins, else interactive menu
    mode = args.mode or _pick_mode_interactive()

    # Determine query -- CLI positional arg wins, else prompt
    query = args.query or input("\nQuestion: ").strip()
    if not query:
        query = "How large is the Amazon rainforest?"

    print("\nLoading retriever ...")
    retriever = PassageRetriever(args.index)

    # Only load generators that are actually needed
    normal_generator = None
    self_generator   = None

    if mode in ("normal", "both"):
        from normal_rag.generator import GeminiGenerator
        print("Initialising Gemini generator ...")
        normal_generator = GeminiGenerator()

    if mode in ("self", "both"):
        backend = os.environ.get("SELFRAG_BACKEND", "llamacpp")
        print(f"Loading Self-RAG generator (backend: {backend}) ...")
        self_generator = SelfRAGGenerator(args.model, download_dir=args.download_dir)

    # Run via the unified comparison runner
    result = run_rag(
        query, mode,
        normal_generator=normal_generator,
        self_generator=self_generator,
        retriever=retriever,
        k=args.k,
    )

    # Display results
    if mode == "normal":
        print_normal_rag(result.normal)
    elif mode == "self":
        # Use the original Self-RAG trace format (print_trace) for full fidelity
        print_trace(result.self_rag)
    else:
        print_comparison(result)


if __name__ == "__main__":
    main()
