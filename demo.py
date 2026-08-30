"""
End-to-end Self-RAG demo against a small PDF.

Usage (default backend: llama.cpp over Metal, Apple silicon):
    python demo.py "How large is the Amazon rainforest?"
    python demo.py "What is 2+2?"                      # should skip retrieval
    python demo.py --model path/to/custom.gguf "..."

On a Linux box with an NVIDIA GPU, switch backends:
    SELFRAG_BACKEND=vllm python demo.py --model selfrag/selfrag_llama2_7b "..."

Prints a full trace: retrieval decision, per-passage ISREL/ISSUP/ISUSE,
which candidate was selected, and the final cleaned answer. The trace is
the point -- it's what proves the control loop is actually doing
something, not just wrapping a plain RAG call.
"""
import argparse
import os
import sys

from retriever.retriever import PassageRetriever
from selfrag.generator import SelfRAGGenerator
from selfrag.pipeline import self_rag_answer


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="How large is the Amazon rainforest?")
    ap.add_argument("--index", default="index", help="Directory with faiss.index/corpus.jsonl")
    ap.add_argument("--model", default="models/selfrag_llama2_7b.Q4_K_M.gguf",
                    help="Path to a .gguf (llamacpp backend) or a HF model id (vllm backend)")
    ap.add_argument("--download-dir", default=None, help="vLLM backend only")
    ap.add_argument("--k", type=int, default=5, help="Number of passages to retrieve")
    args = ap.parse_args()

    print("Loading retriever ...")
    retriever = PassageRetriever(args.index)

    backend = os.environ.get("SELFRAG_BACKEND", "llamacpp")
    print(f"Loading Self-RAG generator (backend: {backend}) ...")
    generator = SelfRAGGenerator(args.model, download_dir=args.download_dir)

    result = self_rag_answer(args.query, generator, retriever, k=args.k)
    print_trace(result)


if __name__ == "__main__":
    main()
