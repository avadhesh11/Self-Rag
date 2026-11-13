"""Unified EvoRAG & RAG Comparison Demonstration CLI.

Supports four primary execution modes:
  1. normal      : Normal RAG baseline (Gemini API)
  2. selfrag     : Original Self-RAG baseline (selfrag_llama2_7b via llama.cpp GGUF)
  3. evorag      : EvoRAG with Experience Memory (GGUF backend + Three-Path Routing)
  4. evorag-mlx  : Fine-Tuned EvoRAG with Experience Memory (Apple Silicon MLX QLoRA)

Additional utility modes:
  - both         : Side-by-side comparison of Normal RAG vs Self-RAG
  - ollama       : Local model served via Ollama

Usage examples:
  # Interactive mode selection menu:
  python demo.py

  # Direct mode execution with query:
  python demo.py --mode normal "What is write-ahead logging in database systems?"
  python demo.py --mode selfrag "What is a B-tree index in PostgreSQL?"
  python demo.py --mode evorag "What is physical data independence?"
  python demo.py --mode evorag-mlx "What is the purpose of a buffer manager?"

  # Verbose reflection critique trace:
  python demo.py --mode evorag-mlx -v "What are ACID properties?"
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure project root is in sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70


def _detect_default_index() -> str:
    """Choose the most appropriate default document index directory."""
    dbms_idx = PROJECT_ROOT / "index" / "dbms_sql_v1"
    base_idx = PROJECT_ROOT / "index"
    if dbms_idx.exists() and (dbms_idx / "faiss.index").exists():
        return str(dbms_idx)
    return str(base_idx)


def _detect_default_memory_index() -> str:
    """Choose default Experience Memory FAISS index directory."""
    if (PROJECT_ROOT / "data" / "experience_faiss.index").exists():
        return "data"
    return "data/experience_index"


def _extract_citations_from_evidence(evidence: List[dict]) -> List[str]:
    """Extract formatted, unique citations from evidence passage dictionaries."""
    citations: List[str] = []
    seen = set()
    for p in evidence:
        src = p.get("source") or p.get("source_id") or "Unknown"
        pg = p.get("page")
        cid = p.get("chunk_id")
        key = (src, pg, cid)
        if key not in seen:
            seen.add(key)
            loc_parts = []
            if pg is not None:
                loc_parts.append(f"page {pg}")
            if cid is not None:
                loc_parts.append(f"chunk {cid}")
            loc = f" ({', '.join(loc_parts)})" if loc_parts else ""
            citations.append(f"{src}{loc}")
    return citations


def _extract_citations_from_selfrag(selfrag_res: Any) -> List[str]:
    """Extract formatted citations from a SelfRAGResult."""
    citations: List[str] = []
    seen = set()
    for seg in getattr(selfrag_res, "segments", []):
        cand = getattr(seg, "selected", None)
        if cand and hasattr(cand, "passage"):
            p = cand.passage
            src = p.get("source", "Unknown")
            pg = p.get("page")
            cid = p.get("chunk_id")
            key = (src, pg, cid)
            if key not in seen:
                seen.add(key)
                loc_parts = []
                if pg is not None:
                    loc_parts.append(f"page {pg}")
                if cid is not None:
                    loc_parts.append(f"chunk {cid}")
                loc = f" ({', '.join(loc_parts)})" if loc_parts else ""
                citations.append(f"{src}{loc}")
    return citations


def _print_evidence_box(passages: List[dict], max_items: int = 5) -> None:
    """Print a clean summary of retrieved document and experience chunks."""
    print(SUB_DIVIDER)
    shown = min(len(passages), max_items)
    print(f"RETRIEVED EVIDENCE (Showing top {shown} of {len(passages)} passages):")
    if not passages:
        print("  (No external documents retrieved - answered directly from parametric memory)")
        return
    for i, p in enumerate(passages[:max_items], 1):
        src = p.get("source", "Unknown")
        pg = p.get("page", "?")
        cid = p.get("chunk_id", "?")
        stype = p.get("source_type", "document")
        score = p.get("retrieval_score") or p.get("score")
        score_str = f" [score: {score:.3f}]" if score is not None else ""
        type_str = " [Experience Memory]" if stype == "experience_memory" else ""
        text = str(p.get("text", "")).replace("\n", " ").strip()
        preview = (text[:150] + "...") if len(text) > 150 else text
        print(f"  [{i}] {src} (page {pg}, chunk {cid}){type_str}{score_str}")
        print(f"      \"{preview}\"")


def _print_routing_box(result: Any) -> None:
    """Display Experience Memory routing decisions and provenance."""
    print(SUB_DIVIDER)
    print("ROUTING & EXPERIENCE MEMORY:")
    mode = getattr(result, "response_mode", "unknown")
    if mode == "direct_reuse":
        print("  Routing Path   : Path A (Direct Reuse)")
        print(f"  Experience ID  : {result.experience_id}")
        if result.similarity_score is not None:
            print(f"  Similarity     : {result.similarity_score:.4f} (Exact normalized match)")
        print("  LLM Inference  : Skipped (0 generator tokens - instant verified retrieval)")
    elif mode == "experience_assisted":
        print("  Routing Path   : Path B (Experience-Assisted Self-RAG)")
        if result.similarity_score is not None:
            print(f"  Similarity     : {result.similarity_score:.4f} (Meets >= 0.85 threshold)")
        exp_ids = getattr(result, "memory_experience_ids", [])
        print(f"  Injected Memory: {len(exp_ids)} verified experience(s) -> {exp_ids}")
        doc_count = len(getattr(result, "document_evidence", []))
        mem_count = len(getattr(result, "experience_evidence", []))
        print(f"  Evidence Mixed : {doc_count} document chunk(s) + {mem_count} memory chunk(s)")
    else:
        print("  Routing Path   : Path C (Normal Self-RAG Fallback)")
        fallback = getattr(result, "fallback_reason", None)
        if fallback:
            print(f"  Fallback Reason: {fallback}")
        elif getattr(result, "similarity_score", None) is not None:
            print(f"  Top Candidate  : Similarity {result.similarity_score:.4f} (Below validation threshold)")
        else:
            print("  Fallback Reason: No candidate experiences found in memory.")
        doc_count = len(getattr(result, "document_evidence", []))
        print(f"  Evidence Used  : {doc_count} document chunk(s)")


def _print_reflection_box(selfrag_res: Any, verbose: bool = False) -> None:
    """Display Self-RAG reflection token decisions and segment critiques."""
    if selfrag_res is None:
        return
    print(SUB_DIVIDER)
    print("SELF-RAG REFLECTION & GENERATION TRACE:")
    if getattr(selfrag_res, "retrieved", False):
        dec = getattr(selfrag_res, "decision", "[Retrieval]")
        print(f"  Retrieval Probe: {dec} -> External evidence required")
    elif getattr(selfrag_res, "decision", None) is None:
        print("  Retrieval Probe: (Direct answer emitted without retrieval)")
    else:
        print(f"  Retrieval Probe: {selfrag_res.decision} -> Self-contained parametric answer")

    segments = getattr(selfrag_res, "segments", [])
    if segments:
        print(f"  Output Segments: {len(segments)} segment(s) generated")
        for seg in segments:
            reused = " (reusing evidence in context)" if getattr(seg, "reused_evidence", False) else ""
            cands = getattr(seg, "candidates", [])
            print(f"  - Segment {seg.index}: {seg.decision or 'End'}{reused} -> {len(cands)} candidate(s)")
            if verbose:
                for c in cands:
                    tag = " -> SELECTED" if c is getattr(seg, "selected", None) else ""
                    p = getattr(c, "passage", {})
                    src = p.get("source", "?")
                    pg = p.get("page", "?")
                    print(f"      * [{src} p.{pg}]: ISREL={c.isrel} ISSUP={c.issup} ISUSE={c.isuse} (score={c.score:.2f}){tag}")


def _print_unified_result(
    title: str,
    mode: str,
    backend_info: str,
    model_info: str,
    query: str,
    latency_ms: float,
    answer: str,
    citations: List[str],
    evidence: Optional[List[dict]] = None,
    evorag_result: Any = None,
    selfrag_result: Any = None,
    verbose: bool = False,
) -> None:
    """Standardized output display for all demonstration modes."""
    print("\n" + DIVIDER)
    print(f"                 {title.upper()}")
    print(DIVIDER)
    print(f"Mode           : {mode}")
    print(f"Backend        : {backend_info}")
    print(f"Model          : {model_info}")
    print(f"Query          : {query}")
    print(f"Latency        : {latency_ms:,.1f} ms ({latency_ms / 1000:.2f}s)")

    if evorag_result is not None:
        _print_routing_box(evorag_result)

    if evidence:
        _print_evidence_box(evidence)

    sr_target = selfrag_result
    if sr_target is None and evorag_result is not None:
        sr_target = getattr(evorag_result, "self_rag_result", None)
    if sr_target is not None:
        _print_reflection_box(sr_target, verbose=verbose)

    print(SUB_DIVIDER)
    print("FINAL ANSWER:")
    print(answer.strip() if answer else "(Empty answer generated)")

    print(SUB_DIVIDER)
    print("CITATIONS & SOURCES:")
    if citations:
        for cit in citations:
            print(f"  - {cit}")
    else:
        print("  (None emitted or parametric generation)")
    print(DIVIDER + "\n")


def check_and_dispatch_environment(mode: str, args: argparse.Namespace) -> Optional[int]:
    """Handle cross-environment dispatching between .venv and .venv-mlx.

    Returns process exit code if dispatched, or None if current environment should proceed.
    """
    is_mlx_mode = (mode == "evorag-mlx")

    # 1. Dispatching to MLX environment if needed
    if is_mlx_mode:
        has_mlx = False
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
            has_mlx = True
        except ImportError:
            has_mlx = False

        if not has_mlx:
            mlx_python = PROJECT_ROOT / ".venv-mlx" / "bin" / "python"
            if not args.no_reexec and os.environ.get("EVORAG_REEXEC") != "1" and mlx_python.exists():
                print(f"[Launcher] Mode 'evorag-mlx' requires Apple Silicon MLX environment.")
                print(f"[Launcher] Dispatching to {mlx_python} ...\n")
                env = dict(os.environ)
                env["EVORAG_REEXEC"] = "1"
                cmd = [str(mlx_python), sys.argv[0]] + sys.argv[1:]
                if sys.platform == "darwin":
                    cmd = ["arch", "-arm64"] + cmd
                proc = subprocess.run(cmd, env=env)
                return proc.returncode
            else:
                print("\n" + DIVIDER)
                print("[ENVIRONMENT ERROR] Mode 'evorag-mlx' requires Apple Silicon MLX.")
                print(DIVIDER)
                print(f"Current Python: {sys.executable}")
                print("The 'mlx' and 'mlx_lm' packages are not installed in this environment.")
                print("\nTo launch this mode:")
                print(f"  .venv-mlx/bin/python demo.py {' '.join(sys.argv[1:])}")
                print("Or activate the environment:")
                print("  source .venv-mlx/bin/activate")
                print("  python demo.py --mode evorag-mlx ...")
                print(DIVIDER + "\n")
                return 1

    # 2. Dispatching to GGUF environment if needed
    if mode in ("selfrag", "self", "evorag", "both"):
        backend = os.environ.get("SELFRAG_BACKEND", "llamacpp")
        if backend == "llamacpp":
            has_llamacpp = False
            try:
                import llama_cpp  # noqa: F401
                has_llamacpp = True
            except ImportError:
                has_llamacpp = False

            if not has_llamacpp:
                venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
                if not args.no_reexec and os.environ.get("EVORAG_REEXEC") != "1" and venv_python.exists():
                    print(f"[Launcher] Mode '{mode}' requires GGUF/llama.cpp environment.")
                    print(f"[Launcher] Dispatching to {venv_python} ...\n")
                    env = dict(os.environ)
                    env["EVORAG_REEXEC"] = "1"
                    cmd = [str(venv_python), sys.argv[0]] + sys.argv[1:]
                    if sys.platform == "darwin":
                        cmd = ["arch", "-arm64"] + cmd
                    proc = subprocess.run(cmd, env=env)
                    return proc.returncode
                else:
                    print("\n" + DIVIDER)
                    print(f"[ENVIRONMENT ERROR] Mode '{mode}' requires llama-cpp-python.")
                    print(DIVIDER)
                    print(f"Current Python: {sys.executable}")
                    print("The 'llama_cpp' package is not installed in this environment.")
                    print("\nTo launch this mode:")
                    print(f"  .venv/bin/python demo.py {' '.join(sys.argv[1:])}")
                    print("Or activate the production environment:")
                    print("  source .venv/bin/activate")
                    print(f"  python demo.py --mode {mode} ...")
                    print(DIVIDER + "\n")
                    return 1

    return None


def _pick_mode_interactive() -> str:
    """Show interactive mode selection menu and return selected mode."""
    print("\n" + DIVIDER)
    print("                     EvoRAG UNIFIED DEMO")
    print(DIVIDER)
    print("  1. Normal RAG           (Gemini Retrieval-Augmented Generation)")
    print("  2. Self-RAG (GGUF)      (Original selfrag_llama2_7b via llama.cpp)")
    print("  3. EvoRAG (GGUF)        (Original GGUF + Experience Memory)")
    print("  4. EvoRAG (MLX QLoRA)   (Fine-Tuned MLX 7B + Experience Memory)")
    print("  5. Compare Both         (Normal RAG vs Self-RAG GGUF)")
    print("  6. Ollama RAG           (Local Ollama model)")
    print(DIVIDER)
    choice = input("Select mode [1/2/3/4/5/6] (default 2): ").strip() or "2"
    mapping = {
        "1": "normal",
        "2": "selfrag",
        "3": "evorag",
        "4": "evorag-mlx",
        "5": "both",
        "6": "ollama",
    }
    return mapping.get(choice, "selfrag")


def run_normal_mode(query: str, args: argparse.Namespace) -> int:
    """Execute Normal RAG using Gemini API."""
    from normal_rag.generator import GeminiGenerator
    from retriever.retriever import PassageRetriever

    if not Path(args.index).exists():
        print(f"[INDEX ERROR] Document index directory not found: {args.index}")
        return 1

    print(f"Initializing PassageRetriever from '{args.index}' ...")
    retriever = PassageRetriever(args.index)

    print("Initializing GeminiGenerator ...")
    try:
        generator = GeminiGenerator()
    except EnvironmentError as e:
        print(f"\n[CONFIGURATION ERROR] {e}")
        print("Please configure GEMINI_API_KEY to run Normal RAG.")
        return 1

    print("Retrieving passages and generating answer ...")
    t0 = time.perf_counter()
    passages = retriever.search(query, k=args.k)
    answer = generator.generate(query, passages)
    latency_ms = (time.perf_counter() - t0) * 1000

    citations = _extract_citations_from_evidence(passages)
    _print_unified_result(
        title="Normal RAG Execution Result",
        mode="Normal RAG (Gemini API)",
        backend_info="Google Gemini API (google-genai)",
        model_info=generator._model,
        query=query,
        latency_ms=latency_ms,
        answer=answer,
        citations=citations,
        evidence=passages,
        verbose=args.verbose,
    )
    return 0


def run_selfrag_mode(query: str, args: argparse.Namespace) -> int:
    """Execute Self-RAG baseline using llama.cpp GGUF."""
    from retriever.retriever import PassageRetriever
    from selfrag.generator import SelfRAGGenerator
    from selfrag.pipeline import self_rag_answer

    if not Path(args.index).exists():
        print(f"[INDEX ERROR] Document index directory not found: {args.index}")
        return 1

    if not Path(args.model).exists() and os.environ.get("SELFRAG_BACKEND") != "vllm":
        print(f"[MODEL ERROR] GGUF model file not found at: {args.model}")
        return 1

    print(f"Initializing PassageRetriever from '{args.index}' ...")
    retriever = PassageRetriever(args.index)

    backend = os.environ.get("SELFRAG_BACKEND", "llamacpp")
    print(f"Loading Self-RAG generator (model: {args.model}, backend: {backend}) ...")
    generator = SelfRAGGenerator(args.model)

    print("Executing Self-RAG inference ...")
    t0 = time.perf_counter()
    res = self_rag_answer(query, generator, retriever, k=args.k, max_segments=args.max_segments)
    latency_ms = (time.perf_counter() - t0) * 1000

    citations = _extract_citations_from_selfrag(res)
    evidence = [c.passage for seg in res.segments for c in seg.candidates if hasattr(c, "passage")]

    _print_unified_result(
        title="Self-RAG Execution Result",
        mode="Self-RAG (Original GGUF)",
        backend_info=f"{backend} (Q4_K_M)",
        model_info=args.model,
        query=query,
        latency_ms=latency_ms,
        answer=res.final_answer,
        citations=citations,
        evidence=evidence,
        selfrag_result=res,
        verbose=args.verbose,
    )
    return 0


def run_evorag_mode(query: str, args: argparse.Namespace) -> int:
    """Execute EvoRAG using GGUF Self-RAG + Experience Memory."""
    from evorag.controller import EvoRAGController
    from evorag.memory.index_builder import ExperienceIndexBuilder
    from evorag.memory.retriever import ExperienceRetriever
    from evorag.memory.store import ExperienceStore
    from evorag.memory.validator import ExperienceValidator
    from retriever.retriever import PassageRetriever
    from selfrag.generator import SelfRAGGenerator

    if not Path(args.index).exists():
        print(f"[INDEX ERROR] Document index directory not found: {args.index}")
        return 1

    if not Path(args.model).exists() and os.environ.get("SELFRAG_BACKEND") != "vllm":
        print(f"[MODEL ERROR] GGUF model file not found at: {args.model}")
        return 1

    print(f"Opening Experience Memory database: {args.memory_db} ...")
    store = ExperienceStore(args.memory_db)
    mem_retriever = ExperienceRetriever(args.memory_index, store=store)
    validator = ExperienceValidator(similarity_threshold=args.threshold)
    index_builder = ExperienceIndexBuilder(index_dir=args.memory_index)

    print(f"Initializing PassageRetriever from '{args.index}' ...")
    doc_retriever = PassageRetriever(args.index)

    backend = os.environ.get("SELFRAG_BACKEND", "llamacpp")
    print(f"Loading Self-RAG generator (model: {args.model}, backend: {backend}) ...")
    self_generator = SelfRAGGenerator(args.model)

    controller = EvoRAGController(
        store=store,
        memory_retriever=mem_retriever,
        validator=validator,
        generator=self_generator,
        retriever=doc_retriever,
        doc_index_dir=args.index,
        index_builder=index_builder,
        k=args.k,
        max_segments=args.max_segments,
    )

    print("Evaluating question through EvoRAG Three-Path Routing ...")
    t0 = time.perf_counter()
    try:
        result = controller.answer(query)
    except Exception as e:
        print(f"\n[INFERENCE ERROR] {e}")
        if args.verbose:
            traceback.print_exc()
        return 1
    finally:
        store.close()
    latency_ms = (time.perf_counter() - t0) * 1000

    citations = _extract_citations_from_evidence(result.evidence)
    _print_unified_result(
        title="EvoRAG Execution Result (GGUF)",
        mode="EvoRAG (GGUF + Experience Memory)",
        backend_info=f"{backend} (Q4_K_M) + SQLite/FAISS Memory",
        model_info=args.model,
        query=query,
        latency_ms=latency_ms,
        answer=result.answer,
        citations=citations,
        evidence=result.evidence,
        evorag_result=result,
        verbose=args.verbose,
    )
    return 0


def run_evorag_mlx_mode(query: str, args: argparse.Namespace) -> int:
    """Execute EvoRAG using fine-tuned MLX QLoRA + Experience Memory."""
    from evorag.controller import EvoRAGController
    from evorag.memory.index_builder import ExperienceIndexBuilder
    from evorag.memory.retriever import ExperienceRetriever
    from evorag.memory.store import ExperienceStore
    from evorag.memory.validator import ExperienceValidator
    from evorag.mlx_generator import MLXSelfRAGGenerator
    from retriever.retriever import PassageRetriever

    if not Path(args.index).exists():
        print(f"[INDEX ERROR] Document index directory not found: {args.index}")
        return 1

    if not Path(args.mlx_model).exists():
        print(f"[MODEL ERROR] MLX base model directory not found at: {args.mlx_model}")
        return 1

    if args.mlx_adapter and not Path(args.mlx_adapter).exists():
        print(f"[ADAPTER ERROR] MLX adapter directory not found at: {args.mlx_adapter}")
        return 1

    print(f"Opening Experience Memory database: {args.memory_db} ...")
    store = ExperienceStore(args.memory_db)
    mem_retriever = ExperienceRetriever(args.memory_index, store=store)
    validator = ExperienceValidator(similarity_threshold=args.threshold)
    index_builder = ExperienceIndexBuilder(index_dir=args.memory_index)

    print(f"Initializing PassageRetriever from '{args.index}' ...")
    doc_retriever = PassageRetriever(args.index)

    print(f"Loading Fine-Tuned MLX Generator (Metal GPU) ...")
    print(f"  Base Model: {args.mlx_model}")
    print(f"  Adapter   : {args.mlx_adapter or '(none)'}")
    t0_load = time.perf_counter()
    mlx_gen = MLXSelfRAGGenerator(
        model_path=args.mlx_model,
        adapter_path=args.mlx_adapter,
        max_tokens=args.max_tokens,
    )
    load_time = time.perf_counter() - t0_load
    print(f"  MLX Model ready in {load_time:.2f}s (Memory: {mlx_gen.memory_after_load_gb:.2f} GB)")

    controller = EvoRAGController(
        store=store,
        memory_retriever=mem_retriever,
        validator=validator,
        generator=mlx_gen,
        retriever=doc_retriever,
        doc_index_dir=args.index,
        index_builder=index_builder,
        k=args.k,
        max_segments=args.max_segments,
    )

    print("Evaluating question through EvoRAG Three-Path Routing ...")
    t0 = time.perf_counter()
    try:
        result = controller.answer(query)
    except Exception as e:
        print(f"\n[INFERENCE ERROR] {e}")
        if args.verbose:
            traceback.print_exc()
        return 1
    finally:
        store.close()
    latency_ms = (time.perf_counter() - t0) * 1000

    citations = _extract_citations_from_evidence(result.evidence)
    adapter_str = f" + Adapter: {args.mlx_adapter}" if args.mlx_adapter else ""
    _print_unified_result(
        title="Fine-Tuned EvoRAG Execution Result (MLX)",
        mode="Fine-Tuned EvoRAG (MLX QLoRA + Experience Memory)",
        backend_info="Apple Silicon MLX (Metal GPU) + SQLite/FAISS Memory",
        model_info=f"{args.mlx_model}{adapter_str}",
        query=query,
        latency_ms=latency_ms,
        answer=result.answer,
        citations=citations,
        evidence=result.evidence,
        evorag_result=result,
        verbose=args.verbose,
    )
    return 0


def run_both_mode(query: str, args: argparse.Namespace) -> int:
    """Execute side-by-side comparison of Normal RAG vs Self-RAG."""
    from comparison.runner import print_comparison, run_rag
    from normal_rag.generator import GeminiGenerator
    from retriever.retriever import PassageRetriever
    from selfrag.generator import SelfRAGGenerator

    print(f"Initializing PassageRetriever from '{args.index}' ...")
    retriever = PassageRetriever(args.index)

    print("Initializing Gemini generator ...")
    normal_generator = GeminiGenerator()

    backend = os.environ.get("SELFRAG_BACKEND", "llamacpp")
    print(f"Loading Self-RAG generator (backend: {backend}) ...")
    self_generator = SelfRAGGenerator(args.model)

    result = run_rag(
        query,
        "both",
        normal_generator=normal_generator,
        self_generator=self_generator,
        retriever=retriever,
        k=args.k,
        max_segments=args.max_segments,
    )
    print_comparison(result)
    return 0


def run_ollama_mode(query: str, args: argparse.Namespace) -> int:
    """Execute local Ollama RAG."""
    from comparison.runner import print_ollama_rag, run_rag
    from normal_rag.ollama_generator import OllamaGenerator
    from retriever.retriever import PassageRetriever

    retriever = PassageRetriever(args.index)
    ollama_generator = OllamaGenerator(model=args.ollama_model, host=args.ollama_host)
    result = run_rag(
        query,
        "ollama",
        ollama_generator=ollama_generator,
        retriever=retriever,
        k=args.k,
    )
    print_ollama_rag(result.ollama, model_name=ollama_generator.name)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    default_index = _detect_default_index()
    default_memory_index = _detect_default_memory_index()

    parser = argparse.ArgumentParser(
        description="Unified EvoRAG & RAG Demonstration CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Question to answer (prompted interactively if omitted)",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "selfrag", "self", "evorag", "evorag-mlx", "both", "ollama"],
        default=None,
        help="RAG mode: normal | selfrag | evorag | evorag-mlx | both | ollama",
    )
    parser.add_argument(
        "--index",
        default=os.environ.get("INDEX_DIR", default_index),
        help="Document FAISS index directory containing faiss.index and meta.json",
    )
    parser.add_argument(
        "--memory-db",
        default="data/experience_memory.db",
        help="Path to SQLite Experience Memory database",
    )
    parser.add_argument(
        "--memory-index",
        default=default_memory_index,
        help="Path to FAISS Experience Memory index directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for Experience Memory validation",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SELF_RAG_MODEL_PATH", "models/selfrag_llama2_7b.Q4_K_M.gguf"),
        help="Path to Self-RAG GGUF model (for selfrag, evorag, both)",
    )
    parser.add_argument(
        "--mlx-model",
        default="models/mlx/selfrag_llama2_7b_4bit",
        help="Path to MLX 4-bit base model directory (for evorag-mlx)",
    )
    parser.add_argument(
        "--mlx-adapter",
        default="training/qlora_dbms/adapters/full_dbms_v1",
        help="Path to fine-tuned MLX QLoRA adapter directory (for evorag-mlx)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of document passages to retrieve",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=3,
        help="Max generation segments for Self-RAG control loop",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=250,
        help="Max output tokens for MLX generation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Display detailed reflection token critiques and candidate scores",
    )
    parser.add_argument(
        "--no-reexec",
        action="store_true",
        help="Disable automatic cross-environment launcher re-execution",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", "llama3:8b"),
        help="Ollama model tag for --mode ollama",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama server URL for --mode ollama",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Determine mode
    mode = args.mode or _pick_mode_interactive()
    if mode == "self":
        mode = "selfrag"

    # Environment check & dispatch
    dispatch_code = check_and_dispatch_environment(mode, args)
    if dispatch_code is not None:
        return dispatch_code

    # Determine query
    query = args.query
    if not query:
        default_q = "What is the purpose of a database buffer pool manager?"
        try:
            user_input = input(f"\nQuestion [Press Enter for default: '{default_q}']: ").strip()
            query = user_input or default_q
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return 0

    if not query.strip():
        print("Empty question provided. Exiting.")
        return 1

    # Route to handler
    if mode == "normal":
        return run_normal_mode(query, args)
    elif mode == "selfrag":
        return run_selfrag_mode(query, args)
    elif mode == "evorag":
        return run_evorag_mode(query, args)
    elif mode == "evorag-mlx":
        return run_evorag_mlx_mode(query, args)
    elif mode == "both":
        return run_both_mode(query, args)
    elif mode == "ollama":
        return run_ollama_mode(query, args)
    else:
        print(f"Unknown mode: {mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
