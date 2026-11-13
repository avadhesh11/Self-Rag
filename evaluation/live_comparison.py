"""Orchestrator for the Five-System Live Comparison and Terminal Demonstration.

Sequentially executes:
  1. Normal RAG (Gemini API via google-genai)
  2. Original Self-RAG GGUF (llama.cpp)
  3. EvoRAG GGUF (GGUF + Experience Memory)
  4. Original Self-RAG MLX (Base model, no adapter)
  5. Fine-Tuned EvoRAG MLX (Base model + QLoRA adapter + Experience Memory)

Measures and presents latency (total, retrieval, generation, and model load time),
prints a consolidated comparison table, and persists timestamped Markdown and JSON reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "evaluation" / "live_comparisons"

DIVIDER = "=" * 80
SUB_DIVIDER = "-" * 80


def _resolve_interpreters() -> tuple[Path, Path]:
    """Identify the python executables for .venv (GGUF) and .venv-mlx (MLX)."""
    venv_py = PROJECT_ROOT / ".venv" / "bin" / "python"
    mlx_py = PROJECT_ROOT / ".venv-mlx" / "bin" / "python"

    # Fallbacks if running inside one of the environments directly
    current_py = Path(sys.executable)
    if not venv_py.exists():
        venv_py = current_py
    if not mlx_py.exists():
        mlx_py = current_py

    return venv_py, mlx_py


def _format_ms(val: Optional[float], suffix: str = " ms") -> str:
    if val is None:
        return "N/A"
    return f"{val:,.1f}{suffix}"


def format_comparison_table(results: List[Dict[str, Any]]) -> str:
    """Build a clean Markdown/text consolidated comparison table."""
    headers = [
        "System",
        "Query Latency",
        "Retrieval Latency",
        "Generation Latency",
        "Model Load Time",
        "Answer Status",
        "Route / Decision",
    ]

    rows = []
    for r in results:
        sys_name = r.get("system_name", r.get("system_id", "Unknown"))
        q_lat = _format_ms(r.get("total_query_latency_ms"))
        r_lat = _format_ms(r.get("retrieval_latency_ms"))
        g_lat = _format_ms(r.get("generation_latency_ms"))
        m_load = _format_ms(r.get("model_load_time_ms"))
        if r.get("system_id") == "normal":
            m_load = "N/A (API)"
        status = r.get("answer_status", "Unknown")

        route = r.get("routing_path") or r.get("retrieval_decision") or "N/A"
        if r.get("memory_hit") and r.get("routing_path"):
            route = r.get("routing_path")

        rows.append([sys_name, q_lat, r_lat, g_lat, m_load, status, route])

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"

    body_lines = []
    for row in rows:
        line = "| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |"
        body_lines.append(line)

    return "\n".join([header_line, separator_line] + body_lines)


def print_system_result_box(result: Dict[str, Any], index: int, total: int) -> None:
    """Print an individual system execution card."""
    sys_name = result.get("system_name", result.get("system_id", "Unknown"))
    print("\n" + DIVIDER)
    print(f"[{index}/{total}] SYSTEM: {sys_name.upper()}")
    print(DIVIDER)
    print(f"Backend          : {result.get('backend', 'Unknown')}")
    print(f"Model            : {result.get('model', 'Unknown')}")
    print(f"Answer Status    : {result.get('answer_status', 'Unknown')}")
    print(f"Total Query Time : {_format_ms(result.get('total_query_latency_ms'))}")
    print(f"Retrieval Time   : {_format_ms(result.get('retrieval_latency_ms'))}")
    print(f"Generation Time  : {_format_ms(result.get('generation_latency_ms'))}")
    print(f"Model Load Time  : {_format_ms(result.get('model_load_time_ms'))} (cold-start)")

    if result.get("routing_path"):
        print(f"EvoRAG Route     : {result.get('routing_path')}")
        if result.get("memory_match_score") is not None:
            print(f"Memory Candidate : Match score {result.get('memory_match_score'):.4f} (ID: {result.get('memory_match_id')})")

    if result.get("retrieval_decision"):
        print(f"Retrieval Probe  : {result.get('retrieval_decision')}")
    print(f"Passages Used    : {result.get('retrieved_passage_count', 0)}")

    print(SUB_DIVIDER)
    print("ANSWER:")
    answer = result.get("answer", "")
    print(answer.strip() if answer else f"(No answer produced - {result.get('answer_status')})")

    citations = result.get("citations", [])
    if citations:
        print(SUB_DIVIDER)
        print("CITATIONS:")
        for cit in citations:
            print(f"  - {cit}")
    print(DIVIDER)


def run_live_comparison(
    question: str,
    category: str = "pdf",
    index_dir: Optional[str] = None,
    memory_db: str = "data/experience_memory.db",
    memory_index: str = "data",
    threshold: float = 0.85,
    model: str = "models/selfrag_llama2_7b.Q4_K_M.gguf",
    mlx_model: str = "models/mlx/selfrag_llama2_7b_4bit",
    mlx_adapter: str = "training/qlora_dbms/adapters/full_dbms_v1",
    k: int = 5,
    max_segments: int = 3,
    max_tokens: int = 250,
    save_report: bool = True,
) -> Dict[str, Any]:
    """Execute all five systems sequentially and return comprehensive results dictionary."""
    if index_dir is None:
        dbms_idx = PROJECT_ROOT / "index" / "dbms_sql_v1"
        index_dir = str(dbms_idx) if dbms_idx.exists() else "index"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

    venv_py, mlx_py = _resolve_interpreters()

    system_configs = [
        {"id": "normal", "name": "Normal RAG (Gemini)", "py": venv_py},
        {"id": "selfrag_gguf", "name": "Original Self-RAG (GGUF)", "py": venv_py},
        {"id": "evorag_gguf", "name": "EvoRAG (GGUF + Experience Memory)", "py": venv_py},
        {"id": "selfrag_mlx", "name": "Original Self-RAG (MLX Base)", "py": mlx_py},
        {"id": "evorag_mlx", "name": "Fine-Tuned EvoRAG (MLX QLoRA)", "py": mlx_py},
    ]

    print("\n" + DIVIDER)
    print("           STARTING FIVE-SYSTEM SEQUENTIAL LIVE COMPARISON")
    print(DIVIDER)
    print(f"Question Category : {category.upper()}")
    print(f"Evaluated Question: \"{question}\"")
    print(f"Corpus Index      : {index_dir}")
    print(f"Timestamp         : {timestamp_str}")
    print("Systems to evaluate sequentially:")
    for i, s in enumerate(system_configs, 1):
        print(f"  {i}. {s['name']} (via {s['py'].parent.parent.name})")
    print(DIVIDER + "\n")

    results: List[Dict[str, Any]] = []

    for idx, sys_cfg in enumerate(system_configs, 1):
        sys_id = sys_cfg["id"]
        sys_name = sys_cfg["name"]
        py_bin = sys_cfg["py"]

        print(f"\n>>> Running [{idx}/5] {sys_name} ...")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            tmp_json_path = tmp_file.name

        cmd = [
            str(py_bin),
            "-m", "evaluation.live_comparison_worker",
            "--system", sys_id,
            "--question", question,
            "--category", category,
            "--index", index_dir,
            "--memory-db", memory_db,
            "--memory-index", memory_index,
            "--threshold", str(threshold),
            "--model", model,
            "--mlx-model", mlx_model,
            "--mlx-adapter", mlx_adapter,
            "--k", str(k),
            "--max-segments", str(max_segments),
            "--max-tokens", str(max_tokens),
            "--output-json", tmp_json_path,
        ]

        if sys.platform == "darwin":
            cmd = ["arch", "-arm64"] + cmd

        t_proc_start = time.perf_counter()
        try:
            env = dict(os.environ)
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            res = subprocess.run(cmd, env=env, capture_output=True, text=True)
            proc_time_ms = (time.perf_counter() - t_proc_start) * 1000

            if Path(tmp_json_path).exists() and Path(tmp_json_path).stat().st_size > 0:
                record = json.loads(Path(tmp_json_path).read_text(encoding="utf-8"))
            else:
                record = {
                    "system_id": sys_id,
                    "system_name": sys_name,
                    "success": False,
                    "error": res.stderr.strip() or f"Process exited with code {res.returncode}",
                    "answer_status": f"Process Error (Exit {res.returncode})",
                    "total_query_latency_ms": None,
                    "retrieval_latency_ms": None,
                    "generation_latency_ms": None,
                    "model_load_time_ms": None,
                }
        except Exception as e:
            record = {
                "system_id": sys_id,
                "system_name": sys_name,
                "success": False,
                "error": str(e),
                "answer_status": f"Launcher Exception: {e}",
                "total_query_latency_ms": None,
                "retrieval_latency_ms": None,
                "generation_latency_ms": None,
                "model_load_time_ms": None,
            }
        finally:
            if Path(tmp_json_path).exists():
                try:
                    os.unlink(tmp_json_path)
                except OSError:
                    pass

        results.append(record)
        print_system_result_box(record, idx, len(system_configs))

    # Consolidated Table
    print("\n" + DIVIDER)
    print("                     CONSOLIDATED COMPARISON TABLE")
    print(DIVIDER)
    table_str = format_comparison_table(results)
    print(table_str)
    print(DIVIDER + "\n")

    session_data = {
        "timestamp": timestamp_str,
        "question": question,
        "category": category,
        "index_dir": index_dir,
        "results": results,
    }

    if save_report:
        json_path = REPORTS_DIR / f"comparison_{timestamp_str}.json"
        md_path = REPORTS_DIR / f"comparison_{timestamp_str}.md"

        json_path.write_text(json.dumps(session_data, indent=2), encoding="utf-8")

        # Markdown Report
        md_content = f"""# EvoRAG Five-System Live Comparison Report

- **Timestamp**: `{timestamp_str}`
- **Question Category**: `{category.upper()}`
- **Question**: **"{question}"**
- **Corpus Index**: `{index_dir}`

---

## 1. Consolidated Latency & Status Matrix

{table_str}

---

## 2. Detailed Per-System Responses & Metadata

"""
        for r in results:
            sys_name = r.get("system_name", r.get("system_id"))
            ans = r.get("answer", "")
            cits = r.get("citations", [])
            cits_str = "\n".join(f"- {c}" for c in cits) if cits else "(None emitted)"
            md_content += f"""### {sys_name}
- **Backend**: `{r.get('backend')}`
- **Model**: `{r.get('model')}`
- **Status**: `{r.get('answer_status')}`
- **Total Latency**: `{_format_ms(r.get('total_query_latency_ms'))}`
- **Retrieval Latency**: `{_format_ms(r.get('retrieval_latency_ms'))}`
- **Generation Latency**: `{_format_ms(r.get('generation_latency_ms'))}`
- **Model Load Time**: `{_format_ms(r.get('model_load_time_ms'))}`
- **Retrieval Decision**: `{r.get('retrieval_decision')}`
- **Routing Path**: `{r.get('routing_path')}`
- **Memory Hit**: `{r.get('memory_hit')}` (Score: `{r.get('memory_match_score')}`)

**Answer**:
> {ans if ans else f"*(No answer generated: {r.get('answer_status')})*"}

**Citations**:
{cits_str}

---
"""
        md_path.write_text(md_content, encoding="utf-8")
        print(f"Saved session JSON  : {json_path}")
        print(f"Saved session Report: {md_path}\n")

    return session_data

