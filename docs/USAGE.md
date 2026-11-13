# EvoRAG User and CLI Guide

## 1. Quick Start

Run the unified demonstration entry point:

```bash
# Interactive mode selection menu:
python demo.py
```

You will see:
```text
======================================================================
                     EvoRAG UNIFIED DEMO
======================================================================
  1. Normal RAG           (Gemini Retrieval-Augmented Generation)
  2. Self-RAG (GGUF)      (Original selfrag_llama2_7b via llama.cpp)
  3. EvoRAG (GGUF)        (Original GGUF + Experience Memory)
  4. EvoRAG (MLX QLoRA)   (Fine-Tuned MLX 7B + Experience Memory)
  5. Compare Both         (Normal RAG vs Self-RAG GGUF)
  6. Ollama RAG           (Local Ollama model)
======================================================================
Select mode [1/2/3/4/5/6] (default 2):
```

---

## 2. Direct Command-Line Execution

### Mode 1: Normal RAG (Gemini API)
```bash
python demo.py --mode normal "What is write-ahead logging in database systems?"
```
*Requires `GEMINI_API_KEY` set in `.env` or environment.*

### Mode 2: Baseline Self-RAG (GGUF via llama.cpp)
```bash
python demo.py --mode selfrag "What is a B-tree index in PostgreSQL?"
```
*Executes the original `models/selfrag_llama2_7b.Q4_K_M.gguf`.*

### Mode 3: EvoRAG with GGUF + Experience Memory
```bash
python demo.py --mode evorag "How does physical data independence protect applications from storage modifications?"
```
*Queries Experience Memory first; if exact match (Path A), returns instantly without running the LLM.*

### Mode 4: Fine-Tuned EvoRAG with Apple Silicon MLX QLoRA
```bash
python demo.py --mode evorag-mlx "What is the purpose of a buffer manager?"
```
*Executes the fine-tuned adapter on Apple Silicon Metal GPU via `.venv-mlx`.*

---

## 3. Five-System Live Comparison Mode (`--compare`)

Task 13 introduces an interactive, sequential comparison harness that runs **exactly one question** through all five systems and outputs a consolidated latency matrix.

> [!NOTE]
> This live comparison is an interactive qualitative inspection tool. The formal, double-verified 32-question evaluation is documented in [`docs/EVALUATION.md`](EVALUATION.md).

### Interactive Mode:
```bash
python demo.py --compare
```
The session interactively prompts for:
1. Question category:
   - `1`: **PDF-related question** (evaluates retrieval over `index/dbms_sql_v1`).
   - `2`: **General knowledge question** (tests adaptive parametric `[No Retrieval]` emission).
2. The question string (or press Enter for sensible defaults).
3. Executes all 5 systems sequentially and prints a consolidated comparison table.
4. Asks whether you would like to run another comparison question.

### Non-Interactive Direct Commands:
```bash
# PDF-related question:
python demo.py --compare --question-type pdf --question "What is write-ahead logging (WAL) in database systems?"

# General knowledge question:
python demo.py --compare --question-type general --question "What is the capital of France?"
```

### Consolidated Table Output Example:
```text
================================================================================
                     CONSOLIDATED COMPARISON TABLE
================================================================================
| System                            | Query Latency | Retrieval Latency | Generation Latency | Model Load Time | Answer Status | Route / Decision                         |
|-----------------------------------|---------------|-------------------|--------------------|-----------------|---------------|------------------------------------------|
| Normal RAG (Gemini)               | 1,148.1 ms    | 258.6 ms          | 889.5 ms           | N/A (API)       | Success       | [Retrieval] (Forced by RAG architecture) |
| Original Self-RAG (GGUF)          | 9,947.6 ms    | 395.1 ms          | 9,552.6 ms         | 2,629.0 ms      | Success       | [Retrieval]                              |
| EvoRAG (GGUF + Experience Memory) | 9,583.5 ms    | 41.9 ms           | 9,541.6 ms         | 552.9 ms        | Success       | Path C (Normal Self-RAG Fallback)        |
| Original Self-RAG (MLX Base)      | 2,941.7 ms    | 353.2 ms          | 2,588.4 ms         | 942.0 ms        | Success       | [Retrieval]                              |
| Fine-Tuned EvoRAG (MLX QLoRA)     | 2,417.7 ms    | 5.5 ms            | 2,412.2 ms         | 458.2 ms        | Success       | Path C (Normal Self-RAG Fallback)        |
================================================================================
```
Every comparison session is automatically persisted to `evaluation/live_comparisons/comparison_<timestamp>.json` and `.md`.

---

## 4. Command-Line Options Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `query` | string | `None` (prompts) | Positional question string. |
| `--mode` | string | `None` (interactive) | Execution mode: `normal`, `selfrag`, `evorag`, `evorag-mlx`, `compare`, `both`, `ollama`. |
| `--compare` | flag | `False` | Run live 5-system comparative evaluation across one question. |
| `--question-type` | string | `pdf` | Question category for `--compare`: `pdf` or `general`. |
| `--question` | string | `None` | Question string for non-interactive `--compare` mode. |
| `--index` | string | `index/dbms_sql_v1` | Document FAISS index directory containing `faiss.index` and `meta.json`. |
| `--memory-db` | string | `data/experience_memory.db` | Path to SQLite Experience Memory database. |
| `--memory-index` | string | `data` | Directory containing `experience_faiss.index` and `experience_meta.json`. |
| `--threshold` | float | `0.85` | Cosine similarity threshold for Experience Memory candidate validation. |
| `--model` | string | `models/selfrag_llama2_7b.Q4_K_M.gguf` | Path to GGUF model for `selfrag` and `evorag`. |
| `--mlx-model` | string | `models/mlx/selfrag_llama2_7b_4bit` | Path to MLX 4-bit base model for `evorag-mlx`. |
| `--mlx-adapter` | string | `training/qlora_dbms/adapters/full_dbms_v1` | Path to fine-tuned MLX QLoRA adapter directory. |
| `--k` | int | `5` | Number of document passages to retrieve. |
| `--max-segments` | int | `3` | Maximum generation segments for Self-RAG control loop. |
| `--max-tokens` | int | `250` | Maximum token budget for MLX generation. |
| `--verbose`, `-v` | flag | `False` | Displays per-candidate critique scores (`ISREL`, `ISSUP`, `ISUSE`) and selection traces. |
| `--no-reexec` | flag | `False` | Disables automatic cross-environment subprocess dispatching. |

---

## 4. Understanding the Demonstration Output

EvoRAG outputs a structured report partitioned into operational sections:

```text
======================================================================
                 FINE-TUNED EVORAG EXECUTION RESULT (MLX)
======================================================================
Mode           : Fine-Tuned EvoRAG (MLX QLoRA + Experience Memory)
Backend        : Apple Silicon MLX (Metal GPU) + SQLite/FAISS Memory
Model          : models/mlx/selfrag_llama2_7b_4bit + Adapter: training/qlora_dbms/adapters/full_dbms_v1
Query          : How does physical data independence protect applications from storage modifications?
Latency        : 175.1 ms (0.18s)
----------------------------------------------------------------------
ROUTING & EXPERIENCE MEMORY:
  Routing Path   : Path A (Direct Reuse)
  Experience ID  : 3239b9d1ea3f3c58779bf8da23fdb9d937c5cf2d30acf3465cbd31bf30e312f4
  Similarity     : 1.0000 (Exact normalized match)
  LLM Inference  : Skipped (0 generator tokens - instant verified retrieval)
----------------------------------------------------------------------
RETRIEVED EVIDENCE (Showing top 1 of 1 passages):
  [1] 01_dbms_fundamentals_and_architecture.pdf (page 1, chunk 2)
      "Problems addressed by a DBMS • Redundancy and inconsistency: the same fact should not be updated independently in many files. • Integrity: rules such ..."
----------------------------------------------------------------------
FINAL ANSWER:
Physical data independence insulates the logical schema and application queries from changes in physical storage structures, such as modifying file organization, record layouts, or adding auxiliary indexes. The DBMS query processor and executor adapt to the underlying physical layout without requiring application rewrites.
----------------------------------------------------------------------
CITATIONS & SOURCES:
  - 01_dbms_fundamentals_and_architecture.pdf (page 1, chunk 2)
======================================================================
```

---

## 5. Example Questions by Routing Path

| Expected Path | Example Question | Behavior |
| :--- | :--- | :--- |
| **Path A (Direct Reuse)** | `"How does physical data independence protect applications from storage modifications?"` | Exact match with verified experience; returns stored answer in $< 0.5$s without LLM generation. |
| **Path B (Experience-Assisted)** | `"Explain how a buffer pool manager handles disk page caching in DBMS."` | Semantically related to `DBMS-TRAIN-0015`; injects verified experience context into Self-RAG. |
| **Path C (Self-RAG Fallback)** | `"What are the operational trade-offs of hash index vs B-tree index in PostgreSQL?"` | No sufficiently similar memory candidate; performs standard adaptive Self-RAG over DBMS PDFs. |
| **Direct Parametric Answer** | `"What is 2 + 2?"` | Emits `[No Retrieval]` and answers directly without retrieving documents or memory. |

