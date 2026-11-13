# Task 13: Interactive Five-System Live Comparison and Terminal Demonstration Report

**Date**: 2026-11-13
**Status**: COMPLETE
**Branch**: `evorag-development`
**Host Platform**: Apple M3 Pro (36 GB Unified Memory), macOS 15.5 Darwin ARM64

---

## 1. Executive Summary

Task 13 extends the EvoRAG platform with an **interactive live comparison tool** that allows entering a single question and evaluating the latency, retrieval behavior, reflection tokens, and final generation across all **five implemented systems** sequentially in real time:

1. **System 1: Normal RAG** — Gemini 2.5 Flash via official `google-genai` SDK with ChromaDB PDF vector index.
2. **System 2: Original Self-RAG (GGUF)** — `selfrag_llama2_7b.Q4_K_M.gguf` via `llama-cpp-python` (Metal GPU offload), without Experience Memory.
3. **System 3: EvoRAG (GGUF)** — Original Self-RAG 7B GGUF with Experience Memory and three-path adaptive routing (Path A: exact cache hit; Path B: semantic cache hit; Path C: full Self-RAG generation with experience ingestion).
4. **System 4: Original Self-RAG (MLX Base)** — 4-bit converted Self-RAG 7B (`models/mlx/selfrag_llama2_7b_4bit`) via native Apple Metal `mlx-lm`, without Experience Memory.
5. **System 5: Fine-Tuned EvoRAG (MLX QLoRA)** — MLX 4-bit base model with the Task 12E `full_dbms_v1` QLoRA adapter and Experience Memory.

### Key Highlights
- **Subprocess Isolation**: To prevent Metal GPU / unified memory contention and resolve ABI incompatibilities between Python 3.13 (`llama-cpp-python`) and Python 3.12 (`mlx-lm`), each system is executed sequentially via dedicated subprocess workers (`live_comparison_worker.py`).
- **Fair Latency Attribution**: Separates model load/initialization time (cold start) from actual query latency, and instruments retrieval time (`TimedRetriever`) to isolate passage retrieval from LLM token generation.
- **Parametric Knowledge & Question Types**: Implemented `--question-type pdf` (default, uses DBMS PDF index) and `--question-type general` (preserves model autonomy on whether to retrieve or generate parametrically).
- **Interactive & Non-Interactive CLI**: Accessible via `python demo.py --compare` (interactive prompt loop) or direct non-interactive CLI flags (`--compare --question "..."`).
- **Data & Benchmark Integrity**: Preserves all model weights, QLoRA adapters, frozen benchmark CSVs, answer keys, and approved Experience Memory entries without alteration.

---

## 2. Architecture and Design

```
+-----------------------------------------------------------------------------------+
|                                  demo.py                                          |
|                Interactive Menu (Option 5) or CLI --compare                       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     evaluation/live_comparison.py Orchestrator                    |
|   - Parses arguments (--question, --question-type, --output-dir)                  |
|   - Dispatches each system sequentially via subprocess worker                     |
|   - Real-time terminal progress cards & formatted comparison table                |
|   - Automatically writes timestamped JSON and Markdown reports                    |
+-----------------------------------------------------------------------------------+
             |                                                  |
             | (.venv - Python 3.13)                            | (.venv-mlx - Python 3.12)
             v                                                  v
+-----------------------------+                    +-----------------------------+
| live_comparison_worker.py   |                    | live_comparison_worker.py   |
|   - System 1: Normal RAG    |                    |   - System 4: MLX Base      |
|   - System 2: Self-RAG GGUF |                    |   - System 5: MLX EvoRAG    |
|   - System 3: EvoRAG GGUF   |                    +-----------------------------+
+-----------------------------+
```

### Worker Instrumentation (`evaluation/live_comparison_worker.py`)
1. **`TimedRetriever`**: Wraps `PassageRetriever` to measure exact passage retrieval latency (`retrieval_latency_ms`) without modifying core retriever classes.
2. **Cold-Start vs. Query Latency**:
   - `model_load_latency_ms`: High-resolution wall-clock timer around model initialization and weight loading.
   - `generation_latency_ms`: Time taken during LLM prompt inference and token generation.
   - `total_query_latency_ms`: End-to-end query time (`retrieval + generation + routing/caching overhead`).
3. **Fault Tolerance**: If any system throws an exception (e.g. API key absent or CUDA/Metal memory spike), the error is caught, captured with full traceback, and returned cleanly as an `ERROR` status so the remaining systems continue uninterrupted.

---

## 3. Files Created and Modified

| File | Status | Description |
|---|---|---|
| `evaluation/live_comparison_worker.py` | **New** | Standalone JSON worker executing a single system in its isolated Python runtime; instruments retrieval and generation latencies. |
| `evaluation/live_comparison.py` | **New** | Multi-system orchestrator, subprocess coordinator, ASCII comparison table formatter, and Markdown/JSON report generator. |
| `tests/test_live_comparison.py` | **New** | Unit and integration tests covering table formatting, sequential execution, error handling, worker routing, and benchmark immutability. |
| `demo.py` | **Modified** | Added `--compare`, `--question-type`, and `--question` arguments, added Option 5 to interactive menu, wired `run_comparison_mode`. |
| `docs/USAGE.md` | **Modified** | Added Section 3 with detailed live comparison usage, CLI examples, parameter explanations, and sample outputs. |
| `README.md` | **Modified** | Added Section 9 documenting the live 5-system comparison mode and quick-start commands. |

---

## 4. Test Verification

All unit and integration test suites run cleanly across both virtual environments:

### Test Suite Results
```bash
$ arch -arm64 .venv/bin/pytest tests/test_live_comparison.py tests/test_unified_cli.py
======================== 22 passed, 1 warning in 3.42s =========================
```

- `tests/test_live_comparison.py` (8 tests):
  - `test_format_comparison_table_normal`: Verifies column formatting, headers, and alignment.
  - `test_format_comparison_table_with_errors`: Ensures failed runs format as `ERROR` and `N/A` without crash.
  - `test_run_five_way_comparison_calls_all_systems`: Ensures exact sequential ordering of systems 1 through 5.
  - `test_run_five_way_comparison_continues_on_failure`: Verifies fault-tolerance when System 1 fails.
  - `test_timed_retriever_measures_latency`: Confirms `TimedRetriever` accumulates elapsed milliseconds.
  - `test_worker_cli_path_a_mock`: Confirms Path A cache hits report 0 ms generation latency and exact cache hit status.
  - `test_demo_cli_compare_args`: Validates CLI argument parser accepts `--compare`, `--question`, and `--question-type`.
  - `test_benchmark_csv_unmodified`: Validates frozen prediction benchmark hash remains unchanged.

- `tests/test_unified_cli.py` (14 tests):
  - All unified CLI parser and execution mode tests passed.

- `tests/test_mlx_generator.py` (10 tests in `.venv-mlx`):
  - All MLX generator unit and integration tests passed.

---

## 5. Live Demonstration Results

### Run 1: Technical DBMS PDF Query
- **Question**: `"What is write-ahead logging (WAL) in database systems?"`
- **Category**: `pdf` (retrieval against DBMS PDF index)
- **Report**: `evaluation/live_comparisons/comparison_20261113_132826Z.md`

#### Consolidated Comparison Table
| System | Total (ms) | Retr (ms) | Gen (ms) | Load (ms) | Retr Used | Path | Tokens / Decision | Output Snippet |
|---|---|---|---|---|---|---|---|---|
| **Normal RAG (Gemini)** | 1,148.1 | 258.6 | 889.5 | 0.0 | Yes (3) | - | - | Write-Ahead Logging (WAL) is a fundamental database technique ensuring Atomicity and Durability (the A and D in ACID) by adhering to a core protocol... |
| **Original Self-RAG (GGUF)** | 9,947.6 | 395.1 | 9,552.6 | 309.2 | Yes (3) | - | `[Retrieval]` `[Relevant]` `[Fully supported]` | Write-ahead logging (WAL) is a technique used in database systems to ensure durability and consistency of data... |
| **EvoRAG (GGUF)** | 9,583.5 | 41.9 | 9,541.6 | 308.1 | Yes (3) | Path C | `[Retrieval]` `[Relevant]` `[Fully supported]` | Write-ahead logging (WAL) is a technique used in database systems to ensure durability and consistency of data... |
| **Original Self-RAG (MLX Base)** | 2,941.7 | 353.2 | 2,588.4 | 1,228.0 | Yes (3) | - | `[Retrieval]` `[Relevant]` `[Fully supported]` | Write-ahead logging (WAL) is a technique used in database management systems to ensure data integrity and durability... |
| **Fine-Tuned EvoRAG (MLX QLoRA)** | 2,417.7 | 5.5 | 2,412.2 | 1,289.4 | Yes (3) | Path C | `[Retrieval]` `[Relevant]` `[Fully supported]` | Write-ahead logging (WAL) is a fundamental protocol in database management systems that ensures the Atomicity and Durability properties... |

#### Key Observations
1. **Speed Advantage of MLX**: MLX generation (`2,412 ms`) was **~4x faster** than GGUF (`9,541 ms`) on Apple M3 Pro Metal GPU.
2. **Retrieval Agreement**: All systems retrieved from `12_database_recovery_management.pdf` and emitted `[Relevant]` and `[Fully supported]` reflection tags.
3. **Fine-Tuned Domain Specialization**: System 5 produced concise, authoritative DBMS terminology directly matching standard textbook definitions.

---

### Run 2: General Knowledge Query (Parametric Test)
- **Question**: `"What is the capital of France?"`
- **Category**: `general` (allows parametric answering without forced retrieval)
- **Report**: `evaluation/live_comparisons/comparison_20261113_133013Z.md`

#### Consolidated Comparison Table
| System | Total (ms) | Retr (ms) | Gen (ms) | Load (ms) | Retr Used | Path | Tokens / Decision | Output Snippet |
|---|---|---|---|---|---|---|---|---|
| **Normal RAG (Gemini)** | 1,415.9 | 269.1 | 1,146.7 | 0.0 | Yes (3) | - | - | I am sorry, but the provided text does not contain information about the capital of France. |
| **Original Self-RAG (GGUF)** | 712.8 | 0.0 | 712.8 | 316.5 | No | - | `[No Retrieval]` | The capital of France is Paris. |
| **EvoRAG (GGUF)** | 849.5 | 0.0 | 849.5 | 312.4 | No | Path C | `[No Retrieval]` | The capital of France is Paris. |
| **Original Self-RAG (MLX Base)** | 801.5 | 0.0 | 801.5 | 1,208.6 | No | - | `[No Retrieval]` | The capital of France is Paris. |
| **Fine-Tuned EvoRAG (MLX QLoRA)** | 1,083.6 | 0.0 | 1,083.6 | 1,273.6 | No | Path C | `[No Retrieval]` | The capital of France is Paris. |

#### Key Observations
1. **Parametric Intelligence**: Systems 2, 3, 4, and 5 correctly recognized that external database retrieval was unnecessary, autonomously emitted the `[No Retrieval]` token, avoided index queries (`0.0 ms` retrieval latency), and answered in under $1.1$ seconds.
2. **Normal RAG Vulnerability**: Normal RAG blindly queried the DBMS PDF vector store, retrieved irrelevant database chunks, and rightfully abstained due to strict groundedness prompts. This highlights the architectural superiority of Self-RAG reflection tokens over naive RAG.

---

## 6. Integrity and Immutability Audit

The integrity of all existing models, adapters, benchmarks, answer keys, and experience memories was verified post-run:

| Artifact | Verified SHA-256 / Record Count | Status |
|---|---|---|
| `evaluation/predictions/frozen/predictions_20261109_151957Z.csv` | `a08d3bd3b0440302a02fd2f81c5dbab3d878cce75406657319940e4bb495d3e7` | **MATCH (Unchanged)** |
| `EvoRAG_Blind_Test_Questions/held_out_test_questions_only.jsonl` | `ce1a1f6fee44b9a7dac7d4d8792b1e7d17b0917a93c4f4e1ee2aa3f3b3726ea1` | **MATCH (Unchanged)** |
| `EvoRAG_Answer_Key_For_Frozen_Run/answer_key_32_for_frozen_run.jsonl` | `f4f646dcc6a55fab2a7e6759a77304875ffe113e9b2fa0880c25769f1699fc0e` | **MATCH (Unchanged)** |
| `models/selfrag_llama2_7b.Q4_K_M.gguf` | `fb39ef3293bffcac2c44415123ec14d7b62c5cdc46a5dbfab9cf3830afe7dc03` | **MATCH (Unchanged)** |
| `models/mlx/selfrag_llama2_7b_4bit/model.safetensors` | `71576febb874c07b71697976f0892c2bab9e972e77bc880dcc0210eb6342db32` | **MATCH (Unchanged)** |
| `training/qlora_dbms/adapters/full_dbms_v1/adapters.safetensors` | `b4d2ba1df01496f43bce46cdae0a6c629e73bf3de68b63afe0c8ecc0ec1d2cfd` | **MATCH (Unchanged)** |
| `data/experience_memory.db` | Exactly 25 approved DBMS experience records | **MATCH (Unchanged)** |

---

## 7. How to Run the Live Comparison

### Interactive Mode
Run the unified CLI without arguments, or directly request compare mode:
```bash
# Option 1: Via interactive menu (select option 5)
python demo.py

# Option 2: Direct interactive compare loop
python demo.py --compare
```

### Non-Interactive CLI Mode
```bash
# DBMS PDF Question (uses DBMS index)
python demo.py --compare --question-type pdf --question "What is write-ahead logging (WAL) in database systems?"

# General Knowledge Question (tests parametric [No Retrieval] decision)
python demo.py --compare --question-type general --question "What is the capital of France?"
```
All runs automatically log consolidated results to `evaluation/live_comparisons/comparison_<timestamp>.json` and `.md`.

