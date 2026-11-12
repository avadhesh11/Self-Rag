# EvoRAG Task 12C: Self-RAG MLX QLoRA Training Dataset Report

**Date:** 2026-11-12
**Task:** Task 12C — Prepare Self-RAG MLX QLoRA Training Dataset
**Branch:** `evorag-development`
**Execution Environment:** Isolated `.venv-mlx` (Apple Silicon M3 Pro, macOS 15.5, ARM64)
**Status:** COMPLETE (All 260 curated examples generated, validated, partitioned, and verified compatible with MLX-LM 0.32.2)

---

## 1. Executive Summary

In Task 12C, a high-quality, supervised fine-tuning dataset was prepared and serialized for local MLX-LM QLoRA training of the 4-bit converted Self-RAG model ([`models/mlx/selfrag_llama2_7b_4bit/`](file:///Users/devj/code/self-rag/self-rag-demo/models/mlx/selfrag_llama2_7b_4bit)).

The dataset contains **260 curated, instruction-following DBMS examples** spanning **13 canonical database categories** (20 examples per category). Every example adheres strictly to original Self-RAG formatting conventions, preserving all 16 reflection tokens (`[Retrieval]`, `[No Retrieval]`, `<paragraph>`, `</paragraph>`, `[Relevant]`, `[Irrelevant]`, `[Fully supported]`, `[Partially supported]`, `[No support / Contradictory]`, and `[Utility:1..5]`).

All examples are grounded in the 15 DBMS educational PDFs and their 232 indexed chunks ([`index/dbms_sql_v1/corpus.jsonl`](file:///Users/devj/code/self-rag/self-rag-demo/index/dbms_sql_v1/corpus.jsonl)), while incorporating relevant conceptual material from the 20 approved DBMS training experiences without duplication.

A rigorous contamination audit verified **zero overlap** (0 exact, 0 normalized, 0 near-duplicates) against the 32 held-out evaluation questions. The dataset was partitioned using deterministic category-stratified splitting into Training (70%, 182 examples), Validation (15%, 39 examples), and Internal Test (15%, 39 examples) sets. Serialized files were tested with MLX-LM's native `load_local_dataset` loader, confirming full prompt-masking compatibility under Apple Silicon Metal.

---

## 2. Dataset Architecture & Inventory

All dataset deliverables are located in [`training/qlora_dbms/mlx_dataset/`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset):

| Deliverable File | Description | Row Count | File Size | SHA-256 Checksum |
| :--- | :--- | :---: | :---: | :--- |
| [`candidate_examples.jsonl`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset/candidate_examples.jsonl) | Full candidate pool with complete provenance | 260 | 951.5 KB | `a9ee8fb55913f01f...` |
| [`approved_examples.jsonl`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset/approved_examples.jsonl) | Verified and approved examples | 260 | 951.5 KB | `a9ee8fb55913f01f...` |
| [`train.jsonl`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset/train.jsonl) | Training split (70% stratified) | 182 | 266.0 KB | `fa5e46bfb7b3be48...` |
| [`valid.jsonl`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset/valid.jsonl) | Validation split (15% stratified) | 39 | 55.2 KB | `0d8b7ea7ea75e47c...` |
| [`internal_test.jsonl`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset/internal_test.jsonl) | Internal test split (15% stratified) | 39 | 52.4 KB | `0816999b9cf9b33a...` |
| [`test.jsonl`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset/test.jsonl) | Compatibility mirror for MLX-LM `load_local_dataset` | 39 | 52.4 KB | `0816999b9cf9b33a...` |
| [`dataset_manifest.json`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset/dataset_manifest.json) | Metadata manifest, statistics, and checksums | — | 3.5 KB | `bb9d6c703e3fba08...` |
| [`dataset_quality_report.md`](file:///Users/devj/code/self-rag/self-rag-demo/training/qlora_dbms/mlx_dataset/dataset_quality_report.md) | Quality inspection report and audit summary | — | 3.8 KB | `829a2e6e3c509315...` |

---

## 3. Category Distribution (Stratified Partitioning)

The 260 examples are evenly distributed across all 13 required DBMS knowledge categories. Each category contains exactly 20 examples partitioned into 14 training (70%), 3 validation (15%), and 3 internal testing (15%) examples using fixed seed `42`:

| Category | Total Count | Train Split (70%) | Valid Split (15%) | Internal Test (15%) |
| :--- | :---: | :---: | :---: | :---: |
| `dbms_definitions` | 20 | 14 | 3 | 3 |
| `conceptual_explanations` | 20 | 14 | 3 | 3 |
| `comparisons` | 20 | 14 | 3 | 3 |
| `algorithms` | 20 | 14 | 3 | 3 |
| `query_processing` | 20 | 14 | 3 | 3 |
| `indexing` | 20 | 14 | 3 | 3 |
| `transactions` | 20 | 14 | 3 | 3 |
| `concurrency_control` | 20 | 14 | 3 | 3 |
| `recovery_and_logging` | 20 | 14 | 3 | 3 |
| `buffer_management` | 20 | 14 | 3 | 3 |
| `storage_architecture` | 20 | 14 | 3 | 3 |
| `scenario_based_reasoning` | 20 | 14 | 3 | 3 |
| `context_grounded_abstention` | 20 | 14 | 3 | 3 |
| **Total** | **260** | **182** | **39** | **39** |

---

## 4. Self-RAG Reflection Token Distribution

The dataset balances autonomous retrieval decisions, evidence relevance judgments, support verification, and multi-grade utility scoring:

### Retrieval Decisions
- `[Retrieval]`: **220 examples (84.6%)** — factual database theory, storage internals, recovery protocols requiring retrieved documentation.
- `[No Retrieval]`: **40 examples (15.4%)** — general database definitions, standard SQL syntax, and basic arithmetic where external retrieval is unneeded.

### Relevance Critique (on Retrieved Evidence)
- `[Relevant]`: **230 examples (88.5%)** — evidence directly supports or addresses the query premise.
- `[Irrelevant]`: **30 examples (11.5%)** — distractor passages, unrelated domain queries, or out-of-scope vendor claims.

### Support Critique
- `[Fully supported]`: **205 examples (78.8%)** — complete answer directly supported by cited evidence chunk.
- `[Partially supported]`: **25 examples (9.6%)** — partial answer where model explicitly identifies missing details.
- `[No support / Contradictory]`: **30 examples (11.5%)** — abstention on irrelevant passages or explicit refutation of false premises.

### Utility Scores
- `[Utility:5]`: **205 examples (78.8%)** — accurate, comprehensive, fully grounded responses.
- `[Utility:4]`: **15 examples (5.8%)** — correct, concise explanations.
- `[Utility:3]`: **20 examples (7.7%)** — partial support answers acknowledging missing facts.
- `[Utility:2]`: **5 examples (1.9%)** — low-utility incomplete contexts.
- `[Utility:1]`: **15 examples (5.8%)** — ungrounded, irrelevant, or refuted premises.

---

## 5. Token Length Statistics (MLX Tokenizer)

Sequence lengths were computed using the converted 4-bit MLX tokenizer ([`models/mlx/selfrag_llama2_7b_4bit/`](file:///Users/devj/code/self-rag/self-rag-demo/models/mlx/selfrag_llama2_7b_4bit)):

| Sequence Segment | Min Tokens | Max Tokens | Mean Tokens | Median Tokens | P95 Tokens | Context Budget (<4096) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prompt (`prompt`)** | 7 | 73 | 18.98 | 19 | 37 | <2% |
| **Completion (`completion`)** | 39 | 369 | 219.52 | 225 | 312 | <9% |
| **Total Sequence** | **51** | **421** | **238.50** | **244** | **331** | **<11%** |

Every example fits comfortably within the model's native 4,096 token context window with ample headroom.

---

## 6. Source Document Coverage

Evidence passages were sampled and verified across all 15 DBMS educational PDFs:

| Source PDF | Examples Grounded | Primary Categories Covered |
| :--- | :---: | :--- |
| `01_dbms_fundamentals_and_architecture.pdf` | 18 | Definitions, Conceptual, Scenario |
| `02_relational_model_keys_and_integrity.pdf` | 14 | Definitions, Keys, Integrity |
| `03_er_modeling_and_database_design.pdf` | 12 | Conceptual, Comparisons |
| `04_relational_algebra_and_calculus.pdf` | 14 | Formal Algebra, Query Processing |
| `05_sql_fundamentals_ddl_dml_and_types.pdf` | 15 | DDL, DML, Transactions |
| `06_sql_queries_joins_aggregation_subqueries.pdf` | 16 | Query Processing, Comparisons |
| `07_functional_dependencies_and_normalization.pdf` | 15 | Normalization, Algorithms |
| `08_database_storage_pages_and_buffer_management.pdf` | 24 | Storage Architecture, Buffer Management |
| `09_index_structures_btrees_hashing_and_postgresql.pdf` | 25 | Indexing, Algorithms, Scenarios |
| `10_transactions_and_acid_properties.pdf` | 18 | Transactions, ACID, Isolation |
| `11_concurrency_control_serializability_mvcc.pdf` | 20 | Concurrency Control, MVCC, 2PL |
| `12_database_recovery_logging_wal_and_backup.pdf` | 22 | Recovery, Logging, WAL, Checkpoints |
| `13_query_processing_and_execution_algorithms.pdf` | 15 | Query Execution, Join Algorithms |
| `14_query_optimization_cost_models_and_explain.pdf` | 16 | Cost Models, EXPLAIN, Statistics |
| `15_distributed_databases_nosql_security_and_administration.pdf` | 16 | Distributed, Replication, Sharding |

---

## 7. Contamination Safeguard Audit

A four-tier contamination check was executed against all 32 official held-out benchmark questions (`EvoRAG_Blind_Test_Questions/held_out_test_questions_only.jsonl`):

1. **Exact Text Overlap:** 0 matches (0 / 32) — **PASS**
2. **Normalized Text Overlap:** 0 matches (0 / 32) — **PASS**
3. **Token Jaccard Overlap ($J \ge 0.65$):** 0 matches (0 / 32) — **PASS**
4. **Answer Key Separation:** Zero content was extracted or derived from `EvoRAG_Answer_Key_For_Frozen_Run/` or `evaluation/predictions/frozen/` — **PASS**

**Result:** `CLEAN_ZERO_CONTAMINATION`.

---

## 8. MLX-LM Loader Compatibility Verification

MLX-LM 0.32.2 dataset loading was validated using `mlx_lm.tuner.datasets.load_dataset`:
- **Loader Path:** `training/qlora_dbms/mlx_dataset/`
- **Format:** `CompletionsDataset` (`prompt` and `completion` keys)
- **Prompt Masking:** Tested with `mask_prompt=True`. Chat template in `tokenizer_config.json` correctly outputs `### Instruction:\n...\n\n### Response:\n` for the prompt, producing positive mask offsets (mean ~28 tokens).
- **Loss Computation:** Verified that loss is masked to 0 on prompt tokens and computed strictly on completion tokens.
- **Reflection Tokens:** Verified that `[Retrieval]`, `[Relevant]`, `<paragraph>`, `[Utility:5]`, etc., encode to their exact special token IDs (32000–32015).

---

## 9. Baseline Safeguards & Regression Suite

1. **Production Tests:** Executed offline in `.venv` (Python 3.13.7):
   ```bash
   HF_HUB_OFFLINE=1 arch -arm64 .venv/bin/pytest tests/test_task10_real_model_warm_cache.py tests/test_task9b_learning.py -v
   ```
   **13 passed, 1 warning in 6.56s**.
2. **Frozen Benchmark Hash:**
   `evaluation/predictions/frozen/predictions_20261109_151957Z.csv` = `a08d3bd3b0440302a02fd2f81c5dbab3d878cce75406657319940e4bb495d3e7` (Preserved).
3. **Model & Memory Invariance:** Original GGUF model (`models/selfrag_llama2_7b.Q4_K_M.gguf`), Experience Memory SQLite database, and FAISS indices remain unmodified.

---

## 10. Conclusion & Readiness

Task 12C is complete. The dataset is fully validated, cleanly partitioned, and verified against MLX-LM 0.32.2.

**Ready for Task 13:** QLoRA Fine-Tuning Execution using MLX-LM on Apple Silicon Metal.

