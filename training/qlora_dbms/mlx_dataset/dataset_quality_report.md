# EvoRAG Self-RAG MLX QLoRA Dataset Quality Report

**Date:** 2026-11-12  
**Total Examples:** 260  
**Random Seed:** 42  
**Contamination Status:** `CLEAN_ZERO_CONTAMINATION` (0 violations across 32 held-out eval questions)  

## 1. Split Distribution

| Split | Rows | Percentage | Category Stratification |
| :--- | :---: | :---: | :---: |
| **Train (`train.jsonl`)** | 182 | 70.0% | 14 per category (13 categories) |
| **Validation (`valid.jsonl`)** | 39 | 15.0% | 3 per category (13 categories) |
| **Internal Test (`internal_test.jsonl`)** | 39 | 15.0% | 3 per category (13 categories) |
| **Total** | **260** | **100.0%** | **20 per category** |

## 2. Category Distribution

| Category | Total Count | Train | Valid | Internal Test |
| :--- | :---: | :---: | :---: | :---: |
| `algorithms` | 20 | 14 | 3 | 3 |
| `buffer_management` | 20 | 14 | 3 | 3 |
| `comparisons` | 20 | 14 | 3 | 3 |
| `conceptual_explanations` | 20 | 14 | 3 | 3 |
| `concurrency_control` | 20 | 14 | 3 | 3 |
| `context_grounded_abstention` | 20 | 14 | 3 | 3 |
| `dbms_definitions` | 20 | 14 | 3 | 3 |
| `indexing` | 20 | 14 | 3 | 3 |
| `query_processing` | 20 | 14 | 3 | 3 |
| `recovery_and_logging` | 20 | 14 | 3 | 3 |
| `scenario_based_reasoning` | 20 | 14 | 3 | 3 |
| `storage_architecture` | 20 | 14 | 3 | 3 |
| `transactions` | 20 | 14 | 3 | 3 |

## 3. Reflection Token Distribution

### Retrieval Decisions
- `[Retrieval]`: 249 (95.8%)
- `[No Retrieval]`: 11 (4.2%)

### Relevance Critique (on Retrieved Evidence)
- `[Irrelevant]`: 10 (4.0%)
- `[Relevant]`: 239 (96.0%)

### Support Critique
- `[Fully supported]`: 229 (92.0%)
- `[No support / Contradictory]`: 15 (6.0%)
- `[Partially supported]`: 5 (2.0%)

### Utility Scores
- `[Utility:1]`: 15 (5.8%)
- `[Utility:3]`: 5 (1.9%)
- `[Utility:4]`: 2 (0.8%)
- `[Utility:5]`: 238 (91.5%)

## 4. Token Length Statistics (MLX Tokenizer)

| Metric | Min | Max | Mean | Median | P95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Prompt Tokens** | 7 | 73 | 18.98 | 17 | 37 |
| **Completion Tokens** | 39 | 369 | 219.52 | 238 | 312 |
| **Total Tokens** | 51 | 421 | 238.5 | 254 | 331 |

## 5. Contamination Safeguard Audit

- **Held-Out Benchmark Set:** `EvoRAG_Blind_Test_Questions/held_out_test_questions_only.jsonl` (32 questions)
- **Exact Question Overlap:** 0 matches (CLEAN)
- **Normalized Overlap:** 0 matches (CLEAN)
- **Token Jaccard Overlap (> 0.65):** 0 matches (CLEAN)
- **Answer Key Separation:** Verified zero leakage from `EvoRAG_Answer_Key_For_Frozen_Run/`

## 6. Source Document Coverage

| Source PDF | Evidence Chunks Used |
| :--- | :---: |
| `01_dbms_fundamentals_and_architecture.pdf` | 14 examples |
| `02_relational_model_keys_and_integrity.pdf` | 11 examples |
| `03_er_modeling_and_database_design.pdf` | 2 examples |
| `04_relational_algebra_and_calculus.pdf` | 6 examples |
| `05_sql_fundamentals_ddl_dml_and_types.pdf` | 12 examples |
| `06_sql_queries_joins_aggregation_subqueries.pdf` | 7 examples |
| `07_functional_dependencies_and_normalization.pdf` | 7 examples |
| `08_database_storage_pages_and_buffer_management.pdf` | 47 examples |
| `09_index_structures_btrees_hashing_and_postgresql.pdf` | 30 examples |
| `10_transactions_and_acid_properties.pdf` | 24 examples |
| `11_concurrency_control_serializability_mvcc.pdf` | 27 examples |
| `12_database_recovery_logging_wal_and_backup.pdf` | 27 examples |
| `13_query_processing_and_execution_algorithms.pdf` | 16 examples |
| `14_query_optimization_cost_models_and_explain.pdf` | 10 examples |
| `15_distributed_databases_nosql_security_and_administration.pdf` | 9 examples |

## 7. MLX-LM Loader Compatibility

- File naming conforms with MLX-LM (`train.jsonl`, `valid.jsonl`, `test.jsonl`).
- `chat_template` is configured in `tokenizer_config.json` for prompt masking (`--mask-prompt`).
- Special Self-RAG reflection tokens (`[Retrieval]`, `[Relevant]`, etc.) are preserved verbatim.
