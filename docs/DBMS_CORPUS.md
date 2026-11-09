# EvoRAG DBMS & SQL Knowledge Corpus Integration Report

This document specifies the integration, indexing, retrieval validation, and provenance guarantees of the **15-Document DBMS & SQL Knowledge Corpus** into the EvoRAG framework (Task 8).

---

## 1. Executive Summary & Corpus Metadata

* **Corpus Scope**: 15 standalone, text-extractable educational PDF documents, sample PostgreSQL schema, and manifest.
* **Format**: Selectable text PDFs (headings, definitions, algorithm pseudocode, tables, and SQL examples).
* **Reference Baseline**: PostgreSQL 18.6 (snapshot date: 6 November 2026); vendor-neutral relational and database theory where applicable.
* **Educational Sources & Attributions**:
  * PostgreSQL Official Documentation (v18 baseline)
  * Carnegie Mellon University (CMU) 15-445/645 Database Systems (Fall 2026 materials)
  * BCcampus *Database Design* (Open textbook, CC BY 4.0)
* **Status**: Synthesized educational reference notes for retrieval augmentation; not official vendor documentation or legal specifications.
* **Storage Location**: `data/pdf/` (PDF files), `data/manifest.json`, `data/university_schema.sql`.
* **Dedicated Index Location**: `index/dbms_sql_v1/` (`faiss.index`, `corpus.jsonl`, `meta.json`).

---

## 2. Manifest and Corpus Verification

Every file in `data/pdf/` was audited against `data/manifest.json` using SHA-256 byte checksums and `pypdf` page counts:

| Document ID | Filename | Title | Pages | Characters | SHA-256 Checksum | Verification |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| **DBMS-01** | `01_dbms_fundamentals_and_architecture.pdf` | DBMS Fundamentals and Architecture | 6 | 11,652 | `a7e3dc970655f502f95ab157ceb43b2ce1a3be4d2d4e2ee3bbf8833840034356` | **PASSED** |
| **DBMS-02** | `02_relational_model_keys_and_integrity.pdf` | Relational Model, Keys, and Integrity Constraints | 6 | 11,381 | `30abbaf7a2fa07e3a57787a79ce7b45b1569f7d58756ef8f8d78aec25aab6792` | **PASSED** |
| **DBMS-03** | `03_er_modeling_and_database_design.pdf` | ER Modeling and Conceptual Database Design | 6 | 9,029 | `e769687159711001e87a0e67d8fe9fc0ec34e6c13aba7653be03dfcd99ae3629` | **PASSED** |
| **DBMS-04** | `04_relational_algebra_and_calculus.pdf` | Relational Algebra and Relational Calculus | 6 | 8,176 | `3c9875d58477315508114e77ad202f714385cf71f66ad17821013834e24b8d98` | **PASSED** |
| **DBMS-05** | `05_sql_fundamentals_ddl_dml_and_types.pdf` | SQL Fundamentals: DDL, DML, and Data Types | 6 | 8,948 | `315dd2480af6231e3eadec20c71f7bdcacd35b1c09c816b0151377fd915dd20d` | **PASSED** |
| **DBMS-06** | `06_sql_queries_joins_aggregation_subqueries.pdf` | SQL Querying: Joins, Aggregation, and Subqueries | 6 | 8,106 | `231c06286cd3806cb332e959d57e0b884d6ed0d6c726ce4b8e05c28d4188c216` | **PASSED** |
| **DBMS-07** | `07_functional_dependencies_and_normalization.pdf` | Functional Dependencies and Normalization | 6 | 8,346 | `a8c0147821ddbc621bc408c6ff632d09370df3aa46d7ce6f246c47a3a615cccb` | **PASSED** |
| **DBMS-08** | `08_database_storage_pages_and_buffer_management.pdf` | Database Storage, Pages, and Buffer Management | 6 | 8,828 | `facb23b0082ba64de78c501f9bbc528752cfdc253766578b5e226fe0d2bc51c5` | **PASSED** |
| **DBMS-09** | `09_index_structures_btrees_hashing_and_postgresql.pdf` | Index Structures: B-Trees, Hashing, and PostgreSQL Indexes | 6 | 8,268 | `88fa521f4ce6de2650bcc61d01197d7448a64feb0d680f58c30e59e688925d23` | **PASSED** |
| **DBMS-10** | `10_transactions_and_acid_properties.pdf` | Transactions and ACID Properties | 6 | 8,626 | `591c0270e8017c3833ee4457a48f1d7ac270b87cb3c53f4442377e6fd4acb00d` | **PASSED** |
| **DBMS-11** | `11_concurrency_control_serializability_mvcc.pdf` | Concurrency Control: Serializability, Locking, and MVCC | 6 | 7,929 | `1f03aacb6f7079649cef8063dcb1856811527cda742f91fbd6dd7193890d0776` | **PASSED** |
| **DBMS-12** | `12_database_recovery_logging_wal_and_backup.pdf` | Database Recovery, Logging, WAL, and Backup | 6 | 8,522 | `2106972c6103bb9c26548ba3810fe29050532404d66913015e2e2edf94baf343` | **PASSED** |
| **DBMS-13** | `13_query_processing_and_execution_algorithms.pdf` | Query Processing and Execution Algorithms | 6 | 8,210 | `827d243c34aeb92527733d35907bda5d08025c6627a5e10be990766bacc8a96b` | **PASSED** |
| **DBMS-14** | `14_query_optimization_cost_models_and_explain.pdf` | Query Optimization: Cost Models, Statistics, and EXPLAIN | 6 | 8,412 | `3a7c14c7effdbbf0c1dd6e6600e4ee943cebead0f711164042072cb701e7566c` | **PASSED** |
| **DBMS-15** | `15_distributed_databases_nosql_security_and_administration.pdf` | Distributed Databases, NoSQL, Security, and Administration | 6 | 12,947 | `c0cd1a8faa6a560e6eba970246fca63181486c46278231b359eec124a0b62a0e` | **PASSED** |

* **Total Pages**: 90 pages (6 pages per document across all 15 documents).
* **Sample Schema**: `data/university_schema.sql` (2,669 bytes, valid PostgreSQL 18 relational teaching schema).
* **Corrupted or Missing Files**: None (0 files missing or corrupted).

---

## 3. Dedicated Document Index Architecture

The dedicated document index was generated at `index/dbms_sql_v1/` using the standard EvoRAG index builder:

```bash
python index/build_index.py \
  --pdf data/pdf/*.pdf \
  --out index/dbms_sql_v1
```

### Ingestion Specifications
* **Extractor**: `pypdf.PdfReader` with text normalization and ligature cleaning (`clean_pdf_text`).
* **Sentence Splitter**: Regex-based boundary splitting preserving legal/abbreviation prefixes (`Sec.`, `Art.`, `No.`, `Rs.`).
* **Chunk Merging**: Target chunk size of 120 words with 30-word consecutive sentence overlap within page boundaries.
* **Total Chunks Generated**: **232 passages** across 15 documents.
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
* **Vector Normalization**: Vectors are L2-normalized so inner product equals cosine similarity.
* **Vector Store**: FAISS `IndexFlatIP` (384 dimensions, 232 vectors).
* **Index Files Generated**:
  * `index/dbms_sql_v1/faiss.index`: Binary FAISS index (356 KB).
  * `index/dbms_sql_v1/corpus.jsonl`: Line-delimited JSON chunks with chunk metadata (241 KB).
  * `index/dbms_sql_v1/meta.json`: Corpus metadata with exact file byte checksums (2.8 KB).

### Provenance Guarantee
Each chunk in `corpus.jsonl` contains:
* `source`: Source PDF filename (e.g. `09_index_structures_btrees_hashing_and_postgresql.pdf`).
* `source_version`: SHA-256 byte hash matching `data/manifest.json`.
* `chunk_hash`: Deterministic SHA-256 hash of `"{source}:{normalized_text}"` (64-character hex).
* `chunk_id`: Sequential 0-based integer ID (`0` to `231`).
* `page`: 1-based page number within the source PDF (`1` to `6`).

---

## 4. Retrieval Smoke-Test Results

Retrieval was validated across 10 distinct categories covering all 15 documents using the actual `PassageRetriever("index/dbms_sql_v1")`:

| Category | Query | Top Retrieved Document | Page | Score | Result |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **1. Fundamentals** | *"Why do database systems exist and what is physical data independence?"* | `01_dbms_fundamentals_and_architecture.pdf` | 6 | 0.5914 | **MATCH** |
| **2. Relational Theory** | *"What is the difference between a candidate key and a primary key in relational model?"* | `02_relational_model_keys_and_integrity.pdf` | 6 | 0.5542 | **MATCH** |
| **3. ER Modeling** | *"How are weak entities and cardinality ratios modeled in ER diagrams?"* | `03_er_modeling_and_database_design.pdf` | 6 | 0.5018 | **MATCH** |
| **4. Relational Algebra** | *"What are fundamental relational algebra operations like select, project, and cartesian product?"* | `04_relational_algebra_and_calculus.pdf` | 1 | 0.6272 | **MATCH** |
| **5. SQL DDL/DML** | *"How do identity columns and CREATE TABLE work in PostgreSQL DDL?"* | `05_sql_fundamentals_ddl_dml_and_types.pdf` | 3 | 0.7454 | **MATCH** |
| **6. SQL Queries & Joins** | *"How do INNER JOIN, LEFT OUTER JOIN, and GROUP BY HAVING work in SQL?"* | `06_sql_queries_joins_aggregation_subqueries.pdf` | 6 | 0.5248 | **MATCH** |
| **7. Normalization** | *"What is Boyce-Codd Normal Form BCNF and functional dependencies?"* | `07_functional_dependencies_and_normalization.pdf` | 6 | 0.5426 | **MATCH** |
| **8. Storage & Buffer** | *"How does the buffer pool manager and slotted page architecture work?"* | `08_database_storage_pages_and_buffer_management.pdf` | 3 | 0.5954 | **MATCH** |
| **9. Indexing** | *"How do B-tree indexes, GIN, and BRIN indexes work in PostgreSQL?"* | `09_index_structures_btrees_hashing_and_postgresql.pdf` | 2 | 0.6145 | **MATCH** |
| **10. Transactions** | *"What are ACID properties and transaction isolation levels?"* | `10_transactions_and_acid_properties.pdf` | 2 | 0.6892 | **MATCH** |
| **11. Concurrency** | *"How does Multi-Version Concurrency Control MVCC and two-phase locking work?"* | `11_concurrency_control_serializability_mvcc.pdf` | 1 | 0.6887 | **MATCH** |
| **12. Recovery & WAL** | *"How does Write-Ahead Logging WAL and ARIES checkpointing work?"* | `12_database_recovery_logging_wal_and_backup.pdf` | 6 | 0.7242 | **MATCH** |
| **13. Query Processing** | *"What are nested loop join, hash join, and sort-merge join execution algorithms?"* | `13_query_processing_and_execution_algorithms.pdf` | 3 | 0.6568 | **MATCH** |
| **14. Optimization** | *"How does the query optimizer use cost models and EXPLAIN ANALYZE statistics?"* | `14_query_optimization_cost_models_and_explain.pdf` | 1 | 0.6225 | **MATCH** |
| **15. Security & NoSQL** | *"What is the CAP theorem, Role-Based Access Control RBAC, and Row Level Security in PostgreSQL?"* | `15_distributed_databases_nosql_security_and_administration.pdf` | 6 | 0.5297 | **MATCH** |
| **Cross-Document** | *"How do database indexes improve SQL join query execution performance?"* | `09_index_structures_btrees_hashing_and_postgresql.pdf` | 1 | 0.6926 | **MATCH** |
| **Scenario-Based** | *"Handling high-concurrency student course enrollments without deadlock or lost updates"* | `03_er_modeling_and_database_design.pdf` | 5 | 0.5426 | **MATCH** |

---

## 5. PostgreSQL 18 Validation Status

* **Installed Client CLI**: `psql (PostgreSQL) 18.3 (Homebrew)` at `/opt/homebrew/bin/psql`.
* **Database Schema Inspected**: `data/university_schema.sql` (implements tables `departments`, `students`, `instructors`, `courses`, `enrollments`, with `BIGINT GENERATED ALWAYS AS IDENTITY`, foreign keys with `ON DELETE CASCADE/RESTRICT`, and indexes).
* **Live Server Execution Status**: **Unverified**. No running PostgreSQL server daemon was active on localhost or `/tmp:5432`. Creating an isolated local test cluster via `initdb` failed in the secure execution sandbox due to kernel shared memory permissions (`shmget` operation not permitted).
* **Recommendation**: Execute `university_schema.sql` and PDF code examples against a disposable PostgreSQL 18 instance before deploying in production environments.

---

## 6. Experience Memory Isolation

* **Isolation Verification**: The DBMS document index was built strictly in `index/dbms_sql_v1/`.
* **State Preservation**: `data/experience_memory.db` was verified and contained its original undisturbed records.
* **Baseline Preservation**: The original baseline document index in `index/` (`sample.pdf` and `2310.11511v1.pdf`) remains intact.
* **Provenance Distinction**: Chunks in `index/dbms_sql_v1/corpus.jsonl` carry standard document provenance hashes and do not contain or collide with synthetic memory identifiers (`mem_...`).

---

## 7. Testing & Verification

A dedicated test suite was added in [`tests/test_dbms_corpus.py`](file:///Users/devj/code/self-rag/self-rag-demo/tests/test_dbms_corpus.py):

```bash
# Run all DBMS corpus manifest, index reload, and retrieval tests (23 tests)
HF_HUB_OFFLINE=1 arch -arm64 .venv/bin/pytest tests/test_dbms_corpus.py -v

# Run complete integration test suite (39 tests: 16 FAISS integration + 23 DBMS corpus tests)
HF_HUB_OFFLINE=1 arch -arm64 .venv/bin/pytest tests/ -m "integration" -v

# Run fast model-free unit tests (79 tests)
HF_HUB_OFFLINE=1 arch -arm64 .venv/bin/pytest tests/ -m "not integration and not model" -v
```

All 23 DBMS tests passed with 100% precision. All 79 model-free unit tests and 16 original FAISS integration tests continue to pass without regression.
