# EvoRAG Model-Free End-to-End Validation Report (Task 7A)

This document provides a comprehensive report of the model-free end-to-end integration validation performed across the EvoRAG infrastructure independently of language model inference.

---

## 1. Objective & Scope

The goal of Task 7A is to validate the integrity of the complete EvoRAG system—document processing, provenance tracking, SQLite Experience Memory, conservative validation heuristics, augmented retrieval, three-path hybrid routing, and recoverable persistence—using **real local components wherever feasible**, without relying on external language model weights or remote API endpoints.

---

## 2. Environment & Dependency Audit

| Dependency / Component | Required For | Status in Local Environment | Notes |
| :--- | :--- | :---: | :--- |
| **Python** | Runtime | **3.12.6 (Win32/x64)** | Tested and verified. |
| **pypdf** | PDF text extraction | **Installed (6.13.2)** | Fully functional for real document extraction. |
| **pymupdf (fitz)** | PDF generation / OCR | **Installed (1.28.0)** | Used to create synthetic realistic test PDFs with legal references. |
| **sqlite3** | Experience Memory store | **Available (Built-in)** | Real ACID storage, indexing, and parameterized queries verified. |
| **numpy** | Vector math / normalization | **Installed (2.4.6)** | Used in normalization and cosine similarity. |
| **pytest** | Automated testing | **Installed (9.1.1)** | Test runner executing all 79 tests. |
| **google-genai** | Gemini baseline comparison | **Installed (2.0.1)** | Available when `GEMINI_API_KEY` is provided. |
| **faiss-cpu / faiss** | Dense vector search | *Not Installed* | Native C++ library missing in local Python environment. |
| **sentence-transformers**| Dense text embedding | *Not Installed* | Model cached on disk (`all-MiniLM-L6-v2`), but package is not installed. |
| **torch** | Deep learning runtime | *Not Installed* | Missing in local environment (requires ~2.5 GB download). |
| **llama-cpp-python** | Local GGUF Self-RAG model | *Not Installed* | Requires local compiler/wheels and model weights. |

---

## 3. Real vs. Mocked Components

### Components Tested with REAL Local Implementations:
1. **Document PDF Text Extraction (`pypdf`)**: Extracted multi-page text from realistic test PDFs.
2. **Text Artifact Cleaning (`clean_pdf_text`)**: Fixed OCR ligature and hyphenation artifacts across line breaks (`SELF -RAG` $\to$ `SELF-RAG`).
3. **Abbreviation-Aware Sentence Splitting (`split_sentences`)**: Verified that legal/numeric references (`Sec. 14`, `Art. 19`, `No. 5`, `Rs. 500`) are not split incorrectly.
4. **Passage Chunking & Overlap Buffer (`merge_to_target_size`)**: Merged sentences to target word limits with trailing context buffers and page boundary flushing.
5. **Deterministic Provenance**: Generated SHA-256 chunk hashes (`compute_chunk_hash`) and PDF byte hashes (`compute_file_hash`).
6. **Corpus & Metadata Generation**: Built `corpus.jsonl` and `meta.json` and verified disk persistence.
7. **Experience Memory Storage (`ExperienceStore`)**: Executed real SQLite database operations on disk—table creation, index lookups, duplicate rejection, and status filtering.
8. **Conservative Validation (`ExperienceValidator`)**: Executed real validation logic on actual data—rejection of unapproved status, polarity/negation mismatches (`NEGATION_WORDS`), numeric scope mismatches, interrogative mismatches, and source version freshness checks against disk metadata.
9. **Experience Augmented Retrieval (`ExperienceAugmentedRetriever`)**: Formatted prior experiences with anti-instruction delimiters (`[PRIOR VERIFIED EXPERIENCE...]`), tagged `source_type="document"` vs `source_type="experience_memory"`, and enforced retrieval budget semantics ($k$ controls document passages; memory passages do not displace document passages).
10. **Controller Routing (`EvoRAGController`)**: Executed three-path routing (Path A direct reuse, Path B experience-assisted, Path C fallback).
11. **Recoverable Persistence (`approve_and_save` & `recover_index`)**: Verified that an index rebuild failure preserves the SQLite record, invalidates the in-memory retriever, and allows clean recovery without duplicate records.
12. **Terminal CLI (`evorag_cli.py`)**: Validated argument parsing and graceful error handling when models or indexes are unconfigured.

### Components Mocked:
1. **Language Model Generator (`SelfRAGGenerator`)**: Mocked with `DeterministicTrackingGenerator` to verify prompt construction, batch passage ingestion, and reflection token handling without downloading 4.1 GB weights.
2. **Vector Embeddings (`SentenceTransformer`) & FAISS C++ Index**: Mocked in vector search tests when `faiss-cpu` / `sentence-transformers` packages are absent.

---

## 4. Verification of the Three Query Paths

| Path | Query Type | Observed Behavior | Generator Invoked? | Document Retriever Invoked? |
| :--- | :--- | :--- | :---: | :---: |
| **Path A: Direct Reuse** | Exact normalized question match (`"  what is section 14? "`) | Returns stored answer, pre-computed evidence, and original provenance. | **No (0 calls)** | **No (0 calls)** |
| **Path B: Experience-Assisted** | Related non-identical question (`"Explain penalty under Section 14"`) | Injects delimited experience context alongside document passages into generator batch; generates fresh answer; separates evidence categories. | **Yes** | **Yes** |
| **Path C: Normal Self-RAG** | Unseen question with empty/invalid memory | Fallback to document retriever; passes solely document passages to generator; sets `memory_context_used=False`. | **Yes** | **Yes** |

---

## 5. Task 7A.1: Real FAISS and SentenceTransformer Validation

Task 7A.1 extends the validation framework to move beyond mock embeddings and verified vector indices, testing the **actual vector retrieval infrastructure** with real local dependencies (`faiss-cpu` and `sentence-transformers`).

### Four Clear Validation Layers

| Layer | Component | Status | Method | Notes |
| :--- | :--- | :---: | :--- | :--- |
| **1. Mock-Based Validation** | Unit tests & fast regressions | **Verified** | `DeterministicTestEmbedder`, `MockIndexFlatIP`, tracking generator | Fast CI testing (79 tests, ~3s), runs with 0 GPU or heavy downloads. |
| **2. Real FAISS Validation** | Dense vector search | **Verified** | `faiss.IndexFlatIP` via `faiss-cpu` (1.15.0) | True C++ vector indexing, serialization (`write_index`/`read_index`), and search. |
| **3. Real SentenceTransformer Validation** | Dense text embeddings | **Verified** | `sentence-transformers/all-MiniLM-L6-v2` (6.0.0) | Real 384-dimensional float32 L2-normalized embeddings and cosine ranking. |
| **4. Full Self-RAG Inference Validation** | Reflection token generation | *Pending* | Quantized 7B LLaMA-2 Self-RAG (`.gguf` / vLLM) | Requires ~4.1 GB weights download or Google Colab GPU environment. |

### Real Validation Results (Task 7A.1)

| Test Area | Real or Mock | Observed Result | Status |
| :--- | :---: | :--- | :---: |
| **SentenceTransformer Loading** | Real | Loaded `all-MiniLM-L6-v2` locally from Hugging Face cache. | **PASSED** |
| **Embedding Generation** | Real | Generated 384-dim float32 vectors with unit L2 norm ($1.0 \pm 10^{-5}$). | **PASSED** |
| **Semantic Similarity Ranking** | Real | Relevant query scored 0.727 vs legal appeal 0.253 vs rainforest -0.033. | **PASSED** |
| **Document FAISS Build & Persistence** | Real | `build_index` built and wrote `faiss.index`, `corpus.jsonl`, `meta.json`. | **PASSED** |
| **Document Retrieval & Metadata** | Real | `PassageRetriever` retrieved top passage with score, chunk_hash, source_version. | **PASSED** |
| **Experience FAISS Build** | Real | Indexed approved experience questions into `experience_faiss.index`. | **PASSED** |
| **Experience Retrieval & Ranking** | Real | Semantic retrieval correctly retrieved matching experience with score > 0.75. | **PASSED** |
| **SQLite ID Mapping** | Real | FAISS vector index resolved strictly via `id_map` to SQLite `experience_id`. | **PASSED** |
| **Incremental Addition** | Real | Adding an experience to SQLite and rebuilding made it immediately retrievable. | **PASSED** |
| **Persistence Across Reopen** | Real | Closed and reopened store and retriever; retrieval results remained intact. | **PASSED** |
| **Recovery: Missing FAISS Index** | Real | Index file deleted; `recover_index()` rebuilt it from SQLite without data loss. | **PASSED** |
| **Recovery: Corrupted FAISS Index** | Real | Corrupted binary file raised `RuntimeError`; `recover_index()` restored clean index. | **PASSED** |
| **Recovery: Corrupted Metadata JSON** | Real | Corrupted JSON raised `JSONDecodeError`; `recover_index()` rebuilt valid metadata. | **PASSED** |
| **Empty Memory Handling** | Real | 0 experiences built empty index; retriever safely returned `[]` without error. | **PASSED** |
| **Three-Path Routing with Real FAISS** | Real | Verified Path A direct reuse (0 LLM calls), Path B assist, and Path C fallback. | **PASSED** |
| **Dimension Mismatch Handling** | Real | Query dimension mismatch explicitly raises `ValueError` with clear message. | **PASSED** |

---

## 6. Commands to Reproduce Validation

```bash
# 1. Run the entire test suite (95 tests: 79 unit + 16 real integration tests)
python -m pytest tests/ -v

# 2. Run ONLY the fast model-free unit tests (79 tests)
python -m pytest tests/ -m "not integration" -v

# 3. Run ONLY the real FAISS and SentenceTransformer integration tests (16 tests)
python -m pytest tests/ -m "integration" -v

# 4. Run the model-free end-to-end validation suite
python -m pytest tests/test_model_free_validation.py -v

# 5. Test CLI configuration handling
python evorag_cli.py --help
python evorag_cli.py "Test question without index or model"
```

---

## 7. Known Limitations & Pending Model Work

> [!IMPORTANT]
> **Actual Language Model Generation is Pending**:
> Task 7A validates the end-to-end software engineering infrastructure, routing contracts, provenance tracking, and storage reliability of EvoRAG. It does **not** evaluate live generation quality or empirical reflection token generation of the 7B LLaMA-2 Self-RAG model.
>
> Running real LLM generation requires downloading `selfrag_llama2_7b.Q4_K_M.gguf` (~4.1 GB) or executing in Google Colab with GPU acceleration.

