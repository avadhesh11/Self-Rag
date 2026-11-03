# EvoRAG Experience Memory Architecture & Integration

This document specifies the design, operational contracts, and provenance guarantees of the **Experience Memory** subsystem and its integration with Self-RAG as established through Tasks 5, 6, 6A, and 6B.

---

## 1. Hybrid Experience Learning Architecture

EvoRAG augments adaptive Self-RAG with an episodic experience store. Every user query routes through one of three distinct paths:

```
                          USER QUERY
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       Exact match check             Semantic search
      (Canonical normalized            (FAISS Memory
       question equality)             Cosine Similarity)
               │                             │
    ┌──────────┴──────────┐                  ▼
    │ Exact Match Found & │        Validate Candidates:
    │ Approved & Fresh?   │        • Approval status
    └──────────┬──────────┘        • Polarity / Negation
               │                   • Numeric & Interrogative scope
      YES ─────┴───── NO           • Source version freshness
       │              │            • Score >= threshold
       ▼              ▼                      │
    PATH A:        Has Valid                 ▼
    Direct Reuse   Candidates? ────► NO ──► PATH C: Normal Self-RAG
    (0 LLM cost)      │                     (Base document retriever)
                      ▼ YES
                   PATH B:
                   Experience-Assisted Self-RAG
                   (Augmented retriever injects experience context)
```

### Path A — Exact-Match Direct Reuse (`response_mode="direct_reuse"`)
- **Condition**: The incoming query's canonical normalized form (`normalize_question(query)`) matches an approved experience in SQLite exactly, and the experience passes all validation checks (source version freshness, non-empty answer, polarity/scope match).
- **Execution**: The stored answer and supporting evidence are returned directly.
- **Cost**: 0 LLM forward passes; 0 generation latency.
- **Result Flags**: `memory_hit=True`, `memory_context_used=False`.

### Path B — Related-Experience-Assisted Generation (`response_mode="experience_assisted"`)
- **Condition**: No exact normalized question match exists, but one or more semantically related experiences pass conservative validation (`score >= threshold`, `status="approved"`, polarity and scope match, source documents unchanged).
- **Execution**: The top related experiences (up to `max_memory_context`, default 2) are wrapped into an `ExperienceAugmentedRetriever`. Self-RAG executes its standard adaptive multi-segment tree-decoding loop, evaluating candidate continuations generated from both document passages and experience context passages.
- **Cost**: Normal Self-RAG inference, enriched with prior verified reasoning.
- **Result Flags**: `memory_hit=False`, `memory_context_used=True`.

### Path C — Normal Self-RAG Fallback (`response_mode="normal_self_rag"`)
- **Condition**: Experience Memory is empty, no candidates exceed the similarity threshold, or retrieved experiences fail validation (unapproved, stale document versions, polarity mismatch, scope mismatch).
- **Execution**: Standard Self-RAG pipeline generates an answer using solely the base document retriever.
- **Result Flags**: `memory_hit=False`, `memory_context_used=False`, `fallback_reason` populated.

---

## 2. Retrieval Budget Semantics

The parameter `k` in EvoRAG strictly governs **document retrieval**, not the combined candidate count:

- `base_retriever.search(query, k=k)` retrieves up to $k$ document passages from the document index.
- Up to `max_memory_context` validated experience passages are appended to the document passages.
- The total candidate count passed to Self-RAG's tree decoding is:
  $$\text{Total Passages} = |\text{Document Passages}| + |\text{Experience Passages}| \le k + \text{max\_memory\_context}$$
- **Non-displacement guarantee**: Document passages are **never** evicted, truncated, or displaced by memory passages.
  - For $k=5$ with 2 experiences: 5 document passages + 2 memory passages = 7 total passages.
  - For $k=5$ with 0 experiences: 5 document passages + 0 memory passages = 5 total passages.
  - For $k=1$ with 2 experiences: 1 document passage + 2 memory passages = 3 total passages.

---

## 3. Provenance Handling and Evidence Tracking

Every passage returned during retrieval carries an explicit `source_type`:
- Document evidence: `source_type="document"`
- Experience evidence: `source_type="experience_memory"`

### Provenance Extraction Rules (`_extract_provenance`)
1. **Document passages**:
   - `chunk_hash` is extracted directly from the deterministic SHA-256 chunk hash.
   - `source_version` is extracted directly from the source PDF byte hash.
2. **Experience passages**:
   - Original document chunk hashes are preserved from `original_chunk_hashes`.
   - Original document source versions are preserved from `original_source_versions`.
   - **Synthetic memory hashes excluded**: Synthetic memory chunk hashes (`mem_...`) and internal identifiers (`experience_memory`) are never treated as original document chunk hashes or source filenames.
3. **Deduplication**:
   - Chunk hashes and source versions are deduplicated while preserving source-to-version integrity.
4. **Resilience**:
   - Malformed, missing, or non-dict passage structures are handled safely without raising exceptions.
5. **Direct Reuse**:
   - Path A directly preserves stored `source_versions` and `chunk_hashes` without modification.

---

## 4. SQLite as Source of Truth & Recoverable FAISS Persistence

To prevent data loss or silent state corruption:

1. **SQLite as the Single Source of Truth**:
   - Experiences are persisted to SQLite with parameterized SQL queries.
   - Verification status, timestamps, evidence, chunk hashes, and source versions are recorded in SQLite.
2. **Recoverable FAISS Index Rebuilding**:
   - `approve_and_save()` writes to SQLite first.
   - If index rebuilding fails (e.g. disk full or FAISS error):
     - The SQLite record is preserved.
     - `self.last_index_error` records the failure.
     - The in-memory retriever index is invalidated (`retriever.invalidate_index()`) so it never queries a corrupted or out-of-sync index.
     - `MemoryIndexRebuildError` is raised with the experience ID and underlying cause.
3. **Explicit Recovery (`recover_index`)**:
   - `controller.recover_index()` queries all `APPROVED` experiences from SQLite, rebuilds the FAISS index and ID mapping from scratch, and reloads the retriever.
   - Repeated recovery calls or retries never duplicate records in SQLite (guaranteed by primary key and existence checks).

---

## 5. Verification & Testing

Run the test suite from the repository root:

```bash
# Run controller and integration tests (24 tests)
python -m pytest tests/test_controller.py -v

# Run experience memory subsystem tests (16 tests)
python -m pytest tests/test_memory.py -v

# Run complete repository test suite (65 tests)
python -m pytest tests/ -v
```

