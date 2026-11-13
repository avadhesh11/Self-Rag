"""
EvoRAG Task 8: DBMS & SQL Corpus Integration and Validation Test Suite.

Validates:
1. All 15 educational DBMS & SQL PDFs are present and discoverable.
2. Every PDF matches the exact SHA-256 checksum and page count in data/manifest.json.
3. The university_schema.sql file exists and is valid.
4. Dedicated DBMS document index at index/dbms_sql_v1 reloads cleanly with real FAISS.
5. Retrieval metadata preservation (source_version, chunk_hash, page, score).
6. Document index isolation from baseline document index and Experience Memory.
"""
import hashlib
import json
from pathlib import Path
import pytest
from pypdf import PdfReader

try:
    import faiss
    HAS_FAISS = hasattr(faiss, "IndexFlatIP")
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

from retriever.retriever import PassageRetriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
if not MANIFEST_PATH.exists():
    MANIFEST_PATH = PROJECT_ROOT / "EvoRAG_DBMS_SQL_15_Document_Corpus" / "manifest.json"
PDF_DIR = PROJECT_ROOT / "data" / "pdf"
if not PDF_DIR.exists():
    PDF_DIR = PROJECT_ROOT / "EvoRAG_DBMS_SQL_15_Document_Corpus" / "pdf"
SCHEMA_PATH = PROJECT_ROOT / "data" / "university_schema.sql"
if not SCHEMA_PATH.exists():
    SCHEMA_PATH = PROJECT_ROOT / "EvoRAG_DBMS_SQL_15_Document_Corpus" / "university_schema.sql"
DBMS_INDEX_DIR = PROJECT_ROOT / "index" / "dbms_sql_v1"

pytestmark = pytest.mark.integration


class TestDBMSCorpusManifestAndIntegrity:
    """Validates the 15-document corpus integrity against data/manifest.json."""

    def test_corpus_manifest_and_files_discoverability(self):
        """Verify that manifest.json lists 15 PDFs and all 15 exist on disk."""
        assert MANIFEST_PATH.exists(), f"Missing manifest at {MANIFEST_PATH}"
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest) == 15, f"Expected 15 documents in manifest, got {len(manifest)}"

        pdf_files = list(PDF_DIR.glob("*.pdf"))
        assert len(pdf_files) == 15, f"Expected 15 PDF files in {PDF_DIR}, got {len(pdf_files)}"

        for entry in manifest:
            pdf_path = PDF_DIR / entry["filename"]
            assert pdf_path.exists(), f"Corpus PDF {entry['filename']} missing from {PDF_DIR}"

    def test_manifest_checksums_and_page_counts(self):
        """Verify every PDF matches the SHA-256 checksum and page count from the manifest."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for entry in manifest:
            pdf_path = PDF_DIR / entry["filename"]
            with open(pdf_path, "rb") as f:
                actual_sha = hashlib.sha256(f.read()).hexdigest()

            assert actual_sha == entry["sha256"], (
                f"SHA-256 mismatch for {entry['filename']}: "
                f"expected {entry['sha256']}, got {actual_sha}"
            )

            reader = PdfReader(pdf_path)
            actual_pages = len(reader.pages)
            assert actual_pages == entry["pages"], (
                f"Page count mismatch for {entry['filename']}: "
                f"expected {entry['pages']}, got {actual_pages}"
            )

    def test_all_pdfs_contain_extractable_text(self):
        """Verify every PDF contains text extractable by pypdf (not scanned images)."""
        pdf_files = sorted(list(PDF_DIR.glob("*.pdf")))
        assert len(pdf_files) == 15

        for pdf_path in pdf_files:
            reader = PdfReader(pdf_path)
            total_text = ""
            for page in reader.pages:
                txt = page.extract_text() or ""
                total_text += txt.strip()

            assert len(total_text) > 500, (
                f"PDF {pdf_path.name} has suspiciously low text length ({len(total_text)} chars)"
            )

    def test_university_schema_exists(self):
        """Verify the PostgreSQL university_schema.sql exists and contains expected DDL."""
        assert SCHEMA_PATH.exists(), f"Schema missing at {SCHEMA_PATH}"
        content = SCHEMA_PATH.read_text(encoding="utf-8")
        assert "CREATE TABLE departments" in content
        assert "CREATE TABLE students" in content
        assert "CREATE TABLE courses" in content
        assert "CREATE TABLE enrollments" in content


class TestDBMSIndexReloadAndRetrieval:
    """Validates loading and querying the dedicated DBMS FAISS index."""

    @pytest.fixture(scope="class")
    @classmethod
    def dbms_retriever(cls):
        """Load PassageRetriever for the dedicated DBMS index."""
        if not HAS_FAISS or not HAS_ST:
            pytest.skip("FAISS or SentenceTransformer not available")
        assert DBMS_INDEX_DIR.exists(), f"DBMS index directory {DBMS_INDEX_DIR} does not exist"
        return PassageRetriever(str(DBMS_INDEX_DIR))

    def test_dbms_index_files_and_metadata(self):
        """Verify that index files exist and meta.json contains all 15 sources."""
        assert (DBMS_INDEX_DIR / "faiss.index").exists()
        assert (DBMS_INDEX_DIR / "corpus.jsonl").exists()
        assert (DBMS_INDEX_DIR / "meta.json").exists()

        meta = json.loads((DBMS_INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
        assert meta["num_chunks"] == 232
        assert len(meta["sources"]) == 15
        assert len(meta["source_versions"]) == 15

    def test_dbms_index_reload_and_chunk_count(self, dbms_retriever):
        """Verify PassageRetriever loads all 232 chunks and FAISS index with correct dimension."""
        assert len(dbms_retriever.corpus) == 232
        assert dbms_retriever.index.ntotal == 232
        assert dbms_retriever.index.d == 384

    @pytest.mark.parametrize("query,expected_doc", [
        ("What is physical data independence and why do database systems exist?", "01_dbms_fundamentals_and_architecture.pdf"),
        ("What is the difference between a candidate key and a primary key in relational model?", "02_relational_model_keys_and_integrity.pdf"),
        ("How are weak entities and cardinality ratios modeled in ER diagrams?", "03_er_modeling_and_database_design.pdf"),
        ("What are the fundamental relational algebra operations like select, project, and cartesian product?", "04_relational_algebra_and_calculus.pdf"),
        ("How do identity columns and CREATE TABLE work in PostgreSQL DDL?", "05_sql_fundamentals_ddl_dml_and_types.pdf"),
        ("How do INNER JOIN, LEFT OUTER JOIN, and GROUP BY HAVING work in SQL?", "06_sql_queries_joins_aggregation_subqueries.pdf"),
        ("What is Boyce-Codd Normal Form BCNF and functional dependencies?", "07_functional_dependencies_and_normalization.pdf"),
        ("How does the buffer pool manager and slotted page architecture work?", "08_database_storage_pages_and_buffer_management.pdf"),
        ("How do B-tree indexes, GIN, and BRIN indexes work in PostgreSQL?", "09_index_structures_btrees_hashing_and_postgresql.pdf"),
        ("What are ACID properties and transaction isolation levels?", "10_transactions_and_acid_properties.pdf"),
        ("How does Multi-Version Concurrency Control MVCC and two-phase locking work?", "11_concurrency_control_serializability_mvcc.pdf"),
        ("How does Write-Ahead Logging WAL and ARIES checkpointing work?", "12_database_recovery_logging_wal_and_backup.pdf"),
        ("What are nested loop join, hash join, and sort-merge join execution algorithms?", "13_query_processing_and_execution_algorithms.pdf"),
        ("How does the query optimizer use cost models and EXPLAIN ANALYZE statistics?", "14_query_optimization_cost_models_and_explain.pdf"),
        ("What is the CAP theorem, Role-Based Access Control RBAC, and Row Level Security in PostgreSQL?", "15_distributed_databases_nosql_security_and_administration.pdf"),
    ])
    def test_smoke_retrieval_targets_all_15_documents(self, dbms_retriever, query, expected_doc):
        """Verify that targeted queries retrieve passages from the expected document."""
        results = dbms_retriever.search(query, k=3)
        assert len(results) >= 1
        top_hit = results[0]
        assert top_hit["source"] == expected_doc, (
            f"Query '{query}' expected top hit from {expected_doc}, got {top_hit['source']}"
        )
        assert top_hit["retrieval_score"] > 0.40, f"Score too low: {top_hit['retrieval_score']}"

    def test_retrieval_provenance_metadata_preservation(self, dbms_retriever):
        """Verify retrieved passages preserve full provenance fields without synthetic memory IDs."""
        results = dbms_retriever.search("ACID transactions atomicity", k=2)
        assert len(results) == 2

        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_lookup = {item["filename"]: item["sha256"] for item in json.load(f)}

        for hit in results:
            assert "source" in hit
            assert "source_version" in hit
            assert "chunk_hash" in hit
            assert "chunk_id" in hit
            assert "page" in hit
            assert "retrieval_score" in hit

            # Must match the manifest's SHA-256 for this file
            expected_sha = manifest_lookup[hit["source"]]
            assert hit["source_version"] == expected_sha

            # Chunk hash must be 64-char hex and must NOT be a synthetic memory ID
            assert len(hit["chunk_hash"]) == 64
            assert not hit["chunk_hash"].startswith("mem_")

            # Page must be 1-based between 1 and 6
            assert 1 <= hit["page"] <= 6

    def test_document_index_isolation(self, dbms_retriever):
        """Verify the DBMS document index is isolated from Experience Memory and base index."""
        # 1. Sources in DBMS index must only be the 15 DBMS PDFs
        meta = json.loads((DBMS_INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
        for src in meta["sources"]:
            assert src.endswith(".pdf")
            assert src.startswith(("01_", "02_", "03_", "04_", "05_", "06_", "07_",
                                  "08_", "09_", "10_", "11_", "12_", "13_", "14_", "15_"))

        # 2. No baseline sample.pdf or 2310.11511v1.pdf in DBMS index
        assert "sample.pdf" not in meta["sources"]
        assert "2310.11511v1.pdf" not in meta["sources"]

        # 3. Dedicated index path is distinct from baseline index path
        assert DBMS_INDEX_DIR != PROJECT_ROOT / "index"
        assert (PROJECT_ROOT / "index" / "faiss.index").exists()
