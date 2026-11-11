# EvoRAG DBMS & SQL Knowledge Corpus

## Contents
This bundle contains 15 standalone, text-extractable PDF educational documents, a PostgreSQL sample schema, and a manifest with page counts and SHA-256 checksums. Each PDF is designed for document ingestion: headings, paragraphs, tables and code examples are selectable text, not scanned page images.

## Technical scope
- Database theory is presented vendor-neutrally where possible.
- Product-specific SQL, commands, and behavior use PostgreSQL 18 documentation as the baseline (research snapshot: PostgreSQL 18.6, 6 November 2026).
- Sources include PostgreSQL's official documentation, Carnegie Mellon University's 15-445/645 Fall 2026 database-systems materials, and the CC BY 4.0 BCcampus Database Design open textbook.
- Every PDF contains topic-specific source URLs and review questions.

## Use with the existing EvoRAG project
1. Extract the PDFs into the existing repository's configured PDF input folder (the implementation plan calls this `data/`; use the actual path configured by your current `index/build_index.py`).
2. Keep the existing PDF reader, chunker, embedding model, FAISS index builder, `PassageRetriever`, and Self-RAG pipeline unchanged.
3. Run the existing index builder to create a fresh document index from the complete PDF folder. Do not mix old and new index artifacts unless the builder is designed to rebuild or append safely.
4. Load the resulting index with the existing `PassageRetriever` and run smoke-test queries across several documents.
5. Preserve each document filename, page number, and source metadata in the corpus records so answers can cite their source. The manifest's SHA-256 values can be used as source-version identifiers.

No new vector database or embedding format is introduced: these are ordinary text PDFs intended for the current PDF-to-embedding-to-FAISS path. The document index remains separate from EvoRAG's Experience Memory index.

## Important distinctions
These PDFs are a retrieval knowledge base, not a fine-tuning dataset and not legal/official product specifications. They are original synthesized educational notes based on cited references, not substitutes for the full references. Verify answers against the linked primary sources when exact implementation behavior matters. SQL examples are written for PostgreSQL 18; this generation environment did not execute them against a live PostgreSQL server, so run the supplied examples in a disposable PostgreSQL 18 database before production use.

## Files
- `pdf/`: 15 standalone educational PDFs.
- `university_schema.sql`: small relational schema for experimenting with SQL examples.
- `manifest.json`: document IDs, page counts, text-extraction counts, baseline and SHA-256 checksums.
