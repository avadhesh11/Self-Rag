"""
Chunk a PDF into passage-sized units and build a FAISS index over them.

Self-RAG retrieves at the *passage* level (roughly paragraph-sized chunks),
not arbitrary token windows, so we chunk paragraph-aligned and merge short
paragraphs together to reach a target size. This keeps each retrieved unit
coherent enough for the [ISREL]/[ISSUP] critique tokens to make a sane
judgment about it.

Usage:
    python index/build_index.py --pdf data/sample.pdf --out index/
"""
import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Any sentence-transformers model works here. The original paper uses
# Contriever-MSMARCO; we default to a small general-purpose model so this
# runs fast on CPU for demo purposes. Swap this out later without touching
# anything else in the pipeline -- that's the whole point of the modularity.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def extract_paragraphs(pdf_path: str) -> list[dict]:
    """Pull text per page and split into sentence-level units.

    Most PDF extractors (including pypdf) do NOT reliably preserve blank-line
    paragraph breaks -- multi_cell/wrapped text often comes back as one
    newline-per-line blob regardless of visual paragraph structure. Splitting
    on blank lines is therefore fragile in practice and can silently collapse
    an entire page into a single "paragraph". Sentence-level splitting is a
    more robust unit to merge upward from, for arbitrary real-world PDFs.
    """
    reader = PdfReader(pdf_path)
    paragraphs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        flat = " ".join(text.split())  # collapse all whitespace/newlines
        if not flat:
            continue
        sentences = _SENTENCE_SPLIT_RE.split(flat)
        for sent in sentences:
            cleaned = sent.strip()
            if cleaned:
                paragraphs.append({"text": cleaned, "page": page_num})
    return paragraphs


def merge_to_target_size(paragraphs: list[dict], target_words: int = 40) -> list[dict]:
    """Merge short consecutive paragraphs (same page) up to ~target_words."""
    chunks = []
    buf_text, buf_page, buf_words = [], None, 0

    def flush():
        if buf_text:
            chunks.append({"text": " ".join(buf_text), "page": buf_page})

    for para in paragraphs:
        words = len(para["text"].split())
        if buf_text and (buf_words + words > target_words * 1.5 or para["page"] != buf_page):
            flush()
            buf_text, buf_page, buf_words = [], None, 0
        buf_text.append(para["text"])
        buf_page = para["page"] if buf_page is None else buf_page
        buf_words += words
        if buf_words >= target_words:
            flush()
            buf_text, buf_page, buf_words = [], None, 0
    flush()
    return chunks


def build_index(pdf_path: str, out_dir: str, embed_model_name: str = DEFAULT_EMBED_MODEL,
                 target_words: int = 40):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Extracting text from {pdf_path} ...")
    paragraphs = extract_paragraphs(pdf_path)
    print(f"      -> {len(paragraphs)} raw sentences")

    print("[2/4] Merging into passage-sized chunks ...")
    chunks = merge_to_target_size(paragraphs, target_words=target_words)
    for i, c in enumerate(chunks):
        c["chunk_id"] = i
    print(f"      -> {len(chunks)} passages")

    print(f"[3/4] Embedding passages with {embed_model_name} ...")
    model = SentenceTransformer(embed_model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    print("[4/4] Building FAISS index (inner product on normalized vectors = cosine) ...")
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, str(out_dir / "faiss.index"))
    with open(out_dir / "corpus.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"embed_model": embed_model_name, "num_chunks": len(chunks)}, f, indent=2)

    print(f"Done. Wrote {out_dir/'faiss.index'} and {out_dir/'corpus.jsonl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to source PDF")
    ap.add_argument("--out", default="index", help="Output directory for index files")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--target-words", type=int, default=40,
                     help="Approx. words per passage before merging stops")
    args = ap.parse_args()
    build_index(args.pdf, args.out, args.embed_model, args.target_words)
