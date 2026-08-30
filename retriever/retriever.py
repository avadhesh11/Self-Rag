"""
Thin retriever wrapper: embed a query, search the FAISS index built by
index/build_index.py, return top-k passages with their text + provenance.

Kept deliberately dumb on purpose -- this is the component you're most
likely to swap out later (different embedding model, a real vector DB,
hybrid BM25 + dense, reranking, etc.), so it exposes exactly one method
that the rest of the pipeline depends on: `.search(query, k)`.
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class PassageRetriever:
    def __init__(self, index_dir: str = "index"):
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text())
        self.embed_model_name = meta["embed_model"]

        self.model = SentenceTransformer(self.embed_model_name)
        self.index = faiss.read_index(str(index_dir / "faiss.index"))

        self.corpus = []
        with open(index_dir / "corpus.jsonl") as f:
            for line in f:
                self.corpus.append(json.loads(line))

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return top-k passages as dicts: {text, page, chunk_id, score}."""
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, indices = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            passage = dict(self.corpus[idx])
            passage["retrieval_score"] = float(score)
            results.append(passage)
        return results


if __name__ == "__main__":
    # Quick manual smoke test: python retriever/retriever.py "your query here"
    import sys
    retriever = PassageRetriever("index")
    query = sys.argv[1] if len(sys.argv) > 1 else "How large is the Amazon rainforest?"
    for r in retriever.search(query, k=3):
        print(f"[page {r['page']}, score {r['retrieval_score']:.3f}] {r['text'][:120]}...")
