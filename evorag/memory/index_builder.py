"""FAISS semantic index builder for EvoRAG Experience Memory."""
import json
from pathlib import Path
from typing import Any

try:
    import faiss
except ImportError:
    faiss = None
import numpy as np

from ..config import DEFAULT_EMBED_MODEL, DEFAULT_INDEX_DIR
from .schema import Experience, VerificationStatus
from .store import ExperienceStore

INDEX_FILE_NAME = "experience_faiss.index"
META_FILE_NAME = "experience_meta.json"
DEFAULT_EMBED_DIM = 384


class ExperienceIndexBuilder:
    """Builds and persists a FAISS vector index over verified experiences."""

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        embedder: Any = None,
    ):
        self.index_dir = Path(index_dir)
        self.embed_model_name = embed_model_name
        self._embedder = embedder

    def _get_embedder(self) -> Any:
        """Lazily initialize SentenceTransformer if no embedder was injected."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized float32 vectors."""
        embedder = self._get_embedder()
        # Support objects with .encode() or callable embedders
        if hasattr(embedder, "encode"):
            embeddings = embedder.encode(texts, normalize_embeddings=True)
        else:
            embeddings = embedder(texts)

        embeddings = np.asarray(embeddings, dtype="float32")
        # Ensure L2 normalization for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return embeddings / norms

    def build_from_experiences(self, experiences: list[Experience]) -> dict[str, Any]:
        """Build FAISS index and ID mapping from a list of experiences and persist to disk.

        Maintains an explicit vector position -> experience_id mapping. Never assumes
        FAISS vector position equals SQLite row ID.
        """
        if faiss is None:
            raise ImportError(
                "FAISS is required for Experience Memory indexing. Install faiss-cpu or faiss-gpu."
            )
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir / INDEX_FILE_NAME
        meta_path = self.index_dir / META_FILE_NAME

        # Filter to approved experiences if not already filtered
        approved_exps = [e for e in experiences if e.verification_status == VerificationStatus.APPROVED]

        if not approved_exps:
            # Handle empty experiences safely: build an empty index
            dim = DEFAULT_EMBED_DIM
            faiss_index = faiss.IndexFlatIP(dim)
            faiss.write_index(faiss_index, str(index_path))

            meta = {
                "embed_model": self.embed_model_name,
                "dimension": dim,
                "num_vectors": 0,
                "id_map": [],
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return meta

        texts = [e.normalized_question for e in approved_exps]
        embeddings = self._encode_texts(texts)
        dim = embeddings.shape[1]

        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, str(index_path))

        id_map = [e.experience_id for e in approved_exps]
        meta = {
            "embed_model": self.embed_model_name,
            "dimension": dim,
            "num_vectors": len(id_map),
            "id_map": id_map,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def rebuild_from_store(
        self,
        store: ExperienceStore,
        status: str = VerificationStatus.APPROVED,
    ) -> dict[str, Any]:
        """Query experiences with the given verification status from SQLite and rebuild the index."""
        experiences = store.list_experiences(status=status)
        return self.build_from_experiences(experiences)

