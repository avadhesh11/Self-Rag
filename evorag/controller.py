"""Controller implementing Hybrid Experience Learning (Three-Path Routing)."""
from pathlib import Path
from typing import Any

from .config import DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD
from .memory.schema import Experience, VerificationStatus, normalize_question
from .memory.store import ExperienceStore
from .memory.retriever import ExperienceRetriever, ExperienceAugmentedRetriever
from .memory.validator import ExperienceValidator
from .memory.index_builder import ExperienceIndexBuilder
from .result import EvoRAGResult, ValidationResult
from selfrag.pipeline import self_rag_answer


class MemoryIndexRebuildError(RuntimeError):
    """Raised when SQLite persistence succeeds but FAISS index rebuilding fails."""

    def __init__(self, experience_id: str, message: str, cause: Exception | None = None):
        super().__init__(
            f"Experience '{experience_id}' was persisted to SQLite, but FAISS index rebuild failed: {message}"
        )
        self.experience_id = experience_id
        self.cause = cause


class EvoRAGController:
    """Orchestrates Hybrid Experience Learning with three distinct execution paths:

    1. Path A ('direct_reuse'): Exact safe normalized question match -> direct stored answer reuse.
    2. Path B ('experience_assisted'): Related validated experience -> injected as context into Self-RAG.
    3. Path C ('normal_self_rag'): No useful validated experience -> standard Self-RAG generation.
    """

    def __init__(
        self,
        store: ExperienceStore,
        memory_retriever: ExperienceRetriever,
        validator: ExperienceValidator,
        generator: Any = None,
        retriever: Any = None,
        doc_index_dir: str | Path | None = None,
        index_builder: ExperienceIndexBuilder | None = None,
        k: int = 5,
        max_segments: int = 3,
        memory_top_k: int = DEFAULT_TOP_K,
        max_memory_context: int = 2,
    ):
        self.store = store
        self.memory_retriever = memory_retriever
        self.validator = validator
        self.generator = generator
        self.retriever = retriever
        self.doc_index_dir = Path(doc_index_dir) if doc_index_dir else None
        self.index_builder = index_builder
        self.k = k
        self.max_segments = max_segments
        self.memory_top_k = memory_top_k
        self.max_memory_context = max_memory_context
        self.last_index_error: str | None = None

    def answer(self, question: str) -> EvoRAGResult:
        """Process a user question through three-path hybrid routing."""
        # 1. Retrieve semantic candidates from Experience Memory
        candidates = self.memory_retriever.retrieve(question, k=self.memory_top_k)
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)

        norm_q = normalize_question(question)

        # ---------------------------------------------------------
        # Path A: Exact Direct Reuse
        # A candidate is eligible for direct reuse ONLY if:
        # - Normalized question matches exactly
        # - Verification status is approved
        # - Validator checks pass (source freshness, polarity, scope, non-empty answer)
        # ---------------------------------------------------------
        exact_candidate = None
        for cand in candidates:
            if normalize_question(cand.experience.question) == norm_q:
                exact_candidate = cand
                break

        if exact_candidate is not None:
            val_res: ValidationResult
            if self.doc_index_dir:
                val_res = self.validator.validate_with_doc_index(
                    exact_candidate,
                    doc_index_dir=self.doc_index_dir,
                    query=question,
                )
            else:
                val_res = self.validator.validate(exact_candidate, query=question)

            if val_res.is_valid:
                # Path A Safe Memory Hit!
                exp = exact_candidate.experience
                doc_ev = [p for p in exp.evidence if p.get("source_type") != "experience_memory"]
                mem_ev = [p for p in exp.evidence if p.get("source_type") == "experience_memory"]

                return EvoRAGResult(
                    question=question,
                    answer=exp.answer,
                    memory_hit=True,
                    response_mode="direct_reuse",
                    memory_context_used=False,
                    experience_id=exp.experience_id,
                    similarity_score=exact_candidate.similarity_score,
                    memory_experiences=[],
                    memory_experience_ids=[],
                    document_evidence=doc_ev,
                    experience_evidence=mem_ev,
                    evidence=exp.evidence,
                    source_versions=exp.source_versions,
                    chunk_hashes=exp.chunk_hashes,
                    self_rag_result=None,
                    fallback_reason=None,
                )

        # ---------------------------------------------------------
        # Path B: Related Experience-Assisted Generation
        # If no exact match, filter and validate candidates for
        # semantic assistance into Self-RAG. Exclude unapproved,
        # stale, contradictory, polarity-mismatched, or scope-mismatched.
        # ---------------------------------------------------------
        valid_related = []
        best_invalid_reason = None
        if not candidates:
            best_invalid_reason = "No candidate experiences found in Experience Memory."

        for cand in candidates:
            val_res = (
                self.validator.validate_with_doc_index(cand, doc_index_dir=self.doc_index_dir, query=question)
                if self.doc_index_dir
                else self.validator.validate(cand, query=question)
            )
            if val_res.is_valid:
                valid_related.append(cand)
            elif best_invalid_reason is None:
                best_invalid_reason = val_res.reason

        if self.generator is None or self.retriever is None:
            raise ValueError(
                "Cannot invoke Self-RAG pipeline: generator and retriever must be provided "
                "when Experience Memory misses or assists."
            )

        selected_related = valid_related[: self.max_memory_context]

        if selected_related:
            # Path B Triggered: Inject validated memory context into Self-RAG
            augmented_retriever = ExperienceAugmentedRetriever(
                base_retriever=self.retriever,
                experiences=selected_related,
            )

            self_rag_res = self_rag_answer(
                question,
                self.generator,
                augmented_retriever,
                k=self.k,
                max_segments=self.max_segments,
            )

            evidence, chunk_hashes, source_versions = self._extract_provenance(self_rag_res)
            doc_ev = [p for p in evidence if p.get("source_type") != "experience_memory"]
            mem_ev = [p for p in evidence if p.get("source_type") == "experience_memory"]

            return EvoRAGResult(
                question=question,
                answer=self_rag_res.final_answer,
                memory_hit=False,
                response_mode="experience_assisted",
                memory_context_used=True,
                experience_id=None,
                similarity_score=selected_related[0].similarity_score,
                memory_experiences=[c.experience.to_dict() for c in selected_related],
                memory_experience_ids=[c.experience.experience_id for c in selected_related],
                document_evidence=doc_ev,
                experience_evidence=mem_ev,
                evidence=evidence,
                source_versions=source_versions,
                chunk_hashes=chunk_hashes,
                self_rag_result=self_rag_res,
                fallback_reason=None,
            )

        # ---------------------------------------------------------
        # Path C: Normal Self-RAG
        # No useful validated experiences found. Run standard Self-RAG.
        # ---------------------------------------------------------
        self_rag_res = self_rag_answer(
            question,
            self.generator,
            self.retriever,
            k=self.k,
            max_segments=self.max_segments,
        )

        evidence, chunk_hashes, source_versions = self._extract_provenance(self_rag_res)
        best_score = candidates[0].similarity_score if candidates else None

        return EvoRAGResult(
            question=question,
            answer=self_rag_res.final_answer,
            memory_hit=False,
            response_mode="normal_self_rag",
            memory_context_used=False,
            experience_id=None,
            similarity_score=best_score,
            memory_experiences=[],
            memory_experience_ids=[],
            document_evidence=evidence,
            experience_evidence=[],
            evidence=evidence,
            source_versions=source_versions,
            chunk_hashes=chunk_hashes,
            self_rag_result=self_rag_res,
            fallback_reason=best_invalid_reason,
        )

    def approve_and_save(
        self,
        result: EvoRAGResult,
        user_approved: bool = True,
    ) -> str | None:
        """Explicitly approve and store a generated result as a verified experience.

        Requirements:
        - Never saves if user_approved is False.
        - Never saves a direct memory-hit result as a new experience.
        - Only saves answers with non-empty text, supporting evidence, and source provenance.
        - Persists an approved experience to SQLite (the single source of truth).
        - Rebuilds or updates the semantic FAISS index.
        - If FAISS index rebuilding fails, raises MemoryIndexRebuildError, preserves the
          SQLite record, invalidates the in-memory retriever index, and records the error.
        """
        if not user_approved:
            return None

        if result.memory_hit or result.response_mode == "direct_reuse":
            # Direct reuse answers are already stored; do not duplicate
            return None

        if not result.answer or not result.answer.strip():
            return None

        if not result.evidence or not result.source_versions:
            # Requires usable evidence and source provenance
            return None

        exp = Experience(
            question=result.question,
            answer=result.answer,
            evidence=result.evidence,
            source_versions=result.source_versions,
            chunk_hashes=result.chunk_hashes,
            verification_status=VerificationStatus.APPROVED,
        )

        if not self.store.experience_exists(exp.experience_id):
            self.store.add_experience(exp)
        else:
            self.store.update_verification_status(exp.experience_id, VerificationStatus.APPROVED)

        # Rebuild or update the semantic FAISS index
        if self.index_builder is not None:
            try:
                self.index_builder.rebuild_from_store(self.store, status=VerificationStatus.APPROVED)
                if self.memory_retriever is not None:
                    self.memory_retriever._load_index()
                self.last_index_error = None
            except Exception as err:
                self.last_index_error = str(err)
                if self.memory_retriever is not None:
                    self.memory_retriever.invalidate_index()
                raise MemoryIndexRebuildError(
                    experience_id=exp.experience_id,
                    message=str(err),
                    cause=err,
                )

        return exp.experience_id

    def recover_index(self) -> dict[str, Any]:
        """Rebuild the FAISS index from the SQLite store (the single source of truth).

        Refreshes the memory retriever on success and clears last_index_error.
        Can be called during initialization, maintenance, or after an index rebuild failure.
        """
        if self.index_builder is None:
            raise ValueError("Cannot rebuild index: no index_builder provided to controller.")

        try:
            meta = self.index_builder.rebuild_from_store(self.store, status=VerificationStatus.APPROVED)
            if self.memory_retriever is not None:
                self.memory_retriever._load_index()
            self.last_index_error = None
            return meta
        except Exception as err:
            self.last_index_error = str(err)
            if self.memory_retriever is not None:
                self.memory_retriever.invalidate_index()
            raise MemoryIndexRebuildError(
                experience_id="recovery",
                message=str(err),
                cause=err,
            )

    def _extract_provenance(
        self,
        self_rag_res: Any,
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
        """Extract supporting passages, chunk hashes, and source versions from SelfRAGResult.

        Provenance extraction rules:
        1. Document passages retain their source version and document chunk hash.
        2. Experience passages contribute their original document source versions and
           original document chunk hashes.
        3. No synthetic memory chunk hash (e.g. 'mem_...') is treated as an original
           document chunk hash.
        4. The resulting provenance is deduplicated while preserving source-to-version mapping.
        5. Malformed or missing provenance fields are handled safely without raising exceptions.
        """
        evidence: list[dict[str, Any]] = []
        seen_keys: set[Any] = set()

        if hasattr(self_rag_res, "segments") and self_rag_res.segments:
            for seg in self_rag_res.segments:
                if getattr(seg, "selected", None) and getattr(seg.selected, "passage", None):
                    p = seg.selected.passage
                    if not isinstance(p, dict):
                        continue
                    key = p.get("chunk_hash") or p.get("chunk_id") or p.get("text")
                    if key not in seen_keys:
                        seen_keys.add(key)
                        evidence.append(p)

        if not evidence and getattr(self_rag_res, "selected", None):
            p = getattr(self_rag_res.selected, "passage", None)
            if isinstance(p, dict):
                evidence.append(p)

        if not evidence and getattr(self_rag_res, "candidates", None):
            for c in self_rag_res.candidates:
                p = getattr(c, "passage", None)
                if isinstance(p, dict):
                    key = p.get("chunk_hash") or p.get("chunk_id") or p.get("text")
                    if key not in seen_keys:
                        seen_keys.add(key)
                        evidence.append(p)

        chunk_hashes: list[str] = []
        seen_chunk_hashes: set[str] = set()
        source_versions: dict[str, str] = {}

        for p in evidence:
            if not isinstance(p, dict):
                continue

            is_memory = p.get("source_type") == "experience_memory"

            if is_memory:
                # 1. Experience passage: extract original document chunk hashes
                orig_hashes = p.get("original_chunk_hashes")
                if isinstance(orig_hashes, list):
                    for h in orig_hashes:
                        if isinstance(h, str) and h.strip() and not h.startswith("mem_"):
                            if h not in seen_chunk_hashes:
                                seen_chunk_hashes.add(h)
                                chunk_hashes.append(h)

                # 2. Experience passage: extract original document source versions
                orig_versions = p.get("original_source_versions")
                if isinstance(orig_versions, dict):
                    for src, ver in orig_versions.items():
                        if isinstance(src, str) and src.strip() and isinstance(ver, str) and ver.strip():
                            if src != "experience_memory":
                                source_versions[src] = ver
            else:
                # 3. Document passage: extract document chunk hash
                ch = p.get("chunk_hash")
                if isinstance(ch, str) and ch.strip() and not ch.startswith("mem_"):
                    if ch not in seen_chunk_hashes:
                        seen_chunk_hashes.add(ch)
                        chunk_hashes.append(ch)

                # 4. Document passage: extract document source version
                src = p.get("source")
                ver = p.get("source_version")
                if isinstance(src, str) and src.strip() and isinstance(ver, str) and ver.strip():
                    if src != "experience_memory":
                        source_versions[src] = ver

        return evidence, chunk_hashes, source_versions
