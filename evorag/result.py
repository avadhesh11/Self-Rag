"""Data structures representing retrieval, validation, and execution results for EvoRAG."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory.schema import Experience


@dataclass
class ExperienceCandidate:
    """A retrieved experience paired with its semantic similarity score."""
    experience: "Experience"
    similarity_score: float

    def to_dict(self) -> dict[str, Any]:
        exp_dict = (
            self.experience.to_dict()
            if hasattr(self.experience, "to_dict")
            else dict(self.experience)
        )
        return {
            "experience": exp_dict,
            "similarity_score": self.similarity_score,
        }


@dataclass
class ValidationResult:
    """Outcome of validating an experience candidate for safe reuse."""
    is_valid: bool
    reason: str
    candidate: ExperienceCandidate | None = None


@dataclass
class EvoRAGResult:
    """Unified result container returned by the EvoRAG controller.

    Supports three execution paths:
    1. 'direct_reuse': Exact safe match; stored answer returned directly.
    2. 'experience_assisted': Related valid experience injected as context into Self-RAG.
    3. 'normal_self_rag': No usable experience; standard Self-RAG generation.
    """
    question: str
    answer: str
    memory_hit: bool
    response_mode: str = "normal_self_rag"  # "direct_reuse" | "experience_assisted" | "normal_self_rag"
    memory_context_used: bool = False
    experience_id: str | None = None
    similarity_score: float | None = None
    memory_experiences: list[dict[str, Any]] = field(default_factory=list)
    memory_experience_ids: list[str] = field(default_factory=list)
    document_evidence: list[dict[str, Any]] = field(default_factory=list)
    experience_evidence: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    source_versions: dict[str, str] = field(default_factory=dict)
    chunk_hashes: list[str] = field(default_factory=list)
    self_rag_result: Any | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "memory_hit": self.memory_hit,
            "response_mode": self.response_mode,
            "memory_context_used": self.memory_context_used,
            "experience_id": self.experience_id,
            "similarity_score": self.similarity_score,
            "memory_experiences": self.memory_experiences,
            "memory_experience_ids": self.memory_experience_ids,
            "document_evidence": self.document_evidence,
            "experience_evidence": self.experience_evidence,
            "evidence": self.evidence,
            "source_versions": self.source_versions,
            "chunk_hashes": self.chunk_hashes,
            "fallback_reason": self.fallback_reason,
        }
