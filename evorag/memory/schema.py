"""Data models and schema definitions for EvoRAG Experience Memory."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any
import unicodedata


class VerificationStatus:
    """Allowed verification status values."""
    APPROVED = "approved"
    PENDING = "pending"
    REJECTED = "rejected"
    ALL = (APPROVED, PENDING, REJECTED)


def normalize_question(text: str) -> str:
    """Normalize question text for consistent matching and hashing (NFKC + collapsed whitespace + lowercased)."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split()).strip().lower()


def generate_experience_id(normalized_question: str, answer: str) -> str:
    """Generate a stable, deterministic SHA-256 identifier for an experience."""
    norm_q = normalized_question.strip().lower()
    norm_a = answer.strip()
    payload = f"{norm_q}\x00{norm_a}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Experience:
    """Representation of a verified question-answer experience and its provenance."""
    question: str
    answer: str
    normalized_question: str = ""
    experience_id: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    source_versions: dict[str, str] = field(default_factory=dict)
    chunk_hashes: list[str] = field(default_factory=list)
    verification_status: str = VerificationStatus.APPROVED
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.normalized_question:
            self.normalized_question = normalize_question(self.question)
        if not self.experience_id:
            self.experience_id = generate_experience_id(self.normalized_question, self.answer)
        now = utc_now_iso()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        """Convert experience to a serializable dictionary."""
        return {
            "experience_id": self.experience_id,
            "question": self.question,
            "normalized_question": self.normalized_question,
            "answer": self.answer,
            "evidence": self.evidence,
            "source_versions": self.source_versions,
            "chunk_hashes": self.chunk_hashes,
            "verification_status": self.verification_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experience":
        """Construct Experience from a dictionary."""
        return cls(
            experience_id=data.get("experience_id", ""),
            question=data.get("question", ""),
            normalized_question=data.get("normalized_question", ""),
            answer=data.get("answer", ""),
            evidence=data.get("evidence", []) if isinstance(data.get("evidence"), list) else [],
            source_versions=data.get("source_versions", {}) if isinstance(data.get("source_versions"), dict) else {},
            chunk_hashes=data.get("chunk_hashes", []) if isinstance(data.get("chunk_hashes"), list) else [],
            verification_status=data.get("verification_status", VerificationStatus.APPROVED),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    @classmethod
    def from_row(cls, row: tuple) -> "Experience":
        """Construct Experience from an SQLite row tuple."""
        (
            exp_id,
            q,
            norm_q,
            ans,
            ev_json,
            sv_json,
            ch_json,
            status,
            created,
            updated,
        ) = row
        return cls(
            experience_id=exp_id,
            question=q,
            normalized_question=norm_q,
            answer=ans,
            evidence=json.loads(ev_json) if ev_json else [],
            source_versions=json.loads(sv_json) if sv_json else {},
            chunk_hashes=json.loads(ch_json) if ch_json else [],
            verification_status=status,
            created_at=created,
            updated_at=updated,
        )

