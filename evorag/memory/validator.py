"""Conservative validation layer for reusable Experience Memory candidates."""
import json
from pathlib import Path
import re
from typing import Any

from ..config import DEFAULT_SIMILARITY_THRESHOLD
from ..result import ExperienceCandidate, ValidationResult
from .schema import VerificationStatus

NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "cannot", "cant", "can't",
    "dont", "don't", "wont", "won't", "shouldnt", "shouldn't", "isnt", "isn't",
    "arent", "aren't", "wasnt", "wasn't", "werent", "weren't", "without", "exclude",
    "excluding", "except"
}

INTERROGATIVE_DIMENSIONS = {"who", "when", "where", "why"}


def _tokenize_words(text: str) -> set[str]:
    """Extract lowercased word tokens including contractions."""
    return set(re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text.lower()))


class ExperienceValidator:
    """Validates whether a retrieved experience candidate is safe to reuse without re-retrieval."""

    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold

    def validate(
        self,
        candidate: ExperienceCandidate,
        current_source_versions: dict[str, str] | None = None,
        query: str | None = None,
    ) -> ValidationResult:
        """Conservatively evaluate an experience candidate against validation criteria.

        Checks:
        1. Verification status is approved.
        2. Polarity and scope alignment (negation, numbers, interrogatives) when query is provided.
        3. Semantic similarity meets or exceeds threshold.
        4. Experience contains a non-empty answer.
        5. Experience contains structured evidence and source version records.
        6. Stored source versions match current document index versions (source freshness).

        TODO (Deeper Intent Verification): Future iterations may introduce a cross-encoder
        or NLI-based bidirectional entailment model for complex sentence-level semantics.
        """
        exp = candidate.experience

        # 1. Verification status check
        if exp.verification_status != VerificationStatus.APPROVED:
            return ValidationResult(
                is_valid=False,
                reason=f"Candidate experience '{exp.experience_id}' is not approved (status: '{exp.verification_status}').",
                candidate=candidate,
            )

        # 2. Polarity and lightweight scope mismatch protection (when query is provided)
        # Never reuse a candidate simply because similarity is high if polarity/scope opposes.
        if query is not None:
            tokens_q = _tokenize_words(query)
            tokens_exp = _tokenize_words(exp.question)

            # Polarity / Negation check
            has_neg_q = bool(tokens_q & NEGATION_WORDS)
            has_neg_exp = bool(tokens_exp & NEGATION_WORDS)
            if has_neg_q != has_neg_exp:
                return ValidationResult(
                    is_valid=False,
                    reason=(
                        f"Polarity mismatch: query {'contains' if has_neg_q else 'lacks'} negation "
                        f"while candidate {'contains' if has_neg_exp else 'lacks'} negation."
                    ),
                    candidate=candidate,
                )

            # Numeric scope check
            nums_q = set(re.findall(r"\b\d+\b", query))
            nums_exp = set(re.findall(r"\b\d+\b", exp.question))
            if nums_q and nums_exp and nums_q != nums_exp:
                return ValidationResult(
                    is_valid=False,
                    reason=(
                        f"Scope mismatch: query references numeric entity {sorted(nums_q)} "
                        f"while candidate references {sorted(nums_exp)}."
                    ),
                    candidate=candidate,
                )

            # Fundamental interrogative dimension check (who / when / where / why)
            int_q = tokens_q & INTERROGATIVE_DIMENSIONS
            int_exp = tokens_exp & INTERROGATIVE_DIMENSIONS
            if int_q and int_exp and int_q != int_exp:
                return ValidationResult(
                    is_valid=False,
                    reason=(
                        f"Interrogative dimension mismatch: query focuses on {sorted(int_q)} "
                        f"while candidate focuses on {sorted(int_exp)}."
                    ),
                    candidate=candidate,
                )

        # 3. Similarity threshold check
        if candidate.similarity_score < self.similarity_threshold:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"Similarity score {candidate.similarity_score:.4f} is below required "
                    f"threshold {self.similarity_threshold:.4f}."
                ),
                candidate=candidate,
            )

        # 4. Non-empty answer check
        if not exp.answer or not exp.answer.strip():
            return ValidationResult(
                is_valid=False,
                reason=f"Candidate experience '{exp.experience_id}' contains an empty or whitespace answer.",
                candidate=candidate,
            )

        # 5. Evidence and source version availability check
        if not exp.evidence:
            return ValidationResult(
                is_valid=False,
                reason=f"Candidate experience '{exp.experience_id}' lacks supporting evidence passages.",
                candidate=candidate,
            )

        if not exp.source_versions:
            return ValidationResult(
                is_valid=False,
                reason=f"Candidate experience '{exp.experience_id}' lacks source version tracking metadata.",
                candidate=candidate,
            )

        # 6. Source version freshness / invalidation check
        if current_source_versions is not None:
            for source_name, stored_version in exp.source_versions.items():
                current_ver = current_source_versions.get(source_name)
                if current_ver is not None and current_ver != stored_version:
                    return ValidationResult(
                        is_valid=False,
                        reason=(
                            f"Experience invalidated by modified source document '{source_name}'. "
                            f"(Stored version {stored_version[:8]} != Current version {current_ver[:8]})."
                        ),
                        candidate=candidate,
                    )

        return ValidationResult(
            is_valid=True,
            reason="Experience candidate passed all validation checks.",
            candidate=candidate,
        )

    def validate_with_doc_index(
        self,
        candidate: ExperienceCandidate,
        doc_index_dir: str | Path,
        query: str | None = None,
    ) -> ValidationResult:
        """Validate candidate against current document index metadata on disk."""
        doc_index_dir = Path(doc_index_dir)
        meta_file = doc_index_dir / "meta.json"
        current_source_versions: dict[str, str] | None = None

        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                current_source_versions = meta.get("source_versions")
            except Exception:
                current_source_versions = None

        return self.validate(
            candidate,
            current_source_versions=current_source_versions,
            query=query,
        )
