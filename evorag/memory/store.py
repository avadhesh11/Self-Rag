"""SQLite persistent storage for EvoRAG Experience Memory."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from .schema import Experience, VerificationStatus, utc_now_iso


class ExperienceStore:
    """Manages persistent SQLite storage of question-answer experiences and their provenance."""

    def __init__(self, db_path: str | Path = "data/experience_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    experience_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    source_versions TEXT NOT NULL,
                    chunk_hashes TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_exp_norm_question
                ON experiences (normalized_question);
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_exp_status
                ON experiences (verification_status);
                """
            )

    def add_experience(self, exp: Experience) -> str:
        """Insert a new experience into the store using parameterized queries.

        Raises:
            ValueError: If an experience with the same experience_id already exists.
        """
        if self.experience_exists(exp.experience_id):
            raise ValueError(f"Experience with ID {exp.experience_id} already exists.")

        evidence_json = json.dumps(exp.evidence, ensure_ascii=False)
        source_versions_json = json.dumps(exp.source_versions, ensure_ascii=False)
        chunk_hashes_json = json.dumps(exp.chunk_hashes, ensure_ascii=False)

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO experiences (
                    experience_id, question, normalized_question, answer,
                    evidence, source_versions, chunk_hashes,
                    verification_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    exp.experience_id,
                    exp.question,
                    exp.normalized_question,
                    exp.answer,
                    evidence_json,
                    source_versions_json,
                    chunk_hashes_json,
                    exp.verification_status,
                    exp.created_at,
                    exp.updated_at,
                ),
            )
        return exp.experience_id

    def get_experience(self, experience_id: str) -> Experience | None:
        """Fetch an experience by its unique experience_id."""
        cursor = self._conn.execute(
            """
            SELECT experience_id, question, normalized_question, answer,
                   evidence, source_versions, chunk_hashes,
                   verification_status, created_at, updated_at
            FROM experiences
            WHERE experience_id = ?;
            """,
            (experience_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return Experience.from_row(tuple(row))

    def experience_exists(self, experience_id: str) -> bool:
        """Check whether an experience with the given ID exists."""
        cursor = self._conn.execute(
            "SELECT 1 FROM experiences WHERE experience_id = ? LIMIT 1;",
            (experience_id,),
        )
        return cursor.fetchone() is not None

    def list_experiences(
        self,
        status: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Experience]:
        """List experiences with optional status filtering and pagination."""
        query = """
            SELECT experience_id, question, normalized_question, answer,
                   evidence, source_versions, chunk_hashes,
                   verification_status, created_at, updated_at
            FROM experiences
        """
        params: list[Any] = []
        if status is not None:
            query += " WHERE verification_status = ?"
            params.append(status)

        query += " ORDER BY created_at ASC"

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()
        return [Experience.from_row(tuple(r)) for r in rows]

    def update_verification_status(self, experience_id: str, status: str) -> bool:
        """Update the verification status of an experience."""
        now = utc_now_iso()
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE experiences
                SET verification_status = ?, updated_at = ?
                WHERE experience_id = ?;
                """,
                (status, now, experience_id),
            )
            return cursor.rowcount > 0

    def delete_experience(self, experience_id: str) -> bool:
        """Delete an experience by its experience_id."""
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM experiences WHERE experience_id = ?;",
                (experience_id,),
            )
            return cursor.rowcount > 0

    def count(self, status: str | None = None) -> int:
        """Count total experiences, optionally filtered by verification status."""
        query = "SELECT COUNT(*) FROM experiences"
        params: list[Any] = []
        if status is not None:
            query += " WHERE verification_status = ?"
            params.append(status)
        cursor = self._conn.execute(query, params)
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the SQLite database connection."""
        self._conn.close()

    def __enter__(self) -> "ExperienceStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

