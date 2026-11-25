"""SQLite persistence for loan application review history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import settings


class ReviewHistoryStore:
    """Stores completed and interrupted loan review records in SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or settings.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_history (
                    thread_id TEXT PRIMARY KEY,
                    application_id TEXT NOT NULL,
                    applicant_name TEXT NOT NULL,
                    loan_type TEXT NOT NULL,
                    loan_amount REAL DEFAULT 0,
                    final_status TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    low_confidence_warning INTEGER DEFAULT 0,
                    human_decision TEXT,
                    human_reviewer TEXT,
                    human_comments TEXT,
                    application_json TEXT NOT NULL,
                    structured_output_json TEXT NOT NULL,
                    execution_trace_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_review(
        self,
        thread_id: str,
        application: dict[str, Any],
        structured_output: dict[str, Any],
        execution_trace: list[str],
    ) -> None:
        """Insert or update a loan review record."""
        now = datetime.now(timezone.utc).isoformat()
        created_at = structured_output.get("created_at") or now

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO review_history (
                    thread_id, application_id, applicant_name, loan_type, loan_amount,
                    final_status, recommendation, confidence, low_confidence_warning,
                    human_decision, human_reviewer, human_comments,
                    application_json, structured_output_json, execution_trace_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    final_status = excluded.final_status,
                    recommendation = excluded.recommendation,
                    confidence = excluded.confidence,
                    low_confidence_warning = excluded.low_confidence_warning,
                    human_decision = excluded.human_decision,
                    human_reviewer = excluded.human_reviewer,
                    human_comments = excluded.human_comments,
                    application_json = excluded.application_json,
                    structured_output_json = excluded.structured_output_json,
                    execution_trace_json = excluded.execution_trace_json,
                    updated_at = excluded.updated_at
                """,
                (
                    thread_id,
                    structured_output.get("application_id", application.get("application_id", "UNKNOWN")),
                    structured_output.get("applicant_name", application.get("full_name", "Unknown")),
                    structured_output.get("loan_type", application.get("loan_type", "personal")),
                    float(application.get("loan_amount", 0) or 0),
                    structured_output.get("final_status", "PENDING_HUMAN_REVIEW"),
                    structured_output.get("recommendation", "MANUAL_REVIEW"),
                    float(structured_output.get("confidence", 0.0)),
                    1 if structured_output.get("low_confidence_warning") else 0,
                    structured_output.get("human_decision"),
                    structured_output.get("human_reviewer"),
                    structured_output.get("human_comments"),
                    json.dumps(application),
                    json.dumps(structured_output),
                    json.dumps(execution_trace),
                    created_at,
                    now,
                ),
            )
            conn.commit()

    def list_reviews(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent loan reviews sorted newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM review_history
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["low_confidence_warning"] = bool(item["low_confidence_warning"])
            item["application"] = json.loads(item.pop("application_json"))
            item["structured_output"] = json.loads(item.pop("structured_output_json"))
            item["execution_trace"] = json.loads(item.pop("execution_trace_json"))
            results.append(item)
        return results

    def get_review(self, thread_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single review record by thread_id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_history WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["low_confidence_warning"] = bool(item["low_confidence_warning"])
        item["application"] = json.loads(item.pop("application_json"))
        item["structured_output"] = json.loads(item.pop("structured_output_json"))
        item["execution_trace"] = json.loads(item.pop("execution_trace_json"))
        return item
