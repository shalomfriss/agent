"""Persistent conversation-session bookkeeping."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path


class SessionManager:
    """Select and track checkpoint threads for one project directory."""

    def __init__(self, connection: sqlite3.Connection, project_root: Path):
        self._connection = connection
        self._project_root = str(project_root)
        self._create_tables()

    def activate(
        self,
        requested_id: str | None = None,
        *,
        force_new: bool = False,
    ) -> tuple[str, bool]:
        """Make a requested, new, or previously active session current.

        The returned boolean says whether the selected session already existed.
        """

        requested_id = self._normalize_session_id(requested_id)
        session_id = requested_id

        if session_id is None and not force_new:
            session_id = self._active_session_id()
        if session_id is None:
            session_id = str(uuid.uuid4())

        already_existed = self._session_exists(session_id)
        self._save_session(session_id)
        self._set_active_session(session_id)
        self._connection.commit()
        return session_id, already_existed

    def list_sessions(self) -> list[tuple[str, str]]:
        """Return session IDs and last-used timestamps, newest first."""

        return self._connection.execute(
            """
            SELECT thread_id, updated_at
            FROM agent_sessions
            WHERE project_root = ?
            ORDER BY updated_at DESC
            """,
            (self._project_root,),
        ).fetchall()

    def touch(self, session_id: str) -> None:
        """Mark a session as recently used after processing a prompt."""

        self._connection.execute(
            """
            UPDATE agent_sessions
            SET updated_at = ?
            WHERE project_root = ? AND thread_id = ?
            """,
            (self._now(), self._project_root, session_id),
        )
        self._connection.commit()

    def _create_tables(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_sessions (
                project_root TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (project_root, thread_id)
            );

            CREATE TABLE IF NOT EXISTS active_agent_sessions (
                project_root TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    def _active_session_id(self) -> str | None:
        row = self._connection.execute(
            """
            SELECT thread_id
            FROM active_agent_sessions
            WHERE project_root = ?
            """,
            (self._project_root,),
        ).fetchone()
        return None if row is None else str(row[0])

    def _session_exists(self, session_id: str) -> bool:
        row = self._connection.execute(
            """
            SELECT 1
            FROM agent_sessions
            WHERE project_root = ? AND thread_id = ?
            """,
            (self._project_root, session_id),
        ).fetchone()
        return row is not None

    def _save_session(self, session_id: str) -> None:
        timestamp = self._now()
        self._connection.execute(
            """
            INSERT INTO agent_sessions (
                project_root, thread_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(project_root, thread_id)
            DO UPDATE SET updated_at = excluded.updated_at
            """,
            (self._project_root, session_id, timestamp, timestamp),
        )

    def _set_active_session(self, session_id: str) -> None:
        self._connection.execute(
            """
            INSERT INTO active_agent_sessions (project_root, thread_id)
            VALUES (?, ?)
            ON CONFLICT(project_root)
            DO UPDATE SET thread_id = excluded.thread_id
            """,
            (self._project_root, session_id),
        )

    @staticmethod
    def _normalize_session_id(session_id: str | None) -> str | None:
        if session_id is None:
            return None
        normalized = session_id.strip()
        if not normalized:
            raise ValueError("Session ID cannot be empty.")
        return normalized

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
