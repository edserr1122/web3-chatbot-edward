"""
SQLite-backed chat history store.
Keeps conversation history per user/session with optional metadata summaries.
"""

import json
import os
import sqlite3
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional

from src.utils import config

logger = logging.getLogger(__name__)


def _current_timestamp() -> int:
    """Return current UTC timestamp as integer seconds."""
    return int(datetime.utcnow().timestamp())


class HistoryStore:
    """Persist chat history (sessions + messages) in SQLite."""

    def __init__(self):
        self.db_path = config.HISTORY_DB_PATH
        self._lock = threading.Lock()
        self._ensure_database()

    def _ensure_database(self):
        """Create data directory and tables if they do not exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    started_at INTEGER NOT NULL,
                    last_active_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp INTEGER NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_time
                ON messages(session_id, timestamp)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_user_time
                ON messages(user_id, timestamp)
                """
            )

    def _execute(self, query: str, params: tuple = (), fetch: bool = False, fetchall: bool = False):
        """Execute a query with locking and optional fetch."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            if fetch:
                return cursor.fetchone()
            if fetchall:
                return cursor.fetchall()
            return None

    def create_session(self, session_id: str, user_id: str):
        """Create session row if it does not exist and touch last_active timestamp."""
        now_ts = _current_timestamp()
        self._execute(
            """
            INSERT OR IGNORE INTO sessions (id, user_id, started_at, last_active_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, user_id, now_ts, now_ts),
        )
        self.touch_session(session_id)

    def touch_session(self, session_id: str):
        """Update session last_active timestamp."""
        now_ts = _current_timestamp()
        self._execute(
            "UPDATE sessions SET last_active_at = ? WHERE id = ?",
            (now_ts, session_id),
        )

    def append_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Insert chat message with optional metadata."""
        now_ts = _current_timestamp()
        meta_serialized = json.dumps(metadata or {})
        self.create_session(session_id, user_id)
        self._execute(
            """
            INSERT INTO messages (session_id, user_id, role, content, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_id, role, content, meta_serialized, now_ts),
        )
        self.touch_session(session_id)

    def get_session_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Return messages for a session ordered by timestamp."""
        sql = """
            SELECT session_id, user_id, role, content, metadata, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """
        params = [session_id]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._execute(sql, tuple(params), fetchall=True) or []
        return [self._row_to_dict(row) for row in rows]

    def get_messages_since(
        self,
        user_id: str,
        since_ts: int,
        limit: int = 50,
    ) -> List[Dict]:
        """Return messages for a user since timestamp."""
        rows = self._execute(
            """
            SELECT session_id, user_id, role, content, metadata, timestamp
            FROM messages
            WHERE user_id = ? AND timestamp >= ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (user_id, since_ts, limit),
            fetchall=True,
        ) or []
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        """Convert SQLite row to dict with parsed metadata."""
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                metadata = {}
        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "role": row["role"],
            "content": row["content"],
            "metadata": metadata,
            "timestamp": row["timestamp"],
        }


history_store = HistoryStore()


