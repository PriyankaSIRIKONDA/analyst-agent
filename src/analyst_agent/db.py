"""Read-only SQLite access with guardrails.

An LLM writes the SQL that runs here, so the database layer must be safe on its own.
Four layers protect the data:

1. Text check: the statement must be a single SELECT or WITH query.
2. The connection is opened read-only (SQLite URI ``mode=ro``).
3. A SQLite authorizer denies every operation except reading.
4. A row limit and a time limit stop runaway queries.

``run_sql`` never raises for bad SQL. It returns a dict with ``ok`` set to False and an
``error`` message, so the agent can show the error to the model and let it repair the query.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

DEFAULT_ROW_LIMIT = 200
DEFAULT_TIMEOUT_S = 5.0

# Authorizer action codes that a read-only query is allowed to trigger.
# Some Python versions do not export SQLITE_RECURSIVE, so we fall back to its value (33).
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")


class SqlRejected(ValueError):
    """Raised when a statement is not a single read-only SELECT/WITH query."""


def check_read_only_sql(sql: str) -> str:
    """Return the cleaned statement, or raise SqlRejected.

    Accepted: one statement that starts with SELECT or WITH (a trailing ';' is fine).
    """
    cleaned = _COMMENT_RE.sub(" ", sql).strip().rstrip(";").strip()
    if not cleaned:
        raise SqlRejected("Empty SQL statement.")

    first_word = cleaned.split(None, 1)[0].upper()
    if first_word not in ("SELECT", "WITH"):
        raise SqlRejected(f"Only SELECT or WITH queries are allowed, got {first_word}.")

    # Remove quoted text first, so a ';' inside a string literal does not count.
    without_strings = _STRING_RE.sub("''", cleaned)
    if ";" in without_strings:
        raise SqlRejected("Only one SQL statement is allowed.")
    return cleaned


def _deny_writes(action: int, *_args) -> int:
    """SQLite authorizer callback: allow reads, deny everything else."""
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


class Database:
    """A SQLite database that can only be read."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Database not found: {self.path}")

    def _connect(self) -> sqlite3.Connection:
        """Open a new read-only connection."""
        uri = f"{self.path.as_uri()}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    # ------------------------------------------------------------------ run_sql
    def run_sql(
        self,
        sql: str,
        limit: int = DEFAULT_ROW_LIMIT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict:
        """Run one read-only query.

        Success: {"ok": True, "columns": [...], "rows": [[...], ...], "truncated": bool}
        Failure: {"ok": False, "error": "message"}
        """
        try:
            cleaned = check_read_only_sql(sql)
        except SqlRejected as exc:
            return {"ok": False, "error": str(exc)}

        conn = self._connect()
        try:
            conn.set_authorizer(_deny_writes)

            # The progress handler is called every 1000 SQLite VM steps.
            # Returning a non-zero value aborts the query.
            deadline = time.monotonic() + timeout_s
            conn.set_progress_handler(lambda: time.monotonic() > deadline, 1000)

            cursor = conn.execute(cleaned)
            columns = [d[0] for d in cursor.description or []]
            fetched = cursor.fetchmany(limit + 1)  # one extra row tells us if we truncated
            truncated = len(fetched) > limit
            rows = [list(row) for row in fetched[:limit]]
            return {"ok": True, "columns": columns, "rows": rows, "truncated": truncated}
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc):
                return {"ok": False, "error": f"Query timed out after {timeout_s} seconds."}
            return {"ok": False, "error": f"SQL error: {exc}"}
        except sqlite3.DatabaseError as exc:  # includes authorizer denials
            return {"ok": False, "error": f"SQL error: {exc}"}
        except (sqlite3.Warning, sqlite3.ProgrammingError) as exc:
            return {"ok": False, "error": f"SQL error: {exc}"}
        finally:
            conn.close()

    # ------------------------------------------------- schema helpers (trusted code)
    def list_tables(self) -> list[str]:
        """Names of all user tables, sorted."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]

    def describe_table(self, table: str) -> dict:
        """Columns, primary key, foreign keys and 3 sample rows of one table.

        Returns {"ok": False, "error": ...} if the table does not exist.
        """
        if table not in self.list_tables():  # also blocks injection through the name
            return {"ok": False, "error": f"Unknown table: {table!r}"}

        quoted = '"' + table.replace('"', '""') + '"'
        conn = self._connect()
        try:
            info = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
            fks = conn.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
            sample_cursor = conn.execute(f"SELECT * FROM {quoted} LIMIT 3")
            sample_rows = [list(r) for r in sample_cursor.fetchall()]
        finally:
            conn.close()

        return {
            "ok": True,
            "table": table,
            "columns": [
                {"name": c[1], "type": c[2], "primary_key": bool(c[5])} for c in info
            ],
            "foreign_keys": [
                {"column": fk[3], "references": f"{fk[2]}({fk[4]})"} for fk in fks
            ],
            "sample_rows": sample_rows,
        }
