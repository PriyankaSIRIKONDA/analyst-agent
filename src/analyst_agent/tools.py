"""The four tools the agent can call, and their JSON schemas.

Tools: list_tables, describe_table, search_docs, run_sql.
Each tool returns a plain dict that can be serialised to JSON and shown to the model.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from analyst_agent.db import DEFAULT_TIMEOUT_S, Database

MODEL_ROW_LIMIT = 50  # rows of a query result that are shown to the model
MAX_CELL_CHARS = 300  # longer cell values are cut before they are shown to the model

# --------------------------------------------------------------------- docs search


def tokenize(text: str) -> list[str]:
    """Lowercase words; underscores split words; a trailing 's' is dropped ('orders' -> 'order')."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words]


def split_docs(markdown: str) -> list[str]:
    """Cut a markdown file into chunks, one per '## ' section (the intro is its own chunk)."""
    chunks = re.split(r"\n(?=## )", markdown)
    return [c.strip() for c in chunks if c.strip()]


class DocsIndex:
    """Keyword search over doc chunks, scored with BM25 (no embeddings in v0)."""

    def __init__(self, chunks: list[str], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(c)) for c in chunks]
        self.lengths = [sum(tc.values()) for tc in self.term_counts]
        self.avg_length = sum(self.lengths) / max(len(chunks), 1)
        # document frequency: in how many chunks does each word appear?
        self.doc_freq: Counter = Counter()
        for tc in self.term_counts:
            self.doc_freq.update(tc.keys())

    @classmethod
    def from_file(cls, path: str | Path) -> DocsIndex:
        return cls(split_docs(Path(path).read_text(encoding="utf-8")))

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_terms: list[str], index: int) -> float:
        """BM25 score of one chunk for the query."""
        counts = self.term_counts[index]
        norm = 1 - self.b + self.b * self.lengths[index] / max(self.avg_length, 1e-9)
        total = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf:
                total += self._idf(term) * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return total

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Best chunks first; chunks with score 0 are left out."""
        terms = tokenize(query)
        scored = [(self.score(terms, i), i) for i in range(len(self.chunks))]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"score": round(s, 3), "text": self.chunks[i]} for s, i in scored[:top_k] if s > 0
        ]


# ------------------------------------------------------------------ tool schemas


def function_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Build one tool definition in the OpenAI function-calling format (LiteLLM uses it)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


LIST_TABLES_SCHEMA = function_schema(
    "list_tables", "List the names of all tables in the database.", {}, []
)
DESCRIBE_TABLE_SCHEMA = function_schema(
    "describe_table",
    "Show the columns, types, keys and 3 sample rows of one table.",
    {"table": {"type": "string", "description": "Exact table name."}},
    ["table"],
)
SEARCH_DOCS_SCHEMA = function_schema(
    "search_docs",
    "Search the written documentation for the tables (column meanings, business definitions "
    "such as what 'revenue' means). Use it before writing SQL.",
    {"query": {"type": "string", "description": "Keywords to look for."}},
    ["query"],
)
RUN_SQL_SCHEMA = function_schema(
    "run_sql",
    f"Run one read-only SQLite SELECT query. At most {MODEL_ROW_LIMIT} rows are returned. "
    "If it fails, the error message is returned so you can fix the query.",
    {"sql": {"type": "string", "description": "A single SELECT or WITH query."}},
    ["sql"],
)


# ------------------------------------------------------------------------ toolbox


def shorten_cell(value):
    """Bound one cell for the model: bytes become '<N bytes>', long text is cut at 300 chars."""
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    text = str(value)
    if len(text) > MAX_CELL_CHARS:
        return text[:MAX_CELL_CHARS] + "...[truncated]"
    return value


def shorten_rows(rows: list[list]) -> list[list]:
    """Apply shorten_cell to every cell of a list of rows."""
    return [[shorten_cell(v) for v in row] for row in rows]


class ToolBox:
    """Holds the database (and docs) and runs tools by name."""

    def __init__(
        self,
        db: Database,
        docs: DocsIndex | None = None,
        model_row_limit: int = MODEL_ROW_LIMIT,
        sql_timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self.db = db
        self.model_row_limit = model_row_limit
        self.sql_timeout_s = sql_timeout_s
        self.docs = docs  # None means search_docs is disabled (the --no-docs ablation)

    def schemas(self) -> list[dict]:
        """Tool definitions to send to the model."""
        schemas = [LIST_TABLES_SCHEMA, DESCRIBE_TABLE_SCHEMA]
        if self.docs is not None:
            schemas.append(SEARCH_DOCS_SCHEMA)
        schemas.append(RUN_SQL_SCHEMA)
        return schemas

    def call(self, name: str, arguments: dict) -> dict:
        """Run the tool called `name`. Unknown tools or bad arguments give an error dict."""
        try:
            if name == "list_tables":
                return {"ok": True, "tables": self.db.list_tables()}
            if name == "describe_table":
                return self._describe_table(arguments["table"])
            if name == "search_docs" and self.docs is not None:
                return {"ok": True, "results": self.docs.search(arguments["query"])}
            if name == "run_sql":
                return self._run_sql(arguments["sql"])
        except KeyError as exc:
            return {"ok": False, "error": f"Missing argument: {exc}"}
        return {"ok": False, "error": f"Unknown tool: {name}"}

    def _run_sql(self, sql: str) -> dict:
        """Run the query with the model-facing row limit and cell truncation."""
        result = self.db.run_sql(sql, limit=self.model_row_limit, timeout_s=self.sql_timeout_s)
        if result["ok"]:
            result["rows"] = shorten_rows(result["rows"])
        return result

    def _describe_table(self, table: str) -> dict:
        """Describe a table; sample rows get the same cell truncation."""
        result = self.db.describe_table(table)
        if result["ok"]:
            result["sample_rows"] = shorten_rows(result["sample_rows"])
        return result
