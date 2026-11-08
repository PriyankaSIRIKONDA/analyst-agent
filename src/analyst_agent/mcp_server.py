"""Read-only MCP server that exposes the agent's own tools to any MCP client.

It wraps a single ToolBox, so every guardrail of db.py (SELECT/WITH only, mode=ro, SQLite
authorizer, row limit, timeout) and every output bound of tools.py (50 rows, 300-char cells)
applies exactly as it does for the agent. The server adds no SQL path of its own, does not
expose final_answer and does not run the LLM agent.

Run with:  python -m analyst_agent.mcp_server            (stdio, for desktop clients)
           python -m analyst_agent.mcp_server --transport http --port 8765

Stdout carries the stdio protocol, so nothing in this module prints to it; messages go to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from analyst_agent.db import DEFAULT_TIMEOUT_S, Database
from analyst_agent.tools import MAX_CELL_CHARS, MODEL_ROW_LIMIT, DocsIndex, ToolBox

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "demo.db"
DEFAULT_DOCS = ROOT / "data" / "demo_docs.md"

SERVER_NAME = "analyst-agent"
INSTRUCTIONS = (
    "Read-only access to one SQLite database. Call list_tables, then describe_table for the "
    "tables you need. Use search_docs (when available) for definitions of business terms, "
    "then write one SELECT query and call run_sql."
)

# Every tool only reads from a local file: safe to retry, never changes anything.
READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)


def to_result(payload: dict) -> CallToolResult:
    """Turn a ToolBox dict into an MCP result; ok=False becomes an isError result.

    The dict is sent twice: as structured content (for programs) and as JSON text (for
    clients that only show text). On failure the dict still carries the ``error`` message.
    """
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=json.loads(text),
        is_error=not payload.get("ok", False),
    )


def build_server(
    db_path: str | Path,
    docs_path: str | Path | None = None,
    sql_timeout_s: float = DEFAULT_TIMEOUT_S,
) -> MCPServer:
    """Create the MCP server over one database (and, optionally, one markdown docs file)."""
    docs = DocsIndex.from_file(docs_path) if docs_path is not None else None
    toolbox = ToolBox(Database(db_path), docs, sql_timeout_s=sql_timeout_s)
    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)

    @server.tool(
        name="list_tables",
        title="List tables",
        description=(
            "List the names of all tables in the database. Call this first to see what data "
            "exists. Returns {ok, tables: [name, ...]}."
        ),
        annotations=READ_ONLY,
    )
    def list_tables() -> CallToolResult:
        return to_result(toolbox.call("list_tables", {}))

    @server.tool(
        name="describe_table",
        title="Describe a table",
        description=(
            "Show the columns (name, type, primary key), foreign keys and 3 sample rows of one "
            f"table. Cells longer than {MAX_CELL_CHARS} characters are cut. Use the exact name "
            "from list_tables; an unknown name returns an error."
        ),
        annotations=READ_ONLY,
    )
    def describe_table(table: str) -> CallToolResult:
        return to_result(toolbox.call("describe_table", {"table": table}))

    @server.tool(
        name="run_sql",
        title="Run a read-only SQL query",
        description=(
            "Run ONE read-only SQLite query (SELECT or WITH only; no INSERT, UPDATE, DELETE, "
            "DROP, ATTACH, PRAGMA or several statements). Returns {ok, columns, rows, "
            f"truncated}}: at most {MODEL_ROW_LIMIT} rows are shown (truncated=true means more "
            f"exist, so aggregate or add LIMIT), cells are cut at {MAX_CELL_CHARS} characters, "
            f"and the query is stopped after {sql_timeout_s:g} seconds. A rejected or failing "
            "query returns an error message so you can fix the SQL. Check column names with "
            "describe_table first."
        ),
        annotations=READ_ONLY,
    )
    def run_sql(sql: str) -> CallToolResult:
        return to_result(toolbox.call("run_sql", {"sql": sql}))

    if docs is not None:

        @server.tool(
            name="search_docs",
            title="Search the table documentation",
            description=(
                "Keyword search (BM25) over the written documentation of the tables: column "
                "meanings and business definitions such as what 'revenue' or 'active customer' "
                "means. Returns {ok, results: [{score, text}, ...]} with the 3 best sections. "
                "Use it BEFORE writing SQL whenever the question contains a business term."
            ),
            annotations=READ_ONLY,
        )
        def search_docs(query: str) -> CallToolResult:
            return to_result(toolbox.call("search_docs", {"query": query}))

    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyst-agent-mcp", description="Read-only MCP server for the analyst agent's tools."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite file (default: demo)")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS, help="markdown docs file")
    parser.add_argument("--no-docs", action="store_true", help="do not expose search_docs")
    parser.add_argument("--sql-timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="http only")
    parser.add_argument("--port", type=int, default=8765, help="http only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.db.is_file():
        print(f"analyst-agent-mcp: database not found: {args.db}", file=sys.stderr)
        return 2
    docs_path = None if args.no_docs else args.docs
    if docs_path is not None and not docs_path.is_file():
        print(
            f"analyst-agent-mcp: docs file not found: {docs_path} (use --no-docs to skip)",
            file=sys.stderr,
        )
        return 2

    server = build_server(args.db, docs_path, sql_timeout_s=args.sql_timeout)
    if args.transport == "http":
        print(
            f"analyst-agent-mcp: serving on http://{args.host}:{args.port}/mcp", file=sys.stderr
        )
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
