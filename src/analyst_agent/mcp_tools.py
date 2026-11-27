"""The graph's database tool: a client of this repository's own read-only MCP server.

The graph never opens the database itself. Every schema lookup and every query goes through
MCP to ``mcp_server.py``, so the guardrails of db.py and the output bounds of tools.py stay the
only path to the data, exactly as for any other MCP client (a desktop assistant, an IDE, ...).

Three ways to reach the server:

* ``MCPTools.stdio(db, docs)``  launches ``python -m analyst_agent.mcp_server`` as a subprocess
  (what the API does);
* ``MCPTools.http(url)``        connects to a server already running with ``--transport http``;
* ``MCPTools.in_process(db, docs)`` builds the server in this process (tests and evals: same
  protocol, no subprocess).
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from analyst_agent.db import DEFAULT_TIMEOUT_S


class MCPToolError(RuntimeError):
    """The MCP call itself failed (transport or protocol), not the SQL. Worth retrying."""


class MCPTools:
    """Async context manager with one method that matters: ``await call(name, arguments)``."""

    def __init__(self, server, *, read_timeout_s: float = 30.0):
        self._server = server
        self._read_timeout_s = read_timeout_s
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self.tool_names: set[str] = set()

    # ------------------------------------------------------------------ constructors
    @classmethod
    def stdio(
        cls,
        db_path: str | Path,
        docs_path: str | Path | None = None,
        sql_timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> MCPTools:
        args = ["-m", "analyst_agent.mcp_server", "--db", str(db_path)]
        args += ["--docs", str(docs_path)] if docs_path is not None else ["--no-docs"]
        args += ["--sql-timeout", str(sql_timeout_s)]
        # Only what the server needs: it must not inherit API keys of the parent process.
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "PYTHONPATH", "HOME", "LANG")}
        return cls(StdioServerParameters(command=sys.executable, args=args, env=env))

    @classmethod
    def http(cls, url: str) -> MCPTools:
        return cls(url)

    @classmethod
    def in_process(
        cls,
        db_path: str | Path,
        docs_path: str | Path | None = None,
        sql_timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> MCPTools:
        from analyst_agent.mcp_server import build_server

        return cls(build_server(db_path, docs_path, sql_timeout_s=sql_timeout_s))

    # ------------------------------------------------------------------ lifecycle
    async def __aenter__(self) -> MCPTools:
        self._stack = AsyncExitStack()
        try:
            self._client = await self._stack.enter_async_context(
                Client(self._server, read_timeout_seconds=self._read_timeout_s)
            )
            listed = await self._client.list_tools()
            self.tool_names = {tool.name for tool in listed.tools}
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = self._client = None

    @property
    def has_docs(self) -> bool:
        return "search_docs" in self.tool_names

    # ------------------------------------------------------------------ calls
    async def call(self, name: str, arguments: dict | None = None) -> dict:
        """Call one tool and return the server's dict: ``{"ok": True, ...}`` or ``{"ok": False,
        "error": ...}``. A rejected or failing query is a normal result, not an exception."""
        if self._client is None:
            raise MCPToolError("MCPTools is not open; use `async with`.")
        if name not in self.tool_names:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        try:
            result = await self._client.call_tool(name, arguments or {})
        except Exception as exc:  # transport or protocol failure
            raise MCPToolError(f"MCP call {name} failed: {type(exc).__name__}: {exc}") from exc
        payload = getattr(result, "structured_content", None)
        if not isinstance(payload, dict):
            texts = [c.text for c in getattr(result, "content", []) if hasattr(c, "text")]
            try:
                payload = json.loads(texts[0]) if texts else {}
            except json.JSONDecodeError:
                payload = {"ok": False, "error": texts[0][:300]}
        payload.setdefault("ok", not getattr(result, "is_error", False))
        return payload
