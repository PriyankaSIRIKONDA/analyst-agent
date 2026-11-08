"""Tests for the read-only MCP server, over the SDK's in-memory client (no subprocess)."""

import shutil
from pathlib import Path

import anyio
import pytest
from mcp import Client

from analyst_agent.mcp_server import build_server, main

ROOT = Path(__file__).resolve().parents[1]
DEMO_DB = ROOT / "data" / "demo.db"
DEMO_DOCS = ROOT / "data" / "demo_docs.md"


@pytest.fixture
def demo_copy(tmp_path) -> Path:
    """A private copy of the demo db, so 'unchanged after attacks' never depends on shared state."""
    path = tmp_path / "demo.db"
    shutil.copy(DEMO_DB, path)
    return path


def with_client(server, fn):
    """Run `fn(client)` (an async function) against the server over an in-memory connection."""

    async def runner():
        async with Client(server) as client:
            return await fn(client)

    return anyio.run(runner)


def call(server, name, arguments):
    return with_client(server, lambda client: client.call_tool(name, arguments))


def fingerprint(path: Path):
    stat = path.stat()
    return path.read_bytes(), stat.st_mtime_ns


def test_tool_list_with_docs(demo_copy):
    tools = with_client(build_server(demo_copy, DEMO_DOCS), lambda c: c.list_tools()).tools
    assert sorted(t.name for t in tools) == [
        "describe_table",
        "list_tables",
        "run_sql",
        "search_docs",
    ]


def test_tool_list_without_docs(demo_copy):
    tools = with_client(build_server(demo_copy), lambda c: c.list_tools()).tools
    assert sorted(t.name for t in tools) == ["describe_table", "list_tables", "run_sql"]


def test_every_tool_is_annotated_read_only(demo_copy):
    tools = with_client(build_server(demo_copy, DEMO_DOCS), lambda c: c.list_tools()).tools
    for tool in tools:
        a = tool.annotations
        assert a is not None, tool.name
        assert a.read_only_hint is True, tool.name
        assert a.destructive_hint is False, tool.name
        assert a.idempotent_hint is True, tool.name
        assert a.open_world_hint is False, tool.name
        assert tool.title and tool.description, tool.name


def test_server_name_and_instructions(demo_copy):
    async def info(client):
        return client.server_info, client.instructions

    server_info, instructions = with_client(build_server(demo_copy), info)
    assert server_info.name == "analyst-agent"
    assert "list_tables" in instructions


def test_list_tables_returns_demo_tables(demo_copy):
    result = call(build_server(demo_copy), "list_tables", {})
    assert not result.is_error
    assert result.structured_content["tables"] == [
        "customers",
        "order_items",
        "orders",
        "products",
    ]


def test_describe_table(demo_copy):
    result = call(build_server(demo_copy), "describe_table", {"table": "orders"})
    assert not result.is_error
    names = [c["name"] for c in result.structured_content["columns"]]
    assert "status" in names


def test_describe_unknown_table_is_error(demo_copy):
    result = call(build_server(demo_copy), "describe_table", {"table": "nope"})
    assert result.is_error
    assert "Unknown table" in result.structured_content["error"]
    assert "Unknown table" in result.content[0].text


def test_run_sql_select_returns_rows(demo_copy):
    result = call(build_server(demo_copy), "run_sql", {"sql": "SELECT COUNT(*) FROM orders"})
    assert not result.is_error
    assert result.structured_content["rows"] == [[2500]]
    assert result.structured_content["truncated"] is False


def test_run_sql_caps_output_at_50_rows(demo_copy):
    result = call(build_server(demo_copy), "run_sql", {"sql": "SELECT * FROM orders"})
    assert not result.is_error
    assert len(result.structured_content["rows"]) == 50
    assert result.structured_content["truncated"] is True


def test_run_sql_cuts_long_cells(demo_copy):
    sql = "SELECT replace(hex(zeroblob(400)), '0', 'a')"
    result = call(build_server(demo_copy), "run_sql", {"sql": sql})
    cell = result.structured_content["rows"][0][0]
    assert cell.endswith("...[truncated]")
    assert len(cell) == 300 + len("...[truncated]")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "DROP TABLE orders",
        "INSERT INTO customers (customer_id) VALUES (999999)",
        "UPDATE orders SET status = 'x'",
        "SELECT 1; DROP TABLE orders",
        "ATTACH DATABASE ':memory:' AS other",
        "WITH t AS (SELECT 1) DELETE FROM orders",
    ],
)
def test_write_attempts_are_errors_and_db_is_unchanged(demo_copy, sql):
    before = fingerprint(demo_copy)
    result = call(build_server(demo_copy), "run_sql", {"sql": sql})
    assert result.is_error
    assert result.structured_content["ok"] is False
    assert result.structured_content["error"]
    assert fingerprint(demo_copy) == before


def test_sql_error_is_error(demo_copy):
    result = call(build_server(demo_copy), "run_sql", {"sql": "SELECT * FROM missing_table"})
    assert result.is_error
    assert "SQL error" in result.structured_content["error"]


def test_search_docs_finds_revenue_definition(demo_copy):
    result = call(build_server(demo_copy, DEMO_DOCS), "search_docs", {"query": "revenue cancelled"})
    assert not result.is_error
    top = result.structured_content["results"][0]["text"]
    assert "Revenue means" in top
    assert "cancelled" in top


def test_search_docs_absent_without_docs(demo_copy):
    async def attempt(client):
        return await client.call_tool("search_docs", {"query": "revenue"})

    # An unknown tool is reported as an error result (or a protocol error), never a success.
    try:
        result = with_client(build_server(demo_copy), attempt)
    except Exception:  # noqa: BLE001 - either failure style is acceptable
        return
    assert result.is_error


def test_main_exits_with_message_when_db_missing(tmp_path, capsys):
    code = main(["--db", str(tmp_path / "absent.db"), "--no-docs"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "database not found" in captured.err
