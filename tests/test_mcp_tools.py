"""The graph's MCP client, against the real MCP server (in process and as a subprocess)."""

import asyncio

import pytest

from analyst_agent.mcp_tools import MCPToolError, MCPTools


def run(coro):
    return asyncio.run(coro)


async def calls(tools: MCPTools):
    async with tools:
        return (
            sorted(tools.tool_names),
            await tools.call("list_tables"),
            await tools.call("run_sql", {"sql": "SELECT COUNT(*) AS n FROM items"}),
            await tools.call("run_sql", {"sql": "DELETE FROM items"}),
            await tools.call("drop_everything", {}),
        )


@pytest.mark.parametrize("transport", ["in_process", "stdio"])
def test_tools_go_through_the_mcp_server(db, docs_file, transport):
    tools = getattr(MCPTools, transport)(db.path, docs_file)
    names, listed, count, delete, unknown = run(calls(tools))
    assert names == ["describe_table", "list_tables", "run_sql", "search_docs"]
    assert listed == {"ok": True, "tables": ["items"]}
    assert count["ok"] and count["rows"] == [[20]]
    assert delete["ok"] is False and "Only SELECT" in delete["error"]
    assert unknown == {"ok": False, "error": "Unknown tool: drop_everything"}
    assert db.run_sql("SELECT COUNT(*) FROM items")["rows"] == [[20]]  # nothing was deleted


def test_without_docs_there_is_no_search_docs(db):
    async def go():
        async with MCPTools.in_process(db.path, None) as tools:
            return tools.has_docs

    assert run(go()) is False


def test_call_before_open_is_an_error(db):
    with pytest.raises(MCPToolError):
        run(MCPTools.in_process(db.path).call("list_tables"))
