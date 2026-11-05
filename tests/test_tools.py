"""Tests for the docs search and the toolbox."""

from analyst_agent.tools import DocsIndex, split_docs, tokenize


def test_tokenize_lowercases_splits_underscores_and_drops_plural_s():
    assert tokenize("Order_Items, orders!") == ["order", "item", "order"]


def test_split_docs_cuts_on_level_two_headings():
    text = "# Title\nintro\n\n## A\nalpha\n\n## B\nbeta\n"
    chunks = split_docs(text)
    assert len(chunks) == 3
    assert chunks[1].startswith("## A")


def test_search_ranks_the_relevant_chunk_first(toolbox):
    hits = toolbox.docs.search("what does revenue mean")
    assert hits
    assert "Revenue means" in hits[0]["text"]


def test_search_with_unknown_words_returns_nothing():
    index = DocsIndex(["## A\nalpha", "## B\nbeta"])
    assert index.search("zzzz") == []


def test_toolbox_runs_each_tool(toolbox):
    assert toolbox.call("list_tables", {})["tables"] == ["items"]
    assert toolbox.call("describe_table", {"table": "items"})["ok"]
    assert toolbox.call("search_docs", {"query": "price"})["results"]
    assert toolbox.call("run_sql", {"sql": "SELECT 1"})["rows"] == [[1]]


def test_toolbox_reports_bad_calls(toolbox):
    assert toolbox.call("run_sql", {})["ok"] is False  # missing argument
    assert toolbox.call("drop_everything", {})["ok"] is False  # unknown tool


def test_every_tool_schema_is_well_formed(toolbox):
    for schema in toolbox.schemas():
        function = schema["function"]
        assert schema["type"] == "function"
        assert function["name"] and function["description"]
        assert function["parameters"]["type"] == "object"
