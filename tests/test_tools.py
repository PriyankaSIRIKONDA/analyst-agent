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


def _long_text_toolbox(tmp_path):
    import sqlite3

    from analyst_agent.db import Database
    from analyst_agent.tools import ToolBox

    path = tmp_path / "long.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER, body TEXT, blob BLOB)")
    rows = [(i, "x" * 1000, bytes(7)) for i in range(80)]
    conn.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return ToolBox(Database(path))


def test_run_sql_shows_the_model_at_most_50_rows(tmp_path):
    result = _long_text_toolbox(tmp_path).call("run_sql", {"sql": "SELECT id FROM t"})
    assert len(result["rows"]) == 50
    assert result["truncated"] is True


def test_long_cells_are_cut_and_bytes_are_summarised(tmp_path):
    result = _long_text_toolbox(tmp_path).call("run_sql", {"sql": "SELECT body, blob FROM t"})
    body, blob = result["rows"][0]
    assert body == "x" * 300 + "...[truncated]"
    assert blob == "<7 bytes>"


def test_describe_table_sample_rows_are_cut(tmp_path):
    result = _long_text_toolbox(tmp_path).call("describe_table", {"table": "t"})
    assert result["sample_rows"][0][1].endswith("...[truncated]")
    assert result["sample_rows"][0][2] == "<7 bytes>"


def test_toolbox_limits_are_configurable(db):
    from analyst_agent.tools import ToolBox

    box = ToolBox(db, model_row_limit=5, sql_timeout_s=0.5)
    assert len(box.call("run_sql", {"sql": "SELECT id FROM items"})["rows"]) == 5
    slow = (
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) "
        "SELECT count(*) FROM r"
    )
    assert "timed out" in box.call("run_sql", {"sql": slow})["error"]


def test_run_sql_schema_mentions_50_rows(toolbox):
    run_sql = [s for s in toolbox.schemas() if s["function"]["name"] == "run_sql"][0]
    assert "50 rows" in run_sql["function"]["description"]
