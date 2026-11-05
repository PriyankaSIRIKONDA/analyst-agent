"""Guardrail tests: the database layer must refuse anything that is not a plain read."""

import sqlite3

import pytest

from analyst_agent.db import Database


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE items",
        "INSERT INTO items VALUES (99, 'x', 1.0)",
        "UPDATE items SET price = 0",
        "DELETE FROM items",
        "CREATE TABLE evil (a INTEGER)",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA writable_schema = 1",
        "PRAGMA table_info(items)",
        "SELECT 1; DROP TABLE items",
        "-- harmless comment\nDROP TABLE items",
        "/* comment */ DELETE FROM items",
        "",
        "   ;  ",
    ],
)
def test_non_select_statements_are_rejected(db, sql):
    result = db.run_sql(sql)
    assert result["ok"] is False
    assert "error" in result
    # and the data is untouched
    assert db.run_sql("SELECT COUNT(*) FROM items")["rows"] == [[20]]


def test_write_hidden_behind_with_is_stopped_by_the_database_layers(db):
    """WITH passes the text check, so the authorizer / read-only mode must stop it."""
    sql = "WITH x AS (SELECT 99 AS a) INSERT INTO items (id, name, price) SELECT a, 'z', 1 FROM x"
    result = db.run_sql(sql)
    assert result["ok"] is False
    assert db.run_sql("SELECT COUNT(*) FROM items")["rows"] == [[20]]


def test_simple_select_works(db):
    result = db.run_sql("SELECT id, name FROM items WHERE id <= 2 ORDER BY id;")
    assert result["ok"] is True
    assert result["columns"] == ["id", "name"]
    assert result["rows"] == [[1, "item1"], [2, "item2"]]
    assert result["truncated"] is False


def test_with_query_and_recursive_cte_work(db):
    sql = (
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) "
        "SELECT n FROM r"
    )
    result = db.run_sql(sql)
    assert result["ok"] is True
    assert [row[0] for row in result["rows"]] == [1, 2, 3, 4, 5]


def test_semicolon_inside_a_string_is_not_a_second_statement(db):
    result = db.run_sql("SELECT 'a;b'")
    assert result["ok"] is True
    assert result["rows"] == [["a;b"]]


def test_row_limit_is_applied(db):
    result = db.run_sql("SELECT id FROM items", limit=5)
    assert result["ok"] is True
    assert len(result["rows"]) == 5
    assert result["truncated"] is True


def test_default_row_limit_is_200(tmp_path):
    path = tmp_path / "big.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (n INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(500)])
    conn.commit()
    conn.close()
    result = Database(path).run_sql("SELECT n FROM t")
    assert len(result["rows"]) == 200
    assert result["truncated"] is True


def test_slow_query_times_out(db):
    sql = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r"
    result = db.run_sql(sql, timeout_s=0.2)
    assert result["ok"] is False
    assert "timed out" in result["error"]


def test_sql_errors_come_back_as_messages_not_exceptions(db):
    result = db.run_sql("SELECT nope FROM items")
    assert result["ok"] is False
    assert "nope" in result["error"]


def test_list_and_describe_table(db):
    assert db.list_tables() == ["items"]
    info = db.describe_table("items")
    assert info["ok"] is True
    assert [c["name"] for c in info["columns"]] == ["id", "name", "price"]
    assert len(info["sample_rows"]) == 3
    assert db.describe_table("missing")["ok"] is False
    assert db.describe_table('items"; DROP TABLE items; --')["ok"] is False
