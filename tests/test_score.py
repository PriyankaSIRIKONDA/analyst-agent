"""Tests for the execution-match scorer."""

import pytest

from score import execution_match, has_order_by, results_match


def test_order_is_ignored_when_gold_has_no_order_by():
    assert results_match([[1], [2], [3]], [[3], [1], [2]], ordered=False)


def test_order_matters_when_gold_has_order_by():
    assert not results_match([[1], [2]], [[2], [1]], ordered=True)
    assert results_match([[1], [2]], [[1], [2]], ordered=True)


def test_floats_are_compared_at_4_decimals():
    assert results_match([[3.14159]], [[3.1416]], ordered=False)
    assert not results_match([[3.14159]], [[3.1426]], ordered=False)


def test_int_and_float_with_same_value_match():
    assert results_match([[5]], [[5.0]], ordered=False)


def test_duplicates_count():
    assert not results_match([[1], [1], [2]], [[1], [2]], ordered=False)


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT a FROM t ORDER BY a", True),
        ("select a from t order  by a desc", True),
        ("SELECT a FROM t", False),
        ("SELECT 'order by' FROM t", False),
    ],
)
def test_has_order_by(sql, expected):
    assert has_order_by(sql) is expected


def test_execution_match_end_to_end(db):
    gold = "SELECT name FROM items WHERE id <= 3"
    reordered = "SELECT name FROM items WHERE id < 4 ORDER BY id DESC"
    assert execution_match(db, gold, reordered)["correct"]
    assert not execution_match(db, gold, "SELECT name FROM items WHERE id <= 2")["correct"]
    broken = execution_match(db, gold, "SELECT nope FROM items")
    assert broken["reason"].startswith("pred_sql_error")
    assert execution_match(db, gold, None)["reason"] == "no_sql"


def test_broken_gold_sql_raises(db):
    with pytest.raises(ValueError):
        execution_match(db, "SELECT nope FROM items", "SELECT 1")
