"""Check that the demo database builds and that every gold question has working SQL."""

import json
from pathlib import Path

import pytest

from analyst_agent.db import Database
from make_demo_db import build

GOLD_PATH = Path(__file__).resolve().parents[1] / "eval" / "demo_gold.jsonl"


def load_gold() -> list[dict]:
    lines = GOLD_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory) -> Database:
    path = tmp_path_factory.mktemp("demo") / "demo.db"
    build(path)
    return Database(path)


def test_gold_file_has_20_unique_questions():
    gold = load_gold()
    assert len(gold) == 20
    assert len({g["id"] for g in gold}) == 20
    assert all({"id", "question", "gold_sql", "difficulty"} <= set(g) for g in gold)


@pytest.mark.parametrize("item", load_gold(), ids=lambda g: g["id"])
def test_every_gold_sql_runs_and_returns_rows(demo_db, item):
    result = demo_db.run_sql(item["gold_sql"])
    assert result["ok"], result.get("error")
    assert result["rows"], "gold SQL returned an empty result"


def test_demo_db_is_reproducible(tmp_path):
    build(tmp_path / "a.db")
    build(tmp_path / "b.db")
    sql = "SELECT SUM(quantity * unit_price) FROM order_items"
    assert Database(tmp_path / "a.db").run_sql(sql) == Database(tmp_path / "b.db").run_sql(sql)
