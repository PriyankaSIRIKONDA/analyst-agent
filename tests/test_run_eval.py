"""Tests for run_eval: oracle mode, the BIRD metric, evidence hints and per-row databases."""

import json
import sqlite3

import pytest

import run_eval
from run_eval import (
    Resources,
    StoppedEarly,
    build_config,
    load_gold,
    oracle_llm,
    parse_args,
    question_text,
    run_all,
)
from score import execution_match


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_resume_keeps_finished_rows_and_redoes_model_errors(tmp_path):
    out = tmp_path / "run.jsonl"
    run_all(parse_args(["--oracle", "--out", str(out), "--limit", "2"]))
    rows = _read_rows(out)
    rows[1]["status"], rows[1]["correct"] = "llm_error", False  # as if a quota stopped it
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    run_all(parse_args(["--oracle", "--out", str(out), "--limit", "3", "--resume"]))
    rows = _read_rows(out)
    assert [r["id"] for r in rows] == ["d01", "d02", "d03"]
    assert all(r["status"] == "ok" for r in rows)


def test_resume_refuses_a_file_with_other_settings(tmp_path):
    out = tmp_path / "run.jsonl"
    run_all(parse_args(["--oracle", "--out", str(out), "--limit", "1"]))
    with pytest.raises(SystemExit):
        run_all(parse_args(["--oracle", "--out", str(out), "--resume", "--metric", "bird"]))


def test_run_stops_after_model_errors_in_a_row(tmp_path, monkeypatch):
    class BrokenClient:
        def __init__(self, model=None, min_interval_s=0.0):
            self.model = "broken"

        def complete(self, messages, tools):
            raise RuntimeError("quota exceeded")

    monkeypatch.setattr(run_eval, "LiteLLMClient", BrokenClient)
    out = tmp_path / "run.jsonl"
    with pytest.raises(StoppedEarly):
        run_all(parse_args(["--out", str(out), "--stop-after-errors", "2"]))
    assert len(_read_rows(out)) == 2


def test_oracle_scores_every_demo_question_correct(tmp_path):
    out = tmp_path / "oracle.jsonl"
    args = parse_args(["--oracle", "--out", str(out)])
    assert run_all(args) == out
    rows = load_gold(out)
    assert len(rows) == 20
    assert all(r["correct"] for r in rows)
    assert {r["config"]["model"] for r in rows} == {"oracle"}
    assert set(rows[0]["config"]) >= {
        "use_docs", "use_repair", "use_evidence", "metric", "max_steps",
        "sql_timeout", "gold", "git_commit",
    }


def test_bird_metric_ignores_order_and_duplicates_but_strict_does_not(db):
    gold = "SELECT id FROM items WHERE id <= 3 ORDER BY id"
    pred = "SELECT id FROM items WHERE id <= 3 UNION ALL SELECT id FROM items WHERE id = 1"
    assert execution_match(db, gold, pred, metric="bird")["correct"]
    assert not execution_match(db, gold, pred)["correct"]
    reordered = "SELECT id FROM items WHERE id <= 3 ORDER BY id DESC"
    assert execution_match(db, gold, reordered, metric="bird")["correct"]
    assert not execution_match(db, gold, reordered, metric="strict")["correct"]


def test_bird_metric_does_not_round_floats(db):
    gold = "SELECT 3.14159"
    assert execution_match(db, gold, "SELECT 3.1416")["correct"]
    assert not execution_match(db, gold, "SELECT 3.1416", metric="bird")["correct"]


def test_evidence_is_added_as_a_hint():
    item = {"question": "How many?", "evidence": "big means > 5"}
    assert question_text(item) == "How many?\n\nHint: big means > 5"
    assert question_text(item, use_evidence=False) == "How many?"
    assert question_text({"question": "How many?", "evidence": ""}) == "How many?"
    assert question_text({"question": "How many?"}) == "How many?"


def test_rows_use_their_own_database_and_docs(tmp_path):
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.sqlite"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE t (x)")
        conn.commit()
        conn.close()
        docs = tmp_path / f"{name}.md"
        docs.write_text(f"# Database {name}\n\n## Table t\n- `x`: column {name}", encoding="utf-8")
        paths.append((path, docs))
    default_db, default_docs = paths[0]
    resources = Resources(default_db, default_docs)

    other = {"db": str(paths[1][0]), "docs": str(paths[1][1])}
    assert resources.database({}).path == default_db.resolve()
    assert resources.database(other).path == paths[1][0].resolve()
    assert resources.database(other) is resources.database(other)  # cached
    assert "column b" in resources.docs(other).chunks[1]
    assert "column a" in resources.docs({}).chunks[1]
    assert Resources(default_db, default_docs, use_docs=False).docs({}) is None


def test_oracle_run_on_a_row_with_its_own_db(tmp_path):
    path = tmp_path / "x.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x)")
    conn.execute("INSERT INTO t VALUES (7)")
    conn.commit()
    conn.close()
    docs = tmp_path / "x.md"
    docs.write_text("# Database x\n\n## Table t\n- `x`: value", encoding="utf-8")
    gold = tmp_path / "gold.jsonl"
    row = {"id": "g1", "question": "q", "evidence": "e", "gold_sql": "SELECT x FROM t",
           "db": str(path), "docs": str(docs), "difficulty": "simple", "db_id": "x"}
    gold.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    args = parse_args(["--oracle", "--gold", str(gold), "--out", str(out), "--metric", "bird"])
    run_all(args)
    result = load_gold(out)[0]
    assert result["correct"] and result["difficulty"] == "simple"
    assert result["config"]["metric"] == "bird"


def test_oracle_llm_has_two_turns():
    llm = oracle_llm("SELECT 1")
    assert len(llm.script) == 2
    assert build_config(parse_args(["--no-evidence", "--no-docs"]), "m")["use_evidence"] is False
