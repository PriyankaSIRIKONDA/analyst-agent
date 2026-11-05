"""Tests for the BIRD loader, using a tiny fake BIRD tree."""

import json
import sqlite3

import pytest

from analyst_agent.tools import split_docs
from load_bird import GuardrailRejected, build_dataset, main


def make_tree(tmp_path, records):
    """mini_dev_data/ with one json file and database `shop` nested like the real download."""
    root = tmp_path / "bird_data" / "extracted" / "mini_dev_data"
    db_dir = root / "dev_databases" / "shop"
    (db_dir / "database_description").mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "shop.sqlite")
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO items VALUES (?, ?)", [(i, f"n{i}") for i in range(10)])
    conn.commit()
    conn.close()
    header = "original_column_name, column_name ,column_description,data_format,value_description\n"
    text = header + "id,item id,the id,integer,\nname,,caf\xe9 name,text,some values\n"
    (db_dir / "database_description" / "items.csv").write_bytes(text.encode("cp1252"))
    (root / "mini_dev_sqlite.json").write_text(json.dumps(records), encoding="utf-8")
    return tmp_path / "bird_data"


def records(extra=()):
    base = [
        {"question_id": 10 + i, "db_id": "shop", "question": f"q{i}", "evidence": "e",
         "SQL": f"SELECT name FROM items WHERE id = {i}", "difficulty": "simple"}
        for i in range(6)
    ]
    base.append({"db_id": "shop", "question": "no id", "evidence": "",
                 "SQL": "SELECT COUNT(*) FROM items", "difficulty": "moderate"})
    base.append({"question_id": 99, "db_id": "shop", "question": "bad", "evidence": "",
                 "SQL": "SELECT nope FROM items", "difficulty": "challenging"})
    return base + list(extra)


def run(tmp_path, bird_dir, name="out", n=50, seed=42):
    out = tmp_path / name / "bird.jsonl"
    manifest = build_dataset(bird_dir, n, seed, out, tmp_path / name / "docs", 5.0)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return manifest, rows, out


def test_loader_keeps_valid_rows_and_counts_drops(tmp_path):
    manifest, rows, out = run(tmp_path, make_tree(tmp_path, records()))
    assert manifest["kept"] == 7
    assert manifest["drops"] == {"sql_error": 1}
    assert manifest["difficulty"] == {"simple": 6, "moderate": 1}
    assert len(manifest["source_sha256"]) == 64
    assert (out.parent / "manifest.json").exists()
    ids = {r["id"] for r in rows}
    assert "bird_10" in ids and "bird_6" in ids  # missing question_id falls back to position
    row = rows[0]
    assert row["db"].endswith("shop.sqlite") and row["docs"].endswith("shop.md")
    assert set(row) == {
        "id", "db_id", "difficulty", "question", "evidence", "gold_sql", "db", "docs"
    }


def test_docs_decode_cp1252_and_split_by_table(tmp_path):
    _, rows, _ = run(tmp_path, make_tree(tmp_path, records()))
    text = open(rows[0]["docs"], encoding="utf-8").read()
    assert "café name" in text
    chunks = split_docs(text)
    assert chunks[0] == "# Database shop"
    assert chunks[1].startswith("## Table items")
    assert "- `id` (item id): the id. Format: integer" in chunks[1]
    assert "- `name`: café name. Format: text. Values: some values" in chunks[1]


def test_same_seed_gives_same_ids_and_n_limits_rows(tmp_path):
    bird_dir = make_tree(tmp_path, records())
    _, first, _ = run(tmp_path, bird_dir, "a", n=4, seed=1)
    _, second, _ = run(tmp_path, bird_dir, "b", n=4, seed=1)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert len(first) == 4


def test_write_statement_as_gold_stops_without_output(tmp_path, capsys):
    bad = {"question_id": 500, "db_id": "shop", "question": "w", "evidence": "",
           "SQL": "DELETE FROM items", "difficulty": "simple"}
    bird_dir = make_tree(tmp_path, records([bad]))
    out = tmp_path / "out" / "bird.jsonl"
    with pytest.raises(GuardrailRejected) as info:
        build_dataset(bird_dir, 50, 42, out, tmp_path / "out" / "docs", 5.0)
    assert info.value.ids == ["bird_500"]
    assert not out.exists() and not (out.parent / "manifest.json").exists()

    code = main(["--bird-dir", str(bird_dir), "--out", str(out),
                 "--docs-dir", str(tmp_path / "out" / "docs")])
    assert code == 1
    assert "bird_500" in capsys.readouterr().out
    assert not out.exists()
