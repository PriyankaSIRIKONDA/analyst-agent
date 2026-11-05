"""Shared fixtures: a tiny SQLite database built fresh in a temp folder."""

import sqlite3

import pytest

from analyst_agent.db import Database
from analyst_agent.tools import DocsIndex, ToolBox


@pytest.fixture
def db(tmp_path) -> Database:
    """A database with one table `items` (id, name, price) holding 20 rows."""
    path = tmp_path / "tiny.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
    conn.executemany(
        "INSERT INTO items VALUES (?, ?, ?)", [(i, f"item{i}", i * 1.5) for i in range(1, 21)]
    )
    conn.commit()
    conn.close()
    return Database(path)


@pytest.fixture
def toolbox(db) -> ToolBox:
    """A toolbox over the tiny database with a three-chunk docs index."""
    docs = DocsIndex(
        [
            "# Intro\nA tiny shop.",
            "## Table items\nProducts for sale. `price` is the price in euros.",
            "## Business definitions\nRevenue means the sum of price, cancelled orders excluded.",
        ]
    )
    return ToolBox(db, docs)
