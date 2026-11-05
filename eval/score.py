"""Execution-match scoring.

A predicted SQL query is correct if it returns the same result set as the gold SQL:
- rows are compared as multisets (order does not matter) unless the gold SQL has ORDER BY;
- floats are rounded to 4 decimal places;
- column names are ignored, column order is not.
"""

from __future__ import annotations

import re
from collections import Counter

from analyst_agent.db import Database

# Gold queries can return more rows than the agent's 200-row limit, so scoring uses its own.
SCORING_ROW_LIMIT = 100_000
SCORING_TIMEOUT_S = 30.0

_STRING_RE = re.compile(r"'(?:[^']|'')*'")
_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def has_order_by(sql: str) -> bool:
    """True if the query contains ORDER BY outside string literals."""
    return bool(_ORDER_BY_RE.search(_STRING_RE.sub("''", sql)))


def normalise_value(value):
    """Round floats to 4 dp so that 3.14159 and 3.1416 compare equal."""
    if isinstance(value, float):
        return round(value, 4)
    return value


def normalise_rows(rows: list[list]) -> list[tuple]:
    return [tuple(normalise_value(v) for v in row) for row in rows]


def results_match(gold_rows: list[list], pred_rows: list[list], ordered: bool) -> bool:
    """Compare two result sets. `ordered` means the row order must be identical too."""
    gold = normalise_rows(gold_rows)
    pred = normalise_rows(pred_rows)
    if ordered:
        return gold == pred
    return Counter(gold) == Counter(pred)


def execution_match(db: Database, gold_sql: str, pred_sql: str | None) -> dict:
    """Run both queries and compare. Returns {"correct": bool, "reason": str}."""
    if not pred_sql or not pred_sql.strip():
        return {"correct": False, "reason": "no_sql"}

    gold = db.run_sql(gold_sql, limit=SCORING_ROW_LIMIT, timeout_s=SCORING_TIMEOUT_S)
    if not gold["ok"]:
        raise ValueError(f"Gold SQL failed, fix the gold file: {gold['error']}")

    pred = db.run_sql(pred_sql, limit=SCORING_ROW_LIMIT, timeout_s=SCORING_TIMEOUT_S)
    if not pred["ok"]:
        return {"correct": False, "reason": f"pred_sql_error: {pred['error']}"}

    if results_match(gold["rows"], pred["rows"], ordered=has_order_by(gold_sql)):
        return {"correct": True, "reason": "match"}
    return {"correct": False, "reason": "result_mismatch"}
