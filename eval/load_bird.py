"""Build a gold file from the BIRD mini-dev set.

Usage (use an absolute path for --bird-dir):
    python eval/load_bird.py --bird-dir F:\\Downloads\\Portfolio_Projects\\bird_data

What it does:
1. Finds mini_dev_sqlite.json and the dev_databases folder somewhere under --bird-dir.
2. Shuffles all records with a fixed seed and walks the list.
3. Runs each gold SQL through the same guarded Database the agent uses. A record is kept
   only if the query runs, finishes in time and returns no more rows than the scoring limit.
4. Writes the kept records (n at most) as JSONL, one markdown docs file per database
   (built from database_description/*.csv), and a manifest.json with counts.

If our guardrails reject a gold query, the script stops without writing anything: valid
benchmark SQL must not be blocked, so that case has to be fixed in db.py, not hidden here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

# With `python -I` the script's folder is not on sys.path; keep it consistent with run_eval.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyst_agent.db import Database  # noqa: E402
from score import SCORING_ROW_LIMIT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JSON_NAME = "mini_dev_sqlite.json"
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
_GUARDRAIL_MARKERS = (
    "only select or with",
    "only one sql statement",
    "empty sql",
    "not authorized",
    "readonly",
    "read-only",
)


class GuardrailRejected(RuntimeError):
    """Raised when our guardrails reject a gold query. Carries the offending ids."""

    def __init__(self, ids: list[str]):
        super().__init__(f"{len(ids)} gold queries rejected by the guardrails")
        self.ids = ids


# ------------------------------------------------------------------ finding files


def find_json(bird_dir: Path) -> Path:
    """The mini_dev_sqlite.json file, wherever it is under bird_dir."""
    matches = sorted(bird_dir.rglob(JSON_NAME))
    if not matches:
        raise FileNotFoundError(f"{JSON_NAME} not found under {bird_dir}")
    return matches[0]


def find_database(bird_dir: Path, db_id: str) -> Path | None:
    """The .sqlite file of one database: dev_databases/<db_id>/**/*.sqlite, or None."""
    for folder in sorted(bird_dir.rglob("dev_databases")):
        files = sorted((folder / db_id).rglob("*.sqlite")) if (folder / db_id).is_dir() else []
        if files:
            return files[0].resolve()
    return None


def read_text_any(path: Path) -> str:
    """Decode a file with utf-8-sig, then cp1252, then latin-1 (latin-1 never fails)."""
    data = path.read_bytes()
    for encoding in ENCODINGS[:-1]:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode(ENCODINGS[-1])


# ------------------------------------------------------------------ docs


def _clean(value: str | None) -> str:
    """One line of text: whitespace runs (including newlines) become single spaces."""
    return " ".join((value or "").split())


def column_bullet(row: dict) -> str | None:
    """One markdown bullet for one description row, or None if the row has no column name."""
    original = _clean(row.get("original_column_name"))
    name = _clean(row.get("column_name"))
    if not original and not name:
        return None
    head = f"- `{original or name}`" + (f" ({name})" if original and name else "")
    parts = [_clean(row.get("column_description"))]
    if _clean(row.get("data_format")):
        parts.append(f"Format: {_clean(row.get('data_format'))}")
    if _clean(row.get("value_description")):
        parts.append(f"Values: {_clean(row.get('value_description'))}")
    parts = [p for p in parts if p]
    return head + (": " + ". ".join(p.rstrip(".") for p in parts) if parts else "")


def read_description_csv(path: Path) -> list[dict]:
    """Rows of one description CSV, with header names stripped of whitespace."""
    reader = csv.reader(io.StringIO(read_text_any(path)))
    header = [h.strip() for h in next(reader, [])]
    return [dict(zip(header, cells, strict=False)) for cells in reader if any(cells)]


def build_docs(bird_dir: Path, db_id: str) -> str:
    """Markdown docs for one database: a '## Table' section per description CSV."""
    lines = [f"# Database {db_id}"]
    for folder in sorted(bird_dir.rglob("dev_databases")):
        desc_dir = folder / db_id / "database_description"
        if not desc_dir.is_dir():
            continue
        for csv_path in sorted(desc_dir.glob("*.csv")):
            bullets = [column_bullet(r) for r in read_description_csv(csv_path)]
            lines += ["", f"## Table {csv_path.stem}", ""]
            lines += [b for b in bullets if b]
        break
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ validating gold SQL


def classify_failure(error: str) -> str:
    """Drop reason for a failed gold query: guardrail_rejected, timeout or sql_error."""
    lowered = error.lower()
    if "timed out" in lowered:
        return "timeout"
    if any(marker in lowered for marker in _GUARDRAIL_MARKERS):
        return "guardrail_rejected"
    return "sql_error"


def check_gold(db: Database, sql: str, timeout_s: float) -> str | None:
    """None if the gold SQL is usable, else the drop reason."""
    result = db.run_sql(sql, limit=SCORING_ROW_LIMIT, timeout_s=timeout_s)
    if not result["ok"]:
        return classify_failure(result["error"])
    return "too_many_rows" if result["truncated"] else None


def load_records(json_path: Path) -> list[tuple[str, dict]]:
    """(question id, record) pairs; the id falls back to the position in the file."""
    records = json.loads(json_path.read_text(encoding="utf-8"))
    return [(str(r.get("question_id", i)), r) for i, r in enumerate(records)]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_records(
    bird_dir: Path, records: list[tuple[str, dict]], n: int, seed: int, gold_timeout: float
) -> tuple[list[dict], Counter, list[str]]:
    """Walk the shuffled records; return (kept rows without docs, drop counts, rejected ids)."""
    order = list(records)
    random.Random(seed).shuffle(order)
    dbs: dict[str, Database | None] = {}
    kept, drops, rejected = [], Counter(), []
    for qid, rec in order:
        if len(kept) >= n:
            break
        db_id = rec["db_id"]
        if db_id not in dbs:
            path = find_database(bird_dir, db_id)
            dbs[db_id] = Database(path) if path else None
        if dbs[db_id] is None:
            drops["missing_db"] += 1
            continue
        reason = check_gold(dbs[db_id], rec["SQL"], gold_timeout)
        if reason == "guardrail_rejected":
            rejected.append(f"bird_{qid}")
        if reason:
            drops[reason] += 1
            continue
        kept.append({
            "id": f"bird_{qid}",
            "db_id": db_id,
            "difficulty": rec.get("difficulty"),
            "question": rec["question"],
            "evidence": rec.get("evidence") or "",
            "gold_sql": rec["SQL"],
            "db": str(dbs[db_id].path),
        })
    return kept, drops, rejected


def build_dataset(
    bird_dir: Path, n: int, seed: int, out: Path, docs_dir: Path, gold_timeout: float
) -> dict:
    """Create the gold JSONL, the docs files and the manifest. Returns the manifest."""
    json_path = find_json(bird_dir)
    kept, drops, rejected = select_records(
        bird_dir, load_records(json_path), n, seed, gold_timeout
    )
    if rejected:
        raise GuardrailRejected(rejected)  # nothing has been written yet

    docs_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    for db_id in sorted({row["db_id"] for row in kept}):
        (docs_dir / f"{db_id}.md").write_text(build_docs(bird_dir, db_id), encoding="utf-8")
    for row in kept:
        row["docs"] = str((docs_dir / f"{row['db_id']}.md").resolve())
    with out.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in kept)

    manifest = {
        "source_json": str(json_path),
        "source_sha256": sha256_of(json_path),
        "n": n,
        "seed": seed,
        "kept": len(kept),
        "drops": dict(drops),
        "difficulty": dict(Counter(row["difficulty"] for row in kept)),
    }
    manifest_path = out.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--bird-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "bird" / "bird150.jsonl")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "data" / "bird" / "docs")
    parser.add_argument("--gold-timeout", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_dataset(
            args.bird_dir, args.n, args.seed, args.out, args.docs_dir, args.gold_timeout
        )
    except GuardrailRejected as exc:
        print("ERROR: the guardrails rejected valid-looking gold SQL. Nothing was written.")
        print("Rejected ids: " + ", ".join(exc.ids))
        return 1
    print(f"Kept {manifest['kept']} of {args.n} requested (seed {args.seed}).")
    print(f"Dropped: {manifest['drops'] or 'none'}")
    print(f"Difficulty: {manifest['difficulty']}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
