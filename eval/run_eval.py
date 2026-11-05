"""Run the agent on a gold file and save one JSON line per question.

Examples:
    python eval/run_eval.py                      # full agent on the demo questions
    python eval/run_eval.py --no-repair          # ablation: SQL errors are not repaired
    python eval/run_eval.py --no-docs            # ablation: search_docs is disabled
    python eval/run_eval.py --limit 3            # quick check on the first 3 questions
    python eval/run_eval.py --oracle             # no LLM: the gold SQL is replayed (must be 100%)
    python eval/run_eval.py --gold data/bird/bird150.jsonl --metric bird --min-interval 1.1

A gold row may carry "db", "docs" and "evidence". "db" and "docs" replace --db and --docs for
that row; "evidence" is appended to the question as a hint unless --no-evidence is given.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# With `python -I` the script's folder is not on sys.path, so add it before importing score.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from analyst_agent.agent import MAX_REPAIRS, MAX_STEPS, Agent  # noqa: E402
from analyst_agent.db import DEFAULT_TIMEOUT_S, Database  # noqa: E402
from analyst_agent.llm import FakeLLM, LiteLLMClient, tool_turn  # noqa: E402
from analyst_agent.tools import DocsIndex, ToolBox  # noqa: E402
from score import execution_match  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_gold(path: Path, limit: int | None = None) -> list[dict]:
    """Read a JSONL gold file: one {"id", "question", "gold_sql", ...} object per line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return rows[:limit] if limit else rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--gold", type=Path, default=ROOT / "eval" / "demo_gold.jsonl")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "demo.db")
    parser.add_argument("--docs", type=Path, default=ROOT / "data" / "demo_docs.md")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=None, help="Output file (default: run_<stamp>)")
    parser.add_argument("--model", default=None, help="Default: MODEL from .env")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    parser.add_argument("--no-repair", action="store_true", help="Disable the SQL repair loop")
    parser.add_argument("--no-docs", action="store_true", help="Disable the search_docs tool")
    parser.add_argument("--no-evidence", action="store_true", help="Do not add the evidence hint")
    parser.add_argument("--metric", choices=["strict", "bird"], default="strict")
    parser.add_argument("--sql-timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--min-interval", type=float, default=0.0,
                        help="Seconds between LLM calls (60 / your requests-per-minute limit)")
    parser.add_argument("--oracle", action="store_true",
                        help="No LLM: replay the gold SQL through the real tools and scorer")
    return parser.parse_args(argv)


# ------------------------------------------------------------------ small helpers


def git_short_hash() -> str | None:
    """Short hash of the current commit, or None if git is missing or this is not a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def question_text(item: dict, use_evidence: bool = True) -> str:
    """The text given to the agent: the question, plus the evidence as a hint if there is any."""
    evidence = (item.get("evidence") or "").strip()
    if use_evidence and evidence:
        return item["question"] + "\n\nHint: " + evidence
    return item["question"]


class Resources:
    """Opens each database and docs file once. A gold row can point to its own db and docs."""

    def __init__(self, default_db: Path, default_docs: Path, use_docs: bool = True):
        self.default_db = default_db
        self.default_docs = default_docs
        self.use_docs = use_docs
        self._dbs: dict[str, Database] = {}
        self._docs: dict[str, DocsIndex] = {}

    def database(self, item: dict) -> Database:
        path = str(item.get("db") or self.default_db)
        if path not in self._dbs:
            self._dbs[path] = Database(path)
        return self._dbs[path]

    def docs(self, item: dict) -> DocsIndex | None:
        if not self.use_docs:
            return None
        path = str(item.get("docs") or self.default_docs)
        if path not in self._docs:
            self._docs[path] = DocsIndex.from_file(path)
        return self._docs[path]


def oracle_llm(gold_sql: str) -> FakeLLM:
    """A scripted model: run the gold SQL, then give it as the final answer."""
    return FakeLLM([
        tool_turn(("run_sql", {"sql": gold_sql})),
        tool_turn(("final_answer", {"answer": "oracle", "sql": gold_sql, "tables_used": []})),
    ])


def build_config(args: argparse.Namespace, model: str) -> dict:
    """Settings recorded in every row, so a run file explains itself."""
    return {
        "model": model,
        "use_docs": not args.no_docs,
        "use_repair": not args.no_repair,
        "use_evidence": not args.no_evidence,
        "metric": args.metric,
        "max_steps": args.max_steps,
        "sql_timeout": args.sql_timeout,
        "gold": args.gold.name,
        "git_commit": git_short_hash(),
    }


def run_question(
    item: dict, args: argparse.Namespace, resources: Resources, llm, config: dict
) -> dict:
    """Run the agent on one gold row, score it, and return the output row.

    `llm` is None in oracle mode: a scripted model is made for the row.
    """
    db = resources.database(item)
    toolbox = ToolBox(db, resources.docs(item), sql_timeout_s=args.sql_timeout)
    agent = Agent(
        llm if llm is not None else oracle_llm(item["gold_sql"]),
        toolbox,
        max_steps=args.max_steps,
        max_repairs=0 if args.no_repair else MAX_REPAIRS,
    )
    result = agent.run(question_text(item, use_evidence=not args.no_evidence))
    pred_sql = result.answer.sql if result.answer else None
    score = execution_match(db, item["gold_sql"], pred_sql, metric=args.metric)
    return {
        **result.to_dict(),
        "id": item["id"],
        "difficulty": item.get("difficulty"),
        "db_id": item.get("db_id"),
        "question": item["question"],  # the original text, without the hint
        "gold_sql": item["gold_sql"],
        "config": config,
        **score,
    }


def output_path(args: argparse.Namespace) -> Path:
    if args.out is not None:
        return args.out
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.out_dir / f"run_{stamp}.jsonl"


def run_all(args: argparse.Namespace) -> Path:
    """Run every question, write the JSONL file, and return its path."""
    if args.oracle:
        llm, model = None, "oracle"
    else:
        llm = LiteLLMClient(model=args.model, min_interval_s=args.min_interval)
        model = llm.model
    config = build_config(args, model)
    resources = Resources(args.db, args.docs, use_docs=not args.no_docs)
    questions = load_gold(args.gold, args.limit)

    out_path = output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for i, item in enumerate(questions, start=1):
            row = run_question(item, args, resources, llm, config)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{i}/{len(questions)}] {item['id']} correct={row['correct']} "
                  f"status={row['status']} steps={row['steps']}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.oracle:
        load_dotenv(ROOT / ".env")
    out_path = run_all(args)
    print(f"Wrote {out_path}")
    print(f"Next: python eval/analyze.py {out_path}")


if __name__ == "__main__":
    main()
