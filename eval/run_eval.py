"""Run the agent on a gold file and save one JSON line per question.

Examples:
    python eval/run_eval.py                      # full agent on the demo questions
    python eval/run_eval.py --no-repair          # ablation: SQL errors are not repaired
    python eval/run_eval.py --no-docs            # ablation: search_docs is disabled
    python eval/run_eval.py --limit 3            # quick check on the first 3 questions
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from analyst_agent.agent import MAX_REPAIRS, Agent
from analyst_agent.db import Database
from analyst_agent.llm import LiteLLMClient
from analyst_agent.tools import DocsIndex, ToolBox
from score import execution_match

ROOT = Path(__file__).resolve().parents[1]


def load_gold(path: Path, limit: int | None = None) -> list[dict]:
    """Read a JSONL gold file: one {"id", "question", "gold_sql", ...} object per line."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows[:limit] if limit else rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--gold", type=Path, default=ROOT / "eval" / "demo_gold.jsonl")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "demo.db")
    parser.add_argument("--docs", type=Path, default=ROOT / "data" / "demo_docs.md")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model", default=None, help="Default: MODEL from .env")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    parser.add_argument("--no-repair", action="store_true", help="Disable the SQL repair loop")
    parser.add_argument("--no-docs", action="store_true", help="Disable the search_docs tool")
    return parser.parse_args()


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()

    db = Database(args.db)
    docs = None if args.no_docs else DocsIndex.from_file(args.docs)
    llm = LiteLLMClient(model=args.model)
    agent = Agent(llm, ToolBox(db, docs), max_repairs=0 if args.no_repair else MAX_REPAIRS)
    config = {"model": llm.model, "use_docs": not args.no_docs, "use_repair": not args.no_repair}

    questions = load_gold(args.gold, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"run_{stamp}.jsonl"

    with out_path.open("w", encoding="utf-8") as out:
        for i, item in enumerate(questions, start=1):
            result = agent.run(item["question"])
            pred_sql = result.answer.sql if result.answer else None
            score = execution_match(db, item["gold_sql"], pred_sql)
            row = {
                "id": item["id"],
                "difficulty": item.get("difficulty"),
                "question": item["question"],
                "gold_sql": item["gold_sql"],
                "config": config,
                **result.to_dict(),
                **score,
            }
            out.write(json.dumps(row) + "\n")
            print(f"[{i}/{len(questions)}] {item['id']} correct={score['correct']} "
                  f"status={result.status} steps={result.steps}")

    print(f"Wrote {out_path}")
    print(f"Next: python eval/analyze.py {out_path}")


if __name__ == "__main__":
    main()
