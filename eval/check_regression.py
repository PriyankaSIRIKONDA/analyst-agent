"""Fail (exit 1) when a run is worse than the stored baseline.

Usage:
    python eval/check_regression.py results/run_X.jsonl --baseline eval/baseline.json
    python eval/check_regression.py results/run_X.jsonl --write-baseline
    python eval/check_regression.py results/ci/oracle.jsonl --min-accuracy 1.0

The run fails if its accuracy is below the lower end of the baseline's 95% bootstrap CI.
--min-accuracy checks a fixed threshold instead (used for the oracle smoke test).
The baseline must come from a real run, so it is never created automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

# With `python -I` the script's folder is not on sys.path, so add it before importing analyze.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import bootstrap_ci, load_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run_accuracy(rows: list[dict]) -> float:
    """Share of correct questions."""
    return sum(1 for r in rows if r["correct"]) / len(rows)


def make_baseline(rows: list[dict], run_file: Path) -> dict:
    """The record stored in baseline.json for this run."""
    correct = np.array([1.0 if r["correct"] else 0.0 for r in rows])
    accuracy, low, high = bootstrap_ci(correct)
    config = rows[0].get("config", {})
    return {
        "accuracy": accuracy,
        "ci_low": low,
        "ci_high": high,
        "n": len(rows),
        "model": config.get("model"),
        "metric": config.get("metric"),
        "gold": config.get("gold"),
        "run_file": run_file.name,
        "date": date.today().isoformat(),
    }


def check_threshold(accuracy: float, minimum: float) -> int:
    """Exit code for --min-accuracy."""
    print(f"run accuracy {accuracy:.4f}, required minimum {minimum:.4f}")
    return 0 if accuracy >= minimum else 1


def check_baseline(accuracy: float, baseline: dict) -> int:
    """Exit code for a baseline check: 1 if the run is below the baseline CI lower bound."""
    low = baseline["ci_low"]
    print(f"run accuracy {accuracy:.4f}, baseline CI lower bound {low:.4f} "
          f"(baseline accuracy {baseline['accuracy']:.4f}, n={baseline['n']})")
    if accuracy < low:
        print("REGRESSION: the run is below the baseline confidence interval.")
        return 1
    print("OK: the run is not below the baseline confidence interval.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("run", type=Path, help="A run_*.jsonl file")
    parser.add_argument("--baseline", type=Path, default=ROOT / "eval" / "baseline.json")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--min-accuracy", type=float, default=None)
    args = parser.parse_args(argv)

    rows = load_run(args.run)
    if not rows:
        print(f"ERROR: {args.run} has no rows.")
        return 1

    if args.write_baseline:
        baseline = make_baseline(rows, args.run)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote baseline {args.baseline}: accuracy {baseline['accuracy']:.4f}, "
              f"CI [{baseline['ci_low']:.4f}, {baseline['ci_high']:.4f}], n={baseline['n']}")
        return 0

    if args.min_accuracy is not None:
        return check_threshold(run_accuracy(rows), args.min_accuracy)

    if not args.baseline.exists():
        print(f"ERROR: no baseline at {args.baseline}.")
        print("Create it from a real run you accept as the reference:")
        print(f"  python eval/check_regression.py {args.run} --baseline {args.baseline} "
              "--write-baseline")
        return 1
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    return check_baseline(run_accuracy(rows), baseline)


if __name__ == "__main__":
    sys.exit(main())
