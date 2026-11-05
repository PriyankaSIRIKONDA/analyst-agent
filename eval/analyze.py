"""Summarise one or two eval runs: accuracy with bootstrap CIs, latency, tokens, cost.

Usage:
    python eval/analyze.py results/run_A.jsonl
    python eval/analyze.py results/run_A.jsonl results/run_B.jsonl   # adds a paired comparison

Writes results/results.md. Every number in that file comes from the run files you pass in.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

N_RESAMPLES = 1000
SEED = 0
ROOT = Path(__file__).resolve().parents[1]


def load_run(path: Path) -> list[dict]:
    """Read a run_*.jsonl file written by run_eval.py."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap CI.

    Resample the questions with replacement, take the mean each time, then read the
    2.5th and 97.5th percentiles of those means.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> tuple[float, float, float]:
    """Mean of (b - a) per question and its 95% bootstrap CI.

    The same resampled questions are used for both runs, so question difficulty cancels out.
    """
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return bootstrap_ci(diffs, n_resamples, seed)


def summarise(rows: list[dict]) -> dict:
    """All headline numbers for one run."""
    correct = np.array([1.0 if r["correct"] else 0.0 for r in rows])
    latency = np.array([r["latency_ms"] for r in rows])
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    acc, low, high = bootstrap_ci(correct)
    return {
        "n": len(rows),
        "accuracy": acc,
        "ci_low": low,
        "ci_high": high,
        "latency_median_ms": float(np.median(latency)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "prompt_tokens": float(np.mean([r["prompt_tokens"] for r in rows])),
        "completion_tokens": float(np.mean([r["completion_tokens"] for r in rows])),
        "cost_usd": float(np.mean(costs)) if costs else None,
        "tool_calls": float(np.mean([len(r["tool_calls"]) for r in rows])),
        "repairs": float(np.mean([r["repairs_used"] for r in rows])),
    }


STATUSES = ("ok", "max_steps", "sql_error", "llm_error", "no_answer")
FAILURE_REASONS = ("no_sql", "pred_sql_error", "result_mismatch")


def status_counts(rows: list[dict]) -> dict[str, int]:
    """How many questions ended with each agent status."""
    counts = Counter(r.get("status", "no_answer") for r in rows)
    return {status: counts.get(status, 0) for status in STATUSES}


def failure_reasons(rows: list[dict]) -> dict[str, int]:
    """Why wrong answers were wrong: no_sql, pred_sql_error or result_mismatch."""
    counts: Counter = Counter()
    for r in rows:
        if not r["correct"]:
            reason = str(r.get("reason", "")).split(":")[0] or "unknown"
            counts[reason] += 1
    return {reason: counts.get(reason, 0) for reason in FAILURE_REASONS}


def by_difficulty(rows: list[dict]) -> dict[str, tuple[int, float]]:
    """Per difficulty label: (number of questions, accuracy). Rows without a label are 'n/a'."""
    groups: dict[str, list[bool]] = {}
    for r in rows:
        groups.setdefault(r.get("difficulty") or "n/a", []).append(bool(r["correct"]))
    return {label: (len(v), sum(v) / len(v)) for label, v in sorted(groups.items())}


def pair_runs(
    first: list[dict], second: list[dict]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Match two runs by question id. Raises ValueError if the id sets differ."""
    by_a = {r["id"]: r for r in first}
    by_b = {r["id"]: r for r in second}
    if len(by_a) != len(first) or len(by_b) != len(second):
        raise ValueError("A run contains the same question id twice; cannot pair the runs.")
    if set(by_a) != set(by_b):
        only_a, only_b = sorted(set(by_a) - set(by_b)), sorted(set(by_b) - set(by_a))
        raise ValueError(
            "The two runs are not on the same questions: "
            f"{len(only_a)} ids only in the first (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in the second (e.g. {only_b[:3]})."
        )
    ids = sorted(by_a)
    a = np.array([1.0 if by_a[i]["correct"] else 0.0 for i in ids])
    b = np.array([1.0 if by_b[i]["correct"] else 0.0 for i in ids])
    return ids, a, b


def warnings_for(runs: dict[str, list[dict]]) -> list[str]:
    """One warning per run where the model API failed (those questions are not real misses)."""
    out = []
    for name, rows in runs.items():
        errors = status_counts(rows)["llm_error"]
        if errors:
            out.append(
                f"WARNING: {name} has {errors} llm_error rows (API failures count as wrong "
                "answers). Re-run before trusting the accuracy."
            )
    return out


def pct(x: float) -> str:
    """0.835 -> '83.5%'."""
    return f"{100 * x:.1f}%"


def config_section(runs: dict[str, list[dict]]) -> list[str]:
    """The settings of each run (from its first row), so results.md explains itself."""
    lines = ["## Run configuration", ""]
    for name, rows in runs.items():
        config = rows[0].get("config", {}) if rows else {}
        lines.append(f"- `{name}`: " + (", ".join(f"{k}={v}" for k, v in config.items()) or "n/a"))
    return lines + [""]


def detail_sections(runs: dict[str, list[dict]]) -> list[str]:
    """Status counts, failure reasons and accuracy by difficulty for each run."""
    lines = ["", "## Status of each run", "", "| Run | " + " | ".join(STATUSES) + " |"]
    lines.append("|---|" + "---|" * len(STATUSES))
    for name, rows in runs.items():
        counts = status_counts(rows)
        lines.append(f"| {name} | " + " | ".join(str(counts[s]) for s in STATUSES) + " |")
    lines += ["", "## Failure reasons (wrong answers only)", ""]
    lines += ["| Run | " + " | ".join(FAILURE_REASONS) + " |"]
    lines.append("|---|" + "---|" * len(FAILURE_REASONS))
    for name, rows in runs.items():
        reasons = failure_reasons(rows)
        lines.append(f"| {name} | " + " | ".join(str(reasons[r]) for r in FAILURE_REASONS) + " |")
    lines += ["", "## Accuracy by difficulty", "", "| Run | Difficulty | N | Accuracy |"]
    lines.append("|---|---|---|---|")
    for name, rows in runs.items():
        for label, (n, acc) in by_difficulty(rows).items():
            lines.append(f"| {name} | {label} | {n} | {pct(acc)} |")
    return lines


def build_report(runs: dict[str, list[dict]]) -> str:
    """Markdown text for results.md."""
    summaries = {name: summarise(rows) for name, rows in runs.items()}

    title = f"## Execution accuracy (95% bootstrap CI, {N_RESAMPLES} resamples)"
    lines = ["# Results", ""]
    lines += config_section(runs)
    lines += [title, ""]
    lines += [
        "| Run | Model | Docs | Repair | N | Accuracy | 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, rows in runs.items():
        s, cfg = summaries[name], rows[0].get("config", {})
        lines.append(
            f"| {name} | {cfg.get('model', '?')} | {cfg.get('use_docs', '?')} | "
            f"{cfg.get('use_repair', '?')} | {s['n']} | {pct(s['accuracy'])} | "
            f"[{pct(s['ci_low'])}, {pct(s['ci_high'])}] |"
        )

    lines += [
        "",
        "## Latency, tokens and cost per question",
        "",
        "| Run | Median latency (ms) | p95 latency (ms) | Prompt tokens | "
        "Completion tokens | Cost (USD) | Tool calls | Repairs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        cost = f"{s['cost_usd']:.5f}" if s["cost_usd"] is not None else "n/a"
        lines.append(
            f"| {name} | {s['latency_median_ms']:.0f} | {s['latency_p95_ms']:.0f} | "
            f"{s['prompt_tokens']:.0f} | {s['completion_tokens']:.0f} | {cost} | "
            f"{s['tool_calls']:.1f} | {s['repairs']:.2f} |"
        )

    lines += detail_sections(runs)

    names = list(runs)
    if len(names) == 2:
        first, second = names
        shared, a, b = pair_runs(runs[first], runs[second])
        diff, low, high = paired_bootstrap_diff(a, b)
        verdict = "CI excludes 0" if low > 0 or high < 0 else "CI includes 0, no clear difference"
        lines += [
            "",
            "## Paired comparison",
            "",
            f"`{second}` minus `{first}` on {len(shared)} shared questions "
            "(the same resampled questions are used for both runs):",
            "",
            f"- Accuracy difference: {100 * diff:+.1f} points, "
            f"95% CI [{100 * low:+.1f}, {100 * high:+.1f}] ({verdict}).",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("runs", nargs="+", type=Path, help="One or two run_*.jsonl files")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "results.md")
    args = parser.parse_args()
    if len(args.runs) > 2:
        parser.error("Pass one run, or two runs for a paired comparison.")

    runs = {path.stem: load_run(path) for path in args.runs}
    try:
        report = build_report(runs)
    except ValueError as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.out}")
    for warning in warnings_for(runs):
        print(warning)


if __name__ == "__main__":
    main()
