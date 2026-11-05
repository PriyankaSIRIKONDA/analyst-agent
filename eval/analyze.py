"""Summarise one or two eval runs: accuracy with bootstrap CIs, latency, tokens, cost.

Usage:
    python eval/analyze.py results/run_A.jsonl
    python eval/analyze.py results/run_A.jsonl results/run_B.jsonl   # adds a paired comparison

Writes results/results.md. Every number in that file comes from the run files you pass in.
"""

from __future__ import annotations

import argparse
import json
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


def pct(x: float) -> str:
    """0.835 -> '83.5%'."""
    return f"{100 * x:.1f}%"


def build_report(runs: dict[str, list[dict]]) -> str:
    """Markdown text for results.md."""
    summaries = {name: summarise(rows) for name, rows in runs.items()}

    title = f"## Execution accuracy (95% bootstrap CI, {N_RESAMPLES} resamples)"
    lines = ["# Results", "", title, ""]
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

    names = list(runs)
    if len(names) == 2:
        first, second = names
        by_id_a = {r["id"]: r for r in runs[first]}
        by_id_b = {r["id"]: r for r in runs[second]}
        shared = sorted(set(by_id_a) & set(by_id_b))
        a = np.array([1.0 if by_id_a[i]["correct"] else 0.0 for i in shared])
        b = np.array([1.0 if by_id_b[i]["correct"] else 0.0 for i in shared])
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
    report = build_report(runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
