"""Tests for the bootstrap functions and the results report."""

import numpy as np

from analyze import bootstrap_ci, build_report, paired_bootstrap_diff


def test_bootstrap_is_deterministic_with_a_seed():
    values = np.arange(30, dtype=float) ** 2  # varied numbers, so different seeds differ
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)
    assert bootstrap_ci(values, seed=7) != bootstrap_ci(values, seed=8)


def test_bootstrap_interval_contains_the_mean():
    values = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=float)
    mean, low, high = bootstrap_ci(values)
    assert mean == 0.7
    assert low <= mean <= high
    assert 0.0 <= low and high <= 1.0


def test_bootstrap_of_constant_values_has_zero_width():
    assert bootstrap_ci(np.ones(20)) == (1.0, 1.0, 1.0)


def test_paired_difference_detects_a_clear_gain():
    assert paired_bootstrap_diff(np.zeros(50), np.ones(50)) == (1.0, 1.0, 1.0)


def test_paired_difference_of_identical_runs_is_zero():
    a = np.array([1, 0, 1, 0, 1, 1], dtype=float)
    assert paired_bootstrap_diff(a, a) == (0.0, 0.0, 0.0)


def _fake_run(correct_flags):
    """Rows shaped like the ones run_eval.py writes (only the fields analyze.py reads)."""
    return [
        {
            "id": f"q{i}",
            "correct": flag,
            "latency_ms": 100.0 + i,
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "cost_usd": None,
            "tool_calls": [{"name": "run_sql"}],
            "repairs_used": 0,
            "config": {"model": "test", "use_docs": True, "use_repair": True},
        }
        for i, flag in enumerate(correct_flags)
    ]


def test_report_has_summary_and_paired_section():
    run_a = _fake_run([True, False, True, False])
    run_b = _fake_run([True, True, True, False])
    report = build_report({"run_a": run_a, "run_b": run_b})
    assert "Execution accuracy" in report
    assert "Paired comparison" in report
    assert "n/a" in report  # no cost reported
