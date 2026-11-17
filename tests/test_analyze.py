"""Tests for the bootstrap functions and the results report."""

import numpy as np
import pytest

from analyze import (
    bootstrap_ci,
    build_report,
    by_difficulty,
    failure_reasons,
    pair_runs,
    paired_bootstrap_diff,
    prepare_run,
    status_counts,
    warnings_for,
)


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


def _row(qid, correct, **extra):
    row = _fake_run([correct])[0]
    row["id"] = qid
    row.update(extra)
    return row


def test_pairing_is_by_id_not_by_position():
    first = [_row("a", True), _row("b", False)]
    second = [_row("b", True), _row("a", True)]  # same ids, other order
    ids, a, b = pair_runs(first, second)
    assert ids == ["a", "b"]
    assert list(a) == [1.0, 0.0] and list(b) == [1.0, 1.0]


def test_pairing_different_id_sets_raises_a_clear_error():
    with pytest.raises(ValueError, match="not on the same questions"):
        pair_runs([_row("a", True), _row("b", True)], [_row("a", True), _row("c", True)])
    with pytest.raises(ValueError, match="twice"):
        pair_runs([_row("a", True), _row("a", True)], [_row("a", True)])
    with pytest.raises(ValueError):
        build_report({"x": [_row("a", True)], "y": [_row("b", True)]})


def test_status_reasons_and_difficulty_tables():
    rows = [
        _row("a", True, status="ok", difficulty="simple", reason="match"),
        _row("b", False, status="ok", difficulty="simple", reason="result_mismatch"),
        _row("c", False, status="llm_error", difficulty="hard", reason="no_sql"),
        _row("d", False, status="max_steps", difficulty="hard", reason="pred_sql_error: boom"),
    ]
    assert status_counts(rows) == {
        "ok": 2, "max_steps": 1, "sql_error": 0, "llm_error": 1, "no_answer": 0
    }
    assert failure_reasons(rows) == {"no_sql": 1, "pred_sql_error": 1, "result_mismatch": 1}
    assert by_difficulty(rows) == {"hard": (2, 0.0), "simple": (2, 0.5)}
    assert len(warnings_for({"r": rows})) == 1
    assert warnings_for({"r": rows[:2]}) == []


def test_report_shows_config_and_the_new_sections():
    report = build_report({"run_a": _fake_run([True, False])})
    assert report.index("Run configuration") < report.index("Execution accuracy")
    assert "model=test" in report
    for heading in ("Status of each run", "Failure reasons", "Accuracy by difficulty"):
        assert heading in report


def test_prepare_run_excludes_ids_and_counts_a_repeated_question_once():
    rows = [
        _row("a", True, question="Q a"),
        _row("b", False, question="Q b"),
        _row("a", True, question="Q a"),  # the dataset lists question a twice
        _row("c", True, question="Q c"),
    ]
    kept, excluded, collapsed = prepare_run(rows, {"c", "missing"})
    assert [r["id"] for r in kept] == ["a", "b"]
    assert excluded == ["c"]
    assert collapsed == ["a"]


def test_prepare_run_refuses_one_id_for_two_different_questions():
    with pytest.raises(ValueError, match="two different questions"):
        prepare_run([_row("a", True, question="one"), _row("a", True, question="two")], set())


def test_runs_with_a_dataset_duplicate_can_be_paired_after_prepare():
    first, _, _ = prepare_run([_row("a", True, question="Q"), _row("a", True, question="Q"),
                               _row("b", False, question="R"), _row("x", True, question="X")],
                              {"x"})
    second, _, _ = prepare_run([_row("a", False, question="Q"), _row("b", True, question="R")],
                               {"x"})
    ids, a, b = pair_runs(first, second)
    assert ids == ["a", "b"]


def test_report_lists_left_out_questions_and_every_git_commit():
    rows = _fake_run([True, False])
    rows[1]["config"] = dict(rows[1]["config"], git_commit="bbb")
    rows[0]["config"] = dict(rows[0]["config"], git_commit="aaa")
    report = build_report({"r": rows}, {"r": (["bird_701"], ["bird_138"])})
    assert "left out with --exclude: bird_701" in report
    assert "duplicate question counted once: bird_138" in report
    assert "aaa (1 rows) + bbb (1 rows)" in report
    assert "Questions left out" not in build_report({"r": rows})
