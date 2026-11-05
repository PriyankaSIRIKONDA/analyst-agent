"""Tests for the regression gate."""

import json

from check_regression import main


def write_run(path, flags, model="m"):
    config = {"model": model, "metric": "strict", "gold": "g.jsonl"}
    rows = [{"id": f"q{i}", "correct": f, "config": config} for i, f in enumerate(flags)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_missing_baseline_exits_1_with_instructions(tmp_path, capsys):
    run = write_run(tmp_path / "run.jsonl", [True] * 10)
    assert main([str(run), "--baseline", str(tmp_path / "baseline.json")]) == 1
    assert "--write-baseline" in capsys.readouterr().out


def test_write_baseline_then_pass_and_fail(tmp_path, capsys):
    base_path = tmp_path / "baseline.json"
    good = write_run(tmp_path / "good.jsonl", [True] * 16 + [False] * 4)  # 80%
    assert main([str(good), "--baseline", str(base_path), "--write-baseline"]) == 0
    baseline = json.loads(base_path.read_text(encoding="utf-8"))
    assert set(baseline) == {
        "accuracy", "ci_low", "ci_high", "n", "model", "metric", "gold", "run_file", "date"
    }
    assert baseline["accuracy"] == 0.8 and baseline["n"] == 20

    assert main([str(good), "--baseline", str(base_path)]) == 0
    bad = write_run(tmp_path / "bad.jsonl", [True] * 8 + [False] * 12)  # 40%
    assert main([str(bad), "--baseline", str(base_path)]) == 1
    out = capsys.readouterr().out
    assert "baseline CI lower bound" in out and "REGRESSION" in out


def test_min_accuracy(tmp_path):
    perfect = write_run(tmp_path / "p.jsonl", [True] * 5)
    one_miss = write_run(tmp_path / "m.jsonl", [True] * 4 + [False])
    assert main([str(perfect), "--min-accuracy", "1.0"]) == 0
    assert main([str(one_miss), "--min-accuracy", "1.0"]) == 1
    assert main([str(one_miss), "--min-accuracy", "0.8"]) == 0
