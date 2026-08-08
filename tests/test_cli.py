"""Tests for the colonyx CLI (optimize/benchmark/report subcommands)."""

from __future__ import annotations

import json

import pytest

from colonyx import cli


def test_optimize_runs_and_prints_json(capsys):
    exit_code = cli.main(
        [
            "optimize",
            "--mode",
            "pso",
            "--objective",
            "sphere",
            "--dimensions",
            "2",
            "--iterations",
            "10",
            "--random-state",
            "1",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "pso"
    assert payload["objective"] == "sphere"
    assert len(payload["best_solution"]) == 2
    assert isinstance(payload["best_score"], float)


def test_optimize_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        cli.main(["optimize", "--mode", "not-a-mode"])


def test_benchmark_runs_every_continuous_mode_and_prints_json(capsys):
    exit_code = cli.main(
        [
            "benchmark",
            "--objective",
            "sphere",
            "--dimensions",
            "2",
            "--iterations",
            "5",
            "--repeats",
            "1",
            "--random-state",
            "2",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == set(cli._CONTINUOUS_MODES)
    for result in payload.values():
        assert "best_score" in result


def test_benchmark_skips_a_failing_mode_instead_of_crashing(capsys, monkeypatch):
    # Regression test: one mode raising used to crash the whole batch. It
    # should now be reported on stderr and simply excluded from the results,
    # with every other mode still completing successfully.
    real_make_optimizer = cli._make_optimizer

    def flaky_make_optimizer(mode, iterations, random_state):
        if mode == "pso":
            raise RuntimeError("simulated failure")
        return real_make_optimizer(mode, iterations, random_state)

    monkeypatch.setattr(cli, "_make_optimizer", flaky_make_optimizer)

    exit_code = cli.main(
        [
            "benchmark",
            "--objective",
            "sphere",
            "--dimensions",
            "2",
            "--iterations",
            "5",
            "--repeats",
            "1",
            "--random-state",
            "2",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "pso" not in payload
    assert set(payload) == set(cli._CONTINUOUS_MODES) - {"pso"}
    assert "pso" in captured.err
    assert "simulated failure" in captured.err


def test_report_json_format(capsys):
    exit_code = cli.main(
        [
            "report",
            "--objective",
            "sphere",
            "--dimensions",
            "2",
            "--iterations",
            "5",
            "--repeats",
            "1",
            "--random-state",
            "3",
            "--format",
            "json",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == set(cli._CONTINUOUS_MODES)
    assert "mean_score" in next(iter(payload.values()))


def test_report_csv_format_written_to_file_ends_with_newline(tmp_path):
    # Regression test: CSV output written to a file used to be missing its
    # trailing newline (buffer.getvalue().strip() ate it).
    output_path = tmp_path / "report.csv"
    exit_code = cli.main(
        [
            "report",
            "--objective",
            "sphere",
            "--dimensions",
            "2",
            "--iterations",
            "5",
            "--repeats",
            "1",
            "--random-state",
            "4",
            "--format",
            "csv",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0

    text = output_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    lines = text.splitlines()
    assert lines[0].startswith("name,best_score")
    assert len(lines) == 1 + len(cli._CONTINUOUS_MODES)


def test_report_csv_printed_to_stdout_has_single_trailing_newline(capsys):
    exit_code = cli.main(
        [
            "report",
            "--objective",
            "sphere",
            "--dimensions",
            "2",
            "--iterations",
            "5",
            "--repeats",
            "1",
            "--random-state",
            "6",
            "--format",
            "csv",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.endswith("\n")
    assert not out.endswith("\n\n")
