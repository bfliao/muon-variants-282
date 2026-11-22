"""Tests for scripts/timing.py (compute-estimate arithmetic) and
scripts/smoke.py (per-unit failure isolation). The scripts are imported from
their files; no real training runs — train.run is monkeypatched."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from muonarch import runner, studies, train

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CSV_FIELDS = ("hostname", "device", "study", "optimizer", "cap",
              "s_per_step", "timestamp")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({"hostname": "mac-test", "device": "cpu",
                        "timestamp": "t", **r})


# ---------------------------------------------------------------- timing.py

def test_timing_manifold_cap_scaling():
    """s(c) follows (c+1)/(c0+1) from one measurement, and the linear fit
    through two measurements."""
    timing = _load_script("timing")
    one = timing.Timings({("mlp128", "manifold", 5): 0.6})
    assert one.manifold_s("mlp128", 10) == pytest.approx(0.6 * 11 / 6)
    assert one.manifold_s("mlp128", 100) == pytest.approx(0.6 * 101 / 6)
    both = timing.Timings({("mlp128", "manifold", 5): 0.6,
                           ("mlp128", "manifold", 10): 1.1})
    assert both.manifold_s("mlp128", 5) == pytest.approx(0.6)
    assert both.manifold_s("mlp128", 10) == pytest.approx(1.1)
    assert both.manifold_s("mlp128", 100) == pytest.approx(0.6 + 0.5 * 95 / 5)


def test_timing_compute_estimate_from_synthetic_csv(tmp_path):
    """Hand-computed case. mlp128's planned manifold jobs execute 45,760
    steps at the scenario cap (10 cd trials + 3 fresh = 13 x 3,520) plus the
    3,520-step faithful rerun at cap 100. With manifold cap-5 s/step = 0.36
    and every other measurement 0 the study's hours are:

      cap 5:  (45760*0.36          + 3520*0.36*101/6) / 3600 = 10.5
      cap 10: (45760*0.36*11/6     + 3520*0.36*101/6) / 3600 = 14.3

    and the cap-10 total fits the 30-hour window, so the verdict must be
    'run Manifold Muon at dual-step cap 10'.
    """
    timing = _load_script("timing")
    rows = []
    for study in studies.STUDIES:
        for opt in ("adamw", "muon", "moonlight", "muonall", "lionmuon"):
            rows.append({"study": study, "optimizer": opt, "cap": "",
                         "s_per_step": 0.0})
        rows.append({"study": study, "optimizer": "manifold", "cap": 5,
                     "s_per_step": 0.36 if study == "mlp128" else 0.0})
    csv_path = tmp_path / "hardware-results.csv"
    write_csv(csv_path, rows)
    out = tmp_path / "compute_estimate.md"
    rc = timing.main(["--csv", str(csv_path), "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    # the hand-computed row, with the (c+1)/(c0+1) scaling for caps 10 & 100
    assert "| mlp128 | 10.5 | 10.5 | 14.3 | 14.3 |" in text
    for study in ("mlp256", "mlp512", "dense", "moe"):
        assert f"| {study} | 0.0 | 0.0 | 0.0 | 0.0 |" in text
    assert "| **total** | **10.5** |" in text and "**14.3**" in text
    assert "run Manifold Muon at dual-step cap 10" in text
    assert "WARNING" not in text                     # every cell measured
    # independent cross-check of the arithmetic through study_hours
    t = timing.Timings(timing.read_measurements(csv_path, None))
    h5, m5 = timing.study_hours("mlp128", t, 5)
    assert h5 == m5 == pytest.approx(
        (45760 * 0.36 + 3520 * 0.36 * 101 / 6) / 3600)
    h10, m10 = timing.study_hours("mlp128", t, 10)
    assert h10 == m10 == pytest.approx(
        (45760 * 0.36 * 11 / 6 + 3520 * 0.36 * 101 / 6) / 3600)


def test_timing_errors_without_measurements(tmp_path):
    timing = _load_script("timing")
    rc = timing.main(["--csv", str(tmp_path / "none.csv"),
                      "--out", str(tmp_path / "out.md")])
    assert rc == 1
    assert not (tmp_path / "out.md").exists()


# ---------------------------------------------------------------- smoke.py

def test_smoke_continues_past_failed_unit(tmp_path, monkeypatch, capsys):
    """One optimizer's smoke unit blows up; the remaining units must still
    run (and be timed) and the script must exit nonzero."""
    smoke = _load_script("smoke")
    executed = []

    def fake_run(cfg, root, device=None, data=None):
        if cfg["optimizer"] == "muon":
            raise RuntimeError("boom: diverged smoke unit")
        executed.append(dict(cfg))
        h = runner.config_hash(cfg)
        res_path = runner.result_path(root, cfg)
        res_path.parent.mkdir(parents=True, exist_ok=True)
        res_path.write_text(json.dumps({
            "hash": h, "config": cfg, "status": "completed",
            "final_val_loss": 1.0}))
        frac = {"adamw": 0.0, "muonall": 1.0}.get(cfg["optimizer"], 0.995)
        log_path = runner.log_path(root, cfg)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            f.write(json.dumps({"t": "start", "hash": h,
                                "fractions": {"muon": frac,
                                              "adamw": 1 - frac}}) + "\n")
            f.write(json.dumps({"t": "eval", "step": 50, "val_loss": 2.0}) + "\n")
            f.write(json.dumps({"t": "eval", "step": 200, "val_loss": 1.0}) + "\n")
        return {}

    monkeypatch.setattr(train, "run", fake_run)
    csv_path = tmp_path / "hw.csv"
    rc = smoke.main(["--study", "mlp128", "--root", str(tmp_path),
                     "--csv", str(csv_path), "--device", "cpu"])
    assert rc == 1                                    # the muon unit failed
    err = capsys.readouterr().err
    assert "FAILED mlp128/muon" in err and "1 unit(s) FAILED" in err
    # the units after the failed one still executed: adamw, moonlight,
    # manifold (cap 5), muonall, lionmuon, manifold (cap 10) -> 6 timed pairs
    with csv_path.open() as f:
        timed = list(csv.DictReader(f))
    assert [(r["optimizer"], r["cap"]) for r in timed] == [
        ("adamw", ""), ("moonlight", ""), ("manifold", "5"),
        ("muonall", ""), ("lionmuon", ""), ("manifold", "10")]
    assert all(r["study"] == "mlp128" for r in timed)
    # each unit ran a 20-step and a 200-step smoke config
    assert sorted({c["budget_base_steps"] for c in executed}) == [20, 200]
    assert all(c["phase"] == "smoke" and c["eval"]["every_steps"] == 50
               for c in executed)
    assert not any(c["optimizer"] == "muon" for c in executed)


def test_smoke_all_green_exits_zero(tmp_path, monkeypatch):
    smoke = _load_script("smoke")

    def fake_run(cfg, root, device=None, data=None):
        h = runner.config_hash(cfg)
        p = runner.result_path(root, cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"hash": h, "config": cfg,
                                 "status": "completed", "final_val_loss": 1.0}))
        frac = {"adamw": 0.0, "muonall": 1.0}.get(cfg["optimizer"], 0.995)
        lp = runner.log_path(root, cfg)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("\n".join([
            json.dumps({"t": "start", "hash": h,
                        "fractions": {"muon": frac, "adamw": 1 - frac}}),
            json.dumps({"t": "eval", "step": 50, "val_loss": 2.0}),
            json.dumps({"t": "eval", "step": 200, "val_loss": 1.0}),
        ]) + "\n")
        return {}

    monkeypatch.setattr(train, "run", fake_run)
    rc = smoke.main(["--study", "mlp128", "--root", str(tmp_path),
                     "--csv", str(tmp_path / "hw.csv"), "--device", "cpu",
                     "--optimizer", "manifold"])
    assert rc == 0
    with (tmp_path / "hw.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["optimizer"], r["cap"]) for r in rows] == [
        ("manifold", "5"), ("manifold", "10")]
