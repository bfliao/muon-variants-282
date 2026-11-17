"""Tests for the Runner class in muonarch/runner.py (execution, sharding,
dry-run, resume). The identity layer is covered by test_runner_identity.py.

Real training NEVER runs: train.run is monkeypatched with a recorder that
plants fake result JSONs — exactly what a completed run leaves behind.
"""

from __future__ import annotations

import json
import math

import pytest

from muonarch import runner, studies, train


def plant(root, cfg, loss=1.0, status="completed"):
    p = runner.result_path(root, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "hash": runner.config_hash(cfg),
        "config": cfg,
        "status": status,
        "final_val_loss": None if status == "diverged" else loss,
    }))


@pytest.fixture
def fake_train(monkeypatch):
    """train.run replacement: records calls, writes a valid fake result."""
    calls = []

    def fake_run(cfg, root, device=None, data=None):
        plant(root, cfg, 1.0)
        calls.append(dict(cfg))
        return json.loads(runner.result_path(root, cfg).read_text())

    monkeypatch.setattr(train, "run", fake_run)
    return calls


# ---------------------------------------------------------------- execution

def test_run_study_completes_and_is_idempotent(tmp_path, fake_train):
    r = runner.Runner(tmp_path)
    n = r.run_study("mlp128")
    assert n == len(fake_train) == 101   # the no-change/no-edge scenario
    # every phase complete, nothing pending
    for phase, ready, cfgs, pending, mine in r._survey("mlp128"):
        assert ready and not pending, phase.name
    # idempotent: a second call executes nothing
    assert r.run_study("mlp128") == 0
    assert len(fake_train) == 101


def test_skip_if_done(tmp_path, fake_train):
    pilots = studies.pilot_configs("mlp128")
    for cfg in pilots[:2]:
        plant(tmp_path, cfg, 1.0)
    runner.Runner(tmp_path).run_study("mlp128")
    done_before = {runner.config_hash(c) for c in pilots[:2]}
    assert done_before.isdisjoint({runner.config_hash(c) for c in fake_train})


def test_diverged_results_are_done(tmp_path, fake_train):
    """A diverged run is a COMPLETED run: it must not re-run."""
    pilots = studies.pilot_configs("mlp128")
    plant(tmp_path, pilots[0], status="diverged")
    runner.Runner(tmp_path).run_study("mlp128")
    assert runner.config_hash(pilots[0]) not in {
        runner.config_hash(c) for c in fake_train}


def test_killed_run_resume(tmp_path, fake_train):
    """A crash leaves a missing result; the next run_study reruns exactly
    that config."""
    r = runner.Runner(tmp_path)
    r.run_study("mlp128")
    victim = next(c for c in fake_train if c["phase"] == "fresh_seed")
    runner.result_path(tmp_path, victim).unlink()
    fake_train.clear()
    r.run_study("mlp128")
    assert [runner.config_hash(c) for c in fake_train] == [
        runner.config_hash(victim)]


def test_corrupt_result_repaired(tmp_path, fake_train):
    """A corrupt (unparseable) result file counts as pending: the runner
    hands the config back to train.run, which repairs it (coordinator
    decision after the train.py re-review)."""
    r = runner.Runner(tmp_path)
    r.run_study("mlp128")
    victim = fake_train[10]
    path = runner.result_path(tmp_path, victim)
    path.write_text("{ not json !!")
    assert runner.is_done(tmp_path, victim)            # bare existence: True
    assert not runner.result_valid(tmp_path, victim)   # but not valid
    fake_train.clear()
    r.run_study("mlp128")
    assert runner.config_hash(victim) in {
        runner.config_hash(c) for c in fake_train}
    assert runner.result_valid(tmp_path, victim)       # repaired


def test_keyboard_interrupt_propagates_and_resumes(tmp_path, monkeypatch):
    calls = []

    def fake_run(cfg, root, device=None, data=None):
        if len(calls) == 5:
            raise KeyboardInterrupt
        plant(root, cfg, 1.0)
        calls.append(cfg)
        return {}

    monkeypatch.setattr(train, "run", fake_run)
    r = runner.Runner(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        r.run_study("mlp128")
    assert len(calls) == 5
    # resume: the interrupted (unwritten) run is simply re-run

    def fake_run2(cfg, root, device=None, data=None):
        plant(root, cfg, 1.0)
        calls.append(cfg)
        return {}

    monkeypatch.setattr(train, "run", fake_run2)
    r.run_study("mlp128")
    assert len(calls) == 101


# ---------------------------------------------------------------- sharding

def test_shard_partition_covers_and_is_disjoint(tmp_path, monkeypatch):
    by_shard = {0: [], 1: []}
    current = {"i": 0}

    def fake_run(cfg, root, device=None, data=None):
        plant(root, cfg, 1.0)
        by_shard[current["i"]].append(dict(cfg))
        return {}

    monkeypatch.setattr(train, "run", fake_run)
    r0 = runner.Runner(tmp_path, laptop_index=0, n_laptops=2)
    r1 = runner.Runner(tmp_path, laptop_index=1, n_laptops=2)
    for _ in range(100):
        current["i"] = 0
        n0 = r0.run_study("mlp128")
        current["i"] = 1
        n1 = r1.run_study("mlp128")
        if n0 == 0 and n1 == 0:
            break
    h0 = {runner.config_hash(c) for c in by_shard[0]}
    h1 = {runner.config_hash(c) for c in by_shard[1]}
    assert h0 and h1 and h0.isdisjoint(h1)
    assert len(h0 | h1) == 101                      # together: the full study
    for i, cfgs in by_shard.items():
        for c in cfgs:
            assert int(runner._shard_key(c), 16) % 2 == i


def test_branch_jobs_shard_with_their_parent(tmp_path, fake_train):
    """budget_branch runs resume from the parent's LOCAL checkpoint, so they
    must always land on the parent constant run's shard."""
    runner.Runner(tmp_path).run_study("mlp128")
    branches = [c for c in fake_train if c["phase"] == "budget_branch"]
    assert len(branches) == 9
    for b in branches:
        assert runner._shard_key(b) == b["branch"]["parent_hash"]
        for n in (2, 3, 4):
            for i in range(n):
                r = runner.Runner(tmp_path, laptop_index=i, n_laptops=n)
                parent_owned = int(b["branch"]["parent_hash"], 16) % n == i
                assert r.owns(b) == parent_owned

    # and in a sharded execution the branch really runs on the parent's shard
    by_shard = {}
    for c in fake_train:
        by_shard.setdefault(int(runner._shard_key(c), 16) % 2, set()).add(
            runner.config_hash(c))
    for b in branches:
        shard = int(b["branch"]["parent_hash"], 16) % 2
        assert b["branch"]["parent_hash"] in by_shard[shard]


# ---------------------------------------------------------------- dry run

def test_dry_run_lists_jobs_and_runs_nothing(tmp_path, monkeypatch):
    def forbid(cfg, root, device=None, data=None):
        raise AssertionError("--dry-run must never execute a run")

    monkeypatch.setattr(train, "run", forbid)
    r = runner.Runner(tmp_path)
    out = r.dry_run("mlp128")
    assert "total: 3 job(s)" in out                 # pilot only, on an empty root
    for cfg in studies.pilot_configs("mlp128"):
        assert runner.config_hash(cfg) in out
    assert "pilot" in out and "adamw" in out
    # FLOPs estimates present and positive
    assert "est FLOPs" in out
    assert runner.estimate_flops(studies.pilot_configs("mlp128")[0]) > 0


def test_dry_run_after_pilot_shows_cd_sweeps(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("dry run executed a run")))
    for cfg in studies.pilot_configs("mlp128"):
        plant(tmp_path, cfg, 1.0)
    out = runner.Runner(tmp_path).dry_run("mlp128")
    # adamw lr sweep (6) + moonlight (6) + muonall (6) are generatable;
    # muon/manifold wait on adamw
    assert "total: 18 job(s)" in out
    assert "muon_cd_p1_lr" not in out and "manifold_cd_p1_lr" not in out


def test_estimate_flops_scales_with_steps_and_optimizer(tmp_path):
    [cfg] = [c for c in studies.pilot_configs("mlp128") if c["seed"] == 101]
    base = runner.estimate_flops(cfg)
    double = runner.estimate_flops(dict(cfg, budget_multiplier=2.0))
    assert double == 2 * base
    # a branch run only pays for its post-checkpoint steps
    br = dict(cfg, phase="budget_branch", budget_multiplier=0.125,
              branch={"parent_hash": "0" * 12, "checkpoint_step": 352})
    assert runner.estimate_flops(br) == pytest.approx(base * 88 / 3520)
    # manifold pays its msign upper bound (dual_cap + 2 calls/step)
    sv = studies.load_study("mlp128")
    man = studies.run_config(sv, "manifold", "smoke",
                             studies.cd_defaults(sv, "manifold"),
                             adamw_group=(1e-3, 0.1))
    muon = studies.run_config(sv, "muon", "smoke",
                              studies.cd_defaults(sv, "muon"),
                              adamw_group=(1e-3, 0.1))
    assert runner.estimate_flops(man) > runner.estimate_flops(muon)


# ---------------------------------------------------------------- status

def test_status_table(tmp_path, fake_train):
    r = runner.Runner(tmp_path)
    out = r.status("mlp128")
    assert "pilot" in out and "complete" in out
    assert out.splitlines()[2].startswith("pilot")
    r.run_study("mlp128")
    out = r.status("mlp128")
    for line in out.splitlines()[2:]:
        assert line.rstrip().endswith("yes")        # every phase complete


def test_runner_validates_shard(tmp_path):
    with pytest.raises(ValueError):
        runner.Runner(tmp_path, laptop_index=4, n_laptops=4)


def test_cli_parse_shard():
    assert runner._parse_shard("2/4") == (2, 4)
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        runner._parse_shard("nope")
