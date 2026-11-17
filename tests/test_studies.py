"""Tests for muonarch/studies.py — study definitions, grids, gating DAG.

No real training ever runs here: results are PLANTED as result JSONs in
tmp_path (the runner/train contract: a run is its results/<study>/<hash>.json),
and end-to-end drives go through a monkeypatched train.run that records calls
and writes fake results.
"""

from __future__ import annotations

import json
import math

import pytest

from muonarch import runner, studies, train
from muonarch.schedule import checkpoint_step


# ---------------------------------------------------------------- helpers

def plant(root, cfg, loss=1.0, status="completed"):
    """Write a fake result JSON for cfg (what a completed run leaves)."""
    p = runner.result_path(root, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "hash": runner.config_hash(cfg),
        "config": cfg,
        "status": status,
        "final_val_loss": None if status == "diverged" else loss,
    }))


def view_of(root):
    return studies.ResultsView(root)


def complete_cd(root, study, opt, loss_fn=lambda cfg: 1.0):
    """Plant results sweep by sweep until the optimizer's coordinate descent
    is done; returns the final replay."""
    for _ in range(200):
        rep = studies.cd_replay(study, opt, view_of(root))
        assert not rep.blocked, "cd blocked (pilot / adamw gate missing)"
        if rep.done:
            return rep
        planted = False
        for ss in rep.sweeps.values():
            for cfg in ss.configs + ss.edge_configs:
                if not runner.is_done(root, cfg):
                    plant(root, cfg, loss_fn(cfg))
                    planted = True
        assert planted, "cd neither done nor plantable"
    raise AssertionError("cd did not converge in 200 rounds")


def plant_pilot(root, study, losses=(1.0, 1.01, 1.02)):
    for cfg, loss in zip(studies.pilot_configs(study), losses):
        plant(root, cfg, loss)


@pytest.fixture(autouse=True)
def _reset_seed_count():
    yield
    studies.set_final_seed_count(3)


@pytest.fixture
def driver(monkeypatch):
    """Monkeypatched train.run: records every call and writes a result whose
    loss comes from a per-test loss function (never real training)."""
    calls = []

    def install(loss_fn=lambda cfg: 1.0):
        def fake_run(cfg, root, device=None, data=None):
            plant(root, cfg, loss_fn(cfg))
            calls.append(cfg)
            return json.loads(runner.result_path(root, cfg).read_text())

        monkeypatch.setattr(train, "run", fake_run)
        return calls

    return install


# ---------------------------------------------------------------- constants

def test_studies_and_optimizers():
    assert studies.STUDIES == ("mlp128", "mlp256", "mlp512", "dense", "moe")
    assert studies.OPTIMIZERS == ("adamw", "muon", "moonlight", "manifold", "muonall")
    assert studies.PHASE1_STUDIES == ("mlp128", "dense")
    assert studies.PHASE2_DONOR == {"mlp256": "mlp128", "mlp512": "mlp128",
                                    "moe": "dense"}


def test_load_study_constants():
    mlp = studies.load_study("mlp128")
    assert (mlp["budget_base_steps"], mlp["batch"], mlp["seq_len"]) == (3520, 256, None)
    assert mlp["eval"]["every_steps"] == 176
    assert mlp["data"]["version"].startswith("cifar5m-v1-")
    lm = studies.load_study("dense")
    assert (lm["budget_base_steps"], lm["batch"], lm["seq_len"]) == (2441, 32, 256)
    assert lm["eval"] == {"every_steps": 100, "val_small_tokens": 500000,
                          "val_final": "full"}
    assert lm["data"]["version"].startswith("fineweb-edu-10bt-v1-")
    moe = studies.load_study("moe")
    assert moe["router_aux_loss_coef"] == 0.01
    # literature default lr centres (DEVIATIONS.md #8) and wds (answer 3)
    for sv in (mlp, lm, moe):
        o = sv["optimizers"]
        assert (o["adamw"]["lr"], o["muon"]["lr"], o["moonlight"]["lr"],
                o["muonall"]["lr"], o["manifold"]["lr"]) == (1e-3, 0.02, 1e-3, 0.02, 0.1)
        assert (o["adamw"]["wd"], o["moonlight"]["wd"], o["muon"]["wd"],
                o["muonall"]["wd"], o["manifold"]["wd"]) == (0.1, 0.1, 0.01, 0.01, None)


# ---------------------------------------------------------------- pilot

def test_pilot_configs_seeds_and_studies():
    # mlp256 gained a pilot with the full re-tune (DEVIATIONS.md #12)
    for study in ("mlp128", "mlp256", "mlp512", "dense"):
        cfgs = studies.pilot_configs(study)
        assert [c["seed"] for c in cfgs] == [101, 102, 103]
        assert all(c["phase"] == "pilot" and c["optimizer"] == "adamw"
                   for c in cfgs)
    assert studies.pilot_configs("moe") == []


def test_pilot_gates_cd(tmp_path):
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    assert rep.blocked and not rep.done
    assert all(ss.configs == [] for ss in rep.sweeps.values())
    plant_pilot(tmp_path, "mlp128")
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    assert not rep.blocked
    assert len(rep.sweeps["p1_lr"].configs) == 6


def test_pilot_seed_std(tmp_path):
    assert studies.pilot_seed_std("mlp128", view_of(tmp_path)) is None
    plant_pilot(tmp_path, "mlp128", (1.0, 1.01, 1.02))
    std = studies.pilot_seed_std("mlp128", view_of(tmp_path))
    assert std == pytest.approx(0.01)


# ---------------------------------------------------------------- DAG gating

def _dag_by_name(study, root):
    return {p.name: p for p in studies.build_dag(study, view_of(root))}


def test_dag_adamw_first_muon_manifold_gate(tmp_path):
    dag = _dag_by_name("mlp128", tmp_path)
    assert dag["adamw_cd_p1_lr"].needs == ("pilot",)
    # Muon and Manifold wait on AdamW's whole Phase-I chain
    assert set(dag["muon_cd_p1_lr"].needs) == {"pilot", "adamw_cd_p2_clip"}
    assert set(dag["manifold_cd_p1_lr"].needs) == {"pilot", "adamw_cd_p2_clip"}
    # Moonlight and MuonAll gate on nothing but the pilot
    assert dag["moonlight_cd_p1_lr"].needs == ("pilot",)
    assert dag["muonall_cd_p1_lr"].needs == ("pilot",)
    # budget curve needs adamw fresh; branches/halflr need the constant run
    assert dag["budget_constant"].needs == ("adamw_fresh",)
    assert dag["budget_branch"].needs == ("budget_constant",)
    assert dag["budget_halflr"].needs == ("budget_constant",)
    assert dag["init_ablation"].needs == ("muon_fresh",)
    assert dag["manifold_faithful"].needs == ("manifold_fresh",)


def test_muon_uses_adamw_phase1_winners(tmp_path):
    plant_pilot(tmp_path, "mlp128", (1.0, 1.01, 1.02))
    d = 1e-3
    # adamw lr sweep: 2x default wins decisively; everything else ties
    complete_cd(tmp_path, "mlp128", "adamw",
                lambda c: 0.5 if math.isclose(c["lr"], 2 * d) else 1.0)
    vals = studies.phase1_values("mlp128", "adamw", view_of(tmp_path))
    assert vals["lr"] == pytest.approx(2 * d)
    rep = studies.cd_replay("mlp128", "muon", view_of(tmp_path))
    assert not rep.blocked
    for cfg in rep.sweeps["p1_lr"].configs:
        assert cfg["adamw_group_lr"] == pytest.approx(2 * d)
        assert cfg["adamw_group_wd"] == pytest.approx(0.1)


def test_muon_blocked_until_adamw_done(tmp_path):
    plant_pilot(tmp_path, "mlp128")
    rep = studies.cd_replay("mlp128", "muon", view_of(tmp_path))
    assert rep.blocked
    rep = studies.cd_replay("mlp128", "moonlight", view_of(tmp_path))
    assert not rep.blocked  # moonlight does not wait on adamw


def test_phase2_donor_gates_stage_a(tmp_path):
    dag = _dag_by_name("mlp256", tmp_path)
    phase = dag["adamw_stage_a"]
    assert not phase.is_ready()          # donor (mlp128) Phase I incomplete
    assert phase.configs() == []
    # muon additionally waits on this study's adamw stages
    assert dag["muon_stage_a"].needs == ("adamw_edge_b",)
    assert dag["moonlight_stage_a"].needs == ()
    assert dag["muonall_stage_a"].needs == ()


# ---------------------------------------------------------------- cd sweeps

def test_cd_sweep_order_grids_and_no_warmup(tmp_path):
    assert studies.CD_ORDER == ("lr", "wd", "momentum", "beta2", "clip")
    assert studies.cd_hypers("adamw") == ("lr", "wd", "momentum", "beta2", "clip")
    assert studies.cd_hypers("muon") == ("lr", "wd", "momentum", "clip")
    assert studies.cd_hypers("manifold") == ("lr", "momentum", "clip")  # no wd
    dag_names = [p.name for p in studies.build_dag("mlp128", view_of(tmp_path))]
    assert not any("warmup" in n for n in dag_names)  # DEVIATIONS.md #6

    plant_pilot(tmp_path, "mlp128")
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    lrs = sorted(c["lr"] for c in rep.sweeps["p1_lr"].configs)
    assert len(lrs) == 6
    assert lrs[-1] / lrs[0] == pytest.approx(32.0)        # spans 32x
    assert any(math.isclose(lr, 1e-3) for lr in lrs)       # centred on default
    for a, b in zip(lrs, lrs[1:]):
        assert b / a == pytest.approx(2.0)                 # factor-2 grid
    # warmup fixed at 5% in every generated config
    for ss in rep.sweeps.values():
        for cfg in ss.configs:
            assert cfg["schedule"]["warmup_frac_of_1x"] == 0.05

    # wd sweep {0, 0.3d, d, 3d}; momentum = beta1 grid for adamw
    for c in rep.sweeps["p1_lr"].configs:
        plant(tmp_path, c)
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    wds = sorted(c["wd"] for c in rep.sweeps["p1_wd"].configs)
    assert wds == pytest.approx([0.0, 0.03, 0.1, 0.3])
    for c in rep.sweeps["p1_wd"].configs:
        plant(tmp_path, c)
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    b1s = sorted(c["adamw_betas"][0] for c in rep.sweeps["p1_momentum"].configs)
    assert b1s == [0.8, 0.9, 0.95]
    assert {c["cd_sweep"] for c in rep.sweeps["p1_momentum"].configs} == {"beta1"}
    for c in rep.sweeps["p1_momentum"].configs:
        plant(tmp_path, c)
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    b2s = sorted(c["adamw_betas"][1] for c in rep.sweeps["p1_beta2"].configs)
    assert b2s == [0.95, 0.99, 0.999]
    for c in rep.sweeps["p1_beta2"].configs:
        plant(tmp_path, c)
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    clips = {c["grad_clip"] for c in rep.sweeps["p1_clip"].configs}
    assert clips == {None, 0.5, 1.0}


def test_muon_momentum_grid(tmp_path):
    plant_pilot(tmp_path, "mlp128")
    complete_cd(tmp_path, "mlp128", "adamw")
    rep = studies.cd_replay("mlp128", "muon", view_of(tmp_path))
    for c in rep.sweeps["p1_lr"].configs:
        plant(tmp_path, c)
    for c in studies.cd_replay("mlp128", "muon", view_of(tmp_path)) \
            .sweeps["p1_wd"].configs:
        plant(tmp_path, c)
    rep = studies.cd_replay("mlp128", "muon", view_of(tmp_path))
    moms = sorted(c["momentum"] for c in rep.sweeps["p1_momentum"].configs)
    assert moms == [0.9, 0.95, 0.98]
    assert {c["cd_sweep"] for c in rep.sweeps["p1_momentum"].configs} == {"momentum"}


def test_cd_sweep_field_present_and_not_hashed(tmp_path):
    plant_pilot(tmp_path, "mlp128")
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    cfg = rep.sweeps["p1_lr"].configs[0]
    assert cfg["cd_sweep"] == "lr"
    bare = {k: v for k, v in cfg.items() if k != "cd_sweep"}
    # NOT hashed: a revisited identical config reuses its result
    assert runner.config_hash(cfg) == runner.config_hash(bare)


def test_duplicate_cells_share_hashes_across_sweeps(tmp_path):
    """The incumbent re-appears in every later sweep and must hash to the
    SAME run (cd_sweep unhashed => dedup => its result is reused)."""
    plant_pilot(tmp_path, "mlp128")
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    lr_hashes = {runner.config_hash(c) for c in rep.sweeps["p1_lr"].configs}
    for c in rep.sweeps["p1_lr"].configs:
        plant(tmp_path, c)
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    wd_hashes = {runner.config_hash(c) for c in rep.sweeps["p1_wd"].configs}
    assert len(lr_hashes & wd_hashes) == 1  # the incumbent (lr=d, wd=d) cell


# ---------------------------------------------------------------- acceptance

def test_cd_acceptance_threshold_arithmetic(tmp_path):
    """Accept only if the winner improves on the incumbent by MORE than
    2 x pilot seed std (std here: 0.01 => threshold 0.02)."""
    d = 1e-3

    def run_case(improvement):
        import shutil
        root = tmp_path / f"case_{improvement}"
        root.mkdir()
        plant_pilot(root, "mlp128", (1.0, 1.01, 1.02))
        complete_cd(root, "mlp128", "adamw",
                    lambda c: 1.0 - improvement
                    if math.isclose(c["lr"], 2 * d) else 1.0)
        return studies.phase1_values("mlp128", "adamw", view_of(root))["lr"]

    assert run_case(0.019) == pytest.approx(d)       # below 2*std: rejected
    assert run_case(0.021) == pytest.approx(2 * d)   # above 2*std: accepted


def test_cd_two_passes_and_early_stop(tmp_path):
    plant_pilot(tmp_path, "mlp128", (1.0, 1.01, 1.02))
    # all ties: pass 1 changes nothing -> pass 2 skipped entirely
    rep = complete_cd(tmp_path, "mlp128", "adamw")
    assert rep.done
    for key, ss in rep.sweeps.items():
        if key.startswith("p2_"):
            assert ss.configs == [] and ss.complete
    n_p1 = sum(len(ss.configs) for ss in rep.sweeps.values())
    assert n_p1 == 6 + 4 + 3 + 3 + 3

    # an accepted pass-1 change -> pass 2 runs, lr grid recentred
    root2 = tmp_path / "changed"
    root2.mkdir()
    plant_pilot(root2, "mlp128", (1.0, 1.01, 1.02))
    d = 1e-3
    rep = complete_cd(root2, "mlp128", "adamw",
                      lambda c: 0.5 if math.isclose(c["lr"], 2 * d) else 1.0)
    assert rep.done
    p2_lrs = sorted(c["lr"] for c in rep.sweeps["p2_lr"].configs)
    assert len(p2_lrs) == 6
    assert any(math.isclose(lr, 2 * d) for lr in p2_lrs)       # new centre
    assert p2_lrs[-1] / p2_lrs[0] == pytest.approx(32.0)


# ---------------------------------------------------------------- edge rule

def test_edge_rule_lr_only_capped_and_on_winners_only(tmp_path):
    plant_pilot(tmp_path, "mlp128", (1.0, 1.01, 1.02))
    d = 1e-3
    lo = d / 8  # the grid's low edge (k = -3)

    # interior winner: no extensions
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    for c in rep.sweeps["p1_lr"].configs:
        plant(tmp_path, c, 0.5 if math.isclose(c["lr"], d / 2) else 1.0)
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    assert rep.sweeps["p1_lr"].edge_configs == []
    assert rep.sweeps["p1_lr"].edge_complete

    # edge winner: extensions appear one at a time, capped at 2
    root2 = tmp_path / "edge"
    root2.mkdir()
    plant_pilot(root2, "mlp128", (1.0, 1.01, 1.02))
    rep = studies.cd_replay("mlp128", "adamw", view_of(root2))
    for c in rep.sweeps["p1_lr"].configs:
        plant(root2, c, 0.5 if math.isclose(c["lr"], lo) else 1.0)
    rep = studies.cd_replay("mlp128", "adamw", view_of(root2))
    ss = rep.sweeps["p1_lr"]
    assert len(ss.edge_configs) == 1 and not ss.edge_complete
    assert ss.edge_configs[0]["lr"] == pytest.approx(lo / 2)
    assert ss.edge_configs[0]["cd_sweep"] == "lr"        # part of the lr sweep
    plant(root2, ss.edge_configs[0], 0.4)                 # extension wins again
    rep = studies.cd_replay("mlp128", "adamw", view_of(root2))
    ss = rep.sweeps["p1_lr"]
    assert len(ss.edge_configs) == 2
    assert ss.edge_configs[1]["lr"] == pytest.approx(lo / 4)
    plant(root2, ss.edge_configs[1], 0.3)                 # wins again: capped
    rep = studies.cd_replay("mlp128", "adamw", view_of(root2))
    ss = rep.sweeps["p1_lr"]
    assert len(ss.edge_configs) == 2 and ss.edge_complete  # <= 2 beyond edge
    # the accepted lr is the best extension (finish the chain consistently)
    def by_lr(c):
        for lr, loss in ((lo, 0.5), (lo / 2, 0.4), (lo / 4, 0.3)):
            if math.isclose(c["lr"], lr):
                return loss
        return 1.0

    complete_cd(root2, "mlp128", "adamw", by_lr)
    vals = studies.phase1_values("mlp128", "adamw", view_of(root2))
    assert vals["lr"] == pytest.approx(lo / 4)


def test_edge_rule_never_fires_on_wd(tmp_path):
    """wd winner on its grid edge (3d) must not produce extensions — the
    edge rule is learning-rate-only (Gate-0 answer 5)."""
    plant_pilot(tmp_path, "mlp128", (1.0, 1.01, 1.02))
    rep = complete_cd(tmp_path, "mlp128", "adamw",
                      lambda c: 0.5 if (c["cd_sweep"] == "wd"
                                        and math.isclose(c["wd"], 0.3)) else 1.0)
    assert rep.done
    for key, ss in rep.sweeps.items():
        if "_wd" in key:
            assert ss.edge_configs == []
    wds = {c["wd"] for c in rep.sweeps["p1_wd"].configs}
    assert max(wds) == pytest.approx(0.3)   # grid never grew past 3d
    assert studies.phase1_values("mlp128", "adamw", view_of(tmp_path))["wd"] \
        == pytest.approx(0.3)


# ---------------------------------------------------------------- Phase II

def finish_phase1(root, study, loss_fn=lambda cfg: 1.0):
    plant_pilot(root, study)
    complete_cd(root, study, "adamw", loss_fn)
    for opt in ("muon", "moonlight", "manifold", "muonall"):
        complete_cd(root, study, opt, loss_fn)


def test_stage_a_grid_positions_shared_across_widths():
    for opt in studies.OPTIMIZERS:
        d = studies.load_study("mlp128")["optimizers"][opt]["lr"]
        ratios = {s: [lr / d for lr in studies.stage_a_grid(s, opt)]
                  for s in ("mlp256", "mlp512")}
        assert ratios["mlp256"] == pytest.approx(ratios["mlp512"])
        assert ratios["mlp256"] == pytest.approx([2.0 ** k for k in (-3, -2, -1, 0, 1, 2)])


def test_phase2_inherits_donor_non_lr_wd(tmp_path):
    plant_pilot(tmp_path, "mlp128", (1.0, 1.01, 1.02))
    complete_cd(tmp_path, "mlp128", "adamw")
    # muon donor cd accepts momentum 0.9 (big win), everything else ties
    complete_cd(tmp_path, "mlp128", "muon",
                lambda c: 0.5 if c["momentum"] == 0.9 else 1.0)
    vals = studies.inherited_values("mlp256", "muon", view_of(tmp_path))
    assert vals is not None
    assert vals["momentum"] == 0.9                 # inherited from donor
    assert vals["lr"] == pytest.approx(0.02)       # lr reset to the default
    assert vals["wd"] == pytest.approx(0.01)       # wd reset to the default
    assert studies.inherited_values("mlp256", "moonlight", view_of(tmp_path)) \
        is None                                    # that donor chain not done


def _finish_stage_a(root, study, opt, loss_fn=lambda c: 1.0):
    for _ in range(10):
        view = view_of(root)
        st = studies._stage_cells(study, opt, view)
        assert st is not None
        cells, edge, edge_complete, *_ = st
        todo = [c for _, c in cells] + edge
        pending = [c for c in todo if not runner.is_done(root, c)]
        if not pending and edge_complete:
            return
        for c in pending:
            plant(root, c, loss_fn(c))
    raise AssertionError("stage A did not finish")


def test_stage_b_two_best_lrs_times_wd_set(tmp_path):
    finish_phase1(tmp_path, "mlp128")
    d_lr, d_wd = 1e-3, 0.1
    # stage A for adamw in mlp256: distinct losses, best lrs d/2 then d
    _finish_stage_a(tmp_path, "mlp256", "adamw",
                    lambda c: {0.5: 0.4, 1.0: 0.5}.get(
                        round(c["lr"] / d_lr, 4), 1.0))
    sb = studies._stage_b_cells("mlp256", "adamw", view_of(tmp_path))
    assert sb is not None
    cfgs = [c for _, c in sb[0]]
    assert len(cfgs) == 6 and all(c["phase"] == "stage_b" for c in cfgs)
    assert sorted({round(c["lr"] / d_lr, 4) for c in cfgs}) == [0.5, 1.0]
    wds = sorted({c["wd"] for c in cfgs})
    assert wds == pytest.approx([0.0, 0.3 * d_wd, 3 * d_wd])   # d NOT re-run


def test_manifold_stage_b_quarter_octave(tmp_path):
    finish_phase1(tmp_path, "mlp128")
    _finish_stage_a(tmp_path, "mlp256", "adamw")
    for _, c in studies._stage_b_cells("mlp256", "adamw", view_of(tmp_path))[0]:
        plant(tmp_path, c)
    _finish_stage_a(tmp_path, "mlp256", "manifold",
                    lambda c: 0.5 if math.isclose(c["lr"], 0.05) else 1.0)
    sb = studies._stage_b_cells("mlp256", "manifold", view_of(tmp_path))
    assert sb is not None
    cfgs = [c for _, c in sb[0]]
    ks = sorted(math.log2(c["lr"] / 0.05) for c in cfgs)
    assert ks == pytest.approx([-0.75, -0.5, -0.25, 0.25, 0.5, 0.75])
    assert all(c["wd"] is None for c in cfgs)     # manifold never gets wd


def test_phase2_selection_min_seed0_loss(tmp_path):
    finish_phase1(tmp_path, "mlp128")
    d_wd = 0.1
    _finish_stage_a(tmp_path, "mlp256", "adamw",
                    lambda c: {0.5: 0.95, 1.0: 0.9}.get(
                        round(c["lr"] / 1e-3, 4), 1.0))
    # a stage-B cell beats every stage-A cell
    for _, c in studies._stage_b_cells("mlp256", "adamw", view_of(tmp_path))[0]:
        plant(tmp_path, c,
              0.4 if (math.isclose(c["lr"], 5e-4)
                      and math.isclose(c["wd"], 0.3 * d_wd)) else 0.95)
    # mlp256 is a FULL_RETUNE study (DEVIATIONS.md #12): its inherited
    # Phase-II track still selects the min seed-0 loss trial ...
    best = studies._phase2_best_trial("mlp256", "adamw", view_of(tmp_path))
    assert best["lr"] == pytest.approx(5e-4)
    assert best["wd"] == pytest.approx(0.3 * d_wd)
    # ... but the headline (best_config / fresh seeds) is the CD track, which
    # has not run here (no mlp256 pilot): nothing is selected yet
    assert studies.best_config("mlp256", "adamw", view_of(tmp_path)) is None
    assert studies.fresh_seed_configs("mlp256", "adamw", view_of(tmp_path)) == []


# ---------------------------------------------------------------- fresh seeds

def test_fresh_seed_count_setting(tmp_path):
    plant_pilot(tmp_path, "mlp128", (1.0, 1.01, 1.02))
    complete_cd(tmp_path, "mlp128", "adamw")
    fresh = studies.fresh_seed_configs("mlp128", "adamw", view_of(tmp_path))
    assert [c["seed"] for c in fresh] == [1, 2, 3]
    studies.set_final_seed_count(5)
    fresh = studies.fresh_seed_configs("mlp128", "adamw", view_of(tmp_path))
    assert [c["seed"] for c in fresh] == [1, 2, 3, 4, 5]
    # the budget curve follows the same count (Gate-0 answer 6)
    consts = studies.budget_constant_configs("mlp128", view_of(tmp_path))
    assert [c["seed"] for c in consts] == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        studies.set_final_seed_count(4)


# ---------------------------------------------------------------- budget curve

def test_budget_constant_and_branch_configs(tmp_path):
    plant_pilot(tmp_path, "mlp128", (1.0, 1.01, 1.02))
    complete_cd(tmp_path, "mlp128", "adamw")
    view = view_of(tmp_path)
    consts = studies.budget_constant_configs("mlp128", view)
    assert [c["seed"] for c in consts] == [1, 2, 3]
    assert all(c["phase"] == "budget_constant"
               and c["budget_multiplier"] == 2.0 and c["branch"] is None
               for c in consts)
    branches = studies.budget_branch_configs("mlp128", view)
    assert len(branches) == 9
    by_parent = {runner.config_hash(c): c for c in consts}
    for b in branches:
        assert b["phase"] == "budget_branch"
        parent = by_parent[b["branch"]["parent_hash"]]
        assert parent["seed"] == b["seed"]            # branch pairs its seed
        m = b["budget_multiplier"]
        assert m in (0.125, 0.25, 0.5)
        assert b["branch"]["checkpoint_step"] == checkpoint_step(3520, m)
    mults = sorted({b["branch"]["checkpoint_step"] for b in branches})
    assert mults == [352, 704, 1408]


def test_halflr_promotion_both_ways(tmp_path):
    plant_pilot(tmp_path, "mlp128", (1.0, 1.01, 1.02))
    complete_cd(tmp_path, "mlp128", "adamw")
    view = view_of(tmp_path)
    consts = {c["seed"]: c for c in studies.budget_constant_configs("mlp128", view)}
    halfs = studies.budget_halflr_configs("mlp128", view)
    assert [c["seed"] for c in halfs] == [1]           # seed 1 ALWAYS
    half1 = halfs[0]
    assert half1["phase"] == "budget_halflr"
    assert half1["lr"] == pytest.approx(consts[1]["lr"] / 2)
    assert half1["wd"] == consts[1]["wd"]              # wd unchanged
    assert half1["budget_multiplier"] == 2.0

    # half-LR loses to the branched 2x -> seed 1 only
    plant(tmp_path, consts[1], 2.0)
    plant(tmp_path, half1, 2.1)
    assert [c["seed"] for c in
            studies.budget_halflr_configs("mlp128", view_of(tmp_path))] == [1]
    # half-LR wins -> promoted to every curve seed
    plant(tmp_path, half1, 1.9)
    halfs = studies.budget_halflr_configs("mlp128", view_of(tmp_path))
    assert [c["seed"] for c in halfs] == [1, 2, 3]
    assert all(c["lr"] == pytest.approx(consts[1]["lr"] / 2) for c in halfs)


# ---------------------------------------------------------------- ablations

def test_init_ablation_configs(tmp_path):
    finish_phase1(tmp_path, "mlp128")
    cfgs = studies.init_ablation_configs("mlp128", view_of(tmp_path))
    assert [c["seed"] for c in cfgs] == [1, 2, 3]
    assert all(c["phase"] == "init_ablation" and c["optimizer"] == "muon"
               and c["init_projection"] is True for c in cfgs)


def test_faithful_three_configs_official_settings(tmp_path):
    finish_phase1(tmp_path, "mlp128")
    cfgs = {c["optimizer"]: c
            for c in studies.faithful_configs("mlp128", view_of(tmp_path))}
    assert set(cfgs) == {"muon", "muonall", "manifold"}
    assert all(c["phase"] == "faithful" and c["seed"] == 1
               for c in cfgs.values())
    assert cfgs["muon"]["ns_dtype"] == "bf16" and cfgs["muon"]["ns_steps"] == 5
    assert cfgs["muonall"]["ns_dtype"] == "bf16" and cfgs["muonall"]["ns_steps"] == 6
    m = cfgs["manifold"]["manifold"]
    assert m["dual_cap"] == 100
    assert m["tol"] == 1e-6 and m["dual_lr"] == 0.01 and m["msign_steps"] == 10
    assert cfgs["manifold"]["init_projection"] is True


# ---------------------------------------------------------------- hygiene

def test_every_generated_config_hashes_cleanly(tmp_path, driver):
    """Drive mlp128 end-to-end through the runner with fake training: every
    generated config must carry every HASHED_FIELD and hash cleanly."""
    calls = driver()
    runner.Runner(tmp_path).run_study("mlp128")
    assert len(calls) > 90
    for cfg in calls:
        assert set(runner.HASHED_FIELDS) <= set(cfg)
        h = runner.config_hash(cfg)
        assert len(h) == 12
        if cfg["phase"] == "cd":
            assert "cd_sweep" in cfg
        if cfg["optimizer"] in ("muon", "manifold"):
            assert cfg["adamw_group_lr"] is not None
            assert cfg["adamw_group_wd"] is not None
        else:
            assert cfg["adamw_group_lr"] is None


def test_planned_job_counts_no_edge_scenario():
    """Gate-report numbers: the planning enumeration under the documented
    no-change/no-edge scenario (matches the fake end-to-end run)."""
    counts = {s: len(studies.planned_jobs(s)) for s in studies.STUDIES}
    # mlp256/mlp512 run both tracks (DEVIATIONS.md #12): +64 cd jobs each,
    # +3 pilot runs at mlp256
    assert counts == {"mlp128": 101, "mlp256": 161, "mlp512": 161,
                      "dense": 101, "moe": 94}
    assert sum(counts.values()) == 618


# ---------------------------------------------------------------- exclusion

@pytest.fixture
def moe_without_manifold(tmp_path, monkeypatch):
    """A config dir where moe excludes manifold (the PLAN cut-order lever),
    installed as the module default so the Runner picks it up too."""
    import shutil
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    for s in studies.STUDIES:
        shutil.copy(studies._CONFIG_DIR / f"{s}.yaml", cfg_dir / f"{s}.yaml")
    text = (cfg_dir / "moe.yaml").read_text()
    text = text.replace("exclude_optimizers: []", "exclude_optimizers: [manifold]")
    (cfg_dir / "moe.yaml").write_text(text)
    monkeypatch.setattr(studies, "_CONFIG_DIR", cfg_dir)
    return cfg_dir


def test_exclude_optimizers_drops_only_that_chain(tmp_path, moe_without_manifold):
    with_m = {
        "manifold_stage_a", "manifold_edge", "manifold_stage_b",
        "manifold_edge_b", "manifold_fresh", "manifold_faithful",
    }
    names = {p.name for p in studies.build_dag("moe", view_of(tmp_path))}
    assert names.isdisjoint(with_m)
    # everything else is still there: vs an unexcluded same-shape study, the
    # difference is EXACTLY the manifold chain
    # (mlp256 also has the pilot and the CD track since DEVIATIONS.md #12;
    # compare against its Phase-II-shaped part)
    other = {p.name for p in studies.build_dag("mlp256", view_of(tmp_path))
             if "_cd_" not in p.name and p.name != "pilot"}
    assert other - names == with_m
    assert names <= other
    # planned enumeration honors the exclusion: 94 - 16 manifold jobs
    jobs = studies.planned_jobs("moe")
    assert len(jobs) == 78
    assert all(j["optimizer"] != "manifold" for j in jobs)


def test_exclude_optimizers_no_deadlock(tmp_path, moe_without_manifold, driver):
    """With manifold cut from moe, the study (and its donor) still runs to
    completion and schedules no manifold job in moe."""
    calls = driver()
    r = runner.Runner(tmp_path)
    r.run_study("dense")          # donor: full 101-run Phase-I study
    n = r.run_study("moe")
    moe_calls = [c for c in calls if c["study"] == "moe"]
    assert n == len(moe_calls) == 78
    assert all(c["optimizer"] != "manifold" for c in moe_calls)
    for phase, ready, cfgs, pending, mine in r._survey("moe"):
        assert ready and not pending, phase.name


def test_exclude_optimizers_validation(tmp_path):
    text = (studies._CONFIG_DIR / "moe.yaml").read_text()
    for sub, bad, match in (("a", "[adamw]", "adamw cannot be excluded"),
                            ("b", "[sgd]", "unknown")):
        cfg_dir = tmp_path / sub      # distinct dirs: load_study caches by dir
        cfg_dir.mkdir()
        (cfg_dir / "moe.yaml").write_text(
            text.replace("exclude_optimizers: []",
                         f"exclude_optimizers: {bad}"))
        with pytest.raises(ValueError, match=match):
            studies.load_study("moe", cfg_dir)


# ---------------------------------------------------------------- edge_trial

def test_edge_trial_field_logged_only(tmp_path):
    """Every edge-extension config carries edge_trial: true (logged-only,
    NOT hashed); ordinary grid cells never do."""
    plant_pilot(tmp_path, "mlp128", (1.0, 1.01, 1.02))
    d = 1e-3
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    for c in rep.sweeps["p1_lr"].configs:
        assert "edge_trial" not in c                 # ordinary cells: no field
        plant(tmp_path, c, 0.5 if math.isclose(c["lr"], d / 8) else 1.0)
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    ext = rep.sweeps["p1_lr"].edge_configs[0]
    assert ext["edge_trial"] is True
    bare = {k: v for k, v in ext.items() if k not in ("edge_trial", "cd_sweep")}
    assert runner.config_hash(ext) == runner.config_hash(bare)   # NOT hashed


def test_edge_trial_field_phase2(tmp_path):
    finish_phase1(tmp_path, "mlp128")
    d = 1e-3
    # stage-A winner at the HIGH grid edge (4d) -> a phase-II edge trial
    _finish_stage_a(tmp_path, "mlp256", "adamw",
                    lambda c: 0.5 if math.isclose(c["lr"], 4 * d)
                    else (0.6 if c.get("edge_trial") else 1.0))
    cells, edge, edge_complete, *_ = studies._stage_cells(
        "mlp256", "adamw", view_of(tmp_path))
    assert edge_complete and len(edge) == 1
    assert edge[0]["phase"] == "edge" and edge[0]["edge_trial"] is True
    assert all("edge_trial" not in c for _, c in cells)
    # a winner that happens to be an edge trial loses the marker on rerun
    fresh = studies.fresh_seed_configs("mlp128", "adamw", view_of(tmp_path))
    assert all("edge_trial" not in c for c in fresh)


# ---------------------------------------------------------------- trial counts

def test_trial_counts(tmp_path):
    plant_pilot(tmp_path, "mlp128")                    # pilots never count
    rep = studies.cd_replay("mlp128", "adamw", view_of(tmp_path))
    for c in rep.sweeps["p1_lr"].configs:              # 6 adamw cd trials
        plant(tmp_path, c)
    sv = studies.load_study("mlp128")
    muon_cd = studies.run_config(sv, "muon", "cd", studies.cd_defaults(sv, "muon"),
                                 adamw_group=(1e-3, 0.1), cd_sweep="lr")
    plant(tmp_path, muon_cd, 1.2)                      # 1 muon cd trial
    plant(tmp_path, studies.run_config(                # diverged trials count too
        sv, "muonall", "cd", studies.cd_defaults(sv, "muonall"), cd_sweep="lr"),
        status="diverged")
    plant(tmp_path, studies.run_config(                # fresh seeds never count
        sv, "moonlight", "fresh_seed", studies.cd_defaults(sv, "moonlight"),
        seed=1))
    counts = studies.trial_counts(view_of(tmp_path), "mlp128")
    assert counts == {"adamw": 6, "muon": 1, "moonlight": 0,
                      "manifold": 0, "muonall": 1}
    assert studies.trial_counts(view_of(tmp_path), "dense") == dict.fromkeys(
        studies.OPTIMIZERS, 0)


# ---------------------------------------------------------------- full re-tune
# MLP protocol change (Gate 3, DEVIATIONS.md #12): mlp256/mlp512 run the full
# Phase-I coordinate descent (headline) AND the inherited Phase-II track (H4).

def test_full_retune_constants_and_unchanged_studies():
    assert studies.FULL_RETUNE_STUDIES == ("mlp256", "mlp512")
    assert studies.PHASE1_STUDIES == ("mlp128", "dense")      # unchanged
    assert "moe" not in studies.FULL_RETUNE_STUDIES           # dense/moe unchanged


@pytest.mark.parametrize("study", ["mlp256", "mlp512"])
def test_full_retune_dag_has_both_tracks(study, tmp_path):
    dag = _dag_by_name(study, tmp_path)
    # CD track: same shape as mlp128's, gated on this study's pilot
    assert dag["adamw_cd_p1_lr"].needs == ("pilot",)
    assert set(dag["muon_cd_p1_lr"].needs) == {"pilot", "adamw_cd_p2_clip"}
    assert set(dag["manifold_cd_p1_lr"].needs) == {"pilot", "adamw_cd_p2_clip"}
    # inherited Phase-II track, unchanged gating, no fresh seeds of its own
    assert dag["muon_stage_a"].needs == ("adamw_edge_b",)
    assert dag["adamw_stage_a"].needs == ()
    # fresh seeds (and so the whole shared tail) hang off the CD track
    for opt in studies.OPTIMIZERS:
        assert dag[f"{opt}_fresh"].needs == (f"{opt}_cd_p2_clip",)
    assert dag["budget_constant"].needs == ("adamw_fresh",)
    assert len([n for n in dag if n.endswith("_fresh")]) == 5


def test_full_retune_cd_uses_own_pilot_and_tracks_keep_own_adamw_group(tmp_path):
    d = 1e-3
    finish_phase1(tmp_path, "mlp128")               # Phase-II donor
    # without mlp256's own pilot the CD track is blocked (own threshold)
    assert studies.cd_replay("mlp256", "adamw", view_of(tmp_path)).blocked
    plant_pilot(tmp_path, "mlp256", (1.0, 1.01, 1.02))
    # CD track: 2x the default lr wins decisively
    complete_cd(tmp_path, "mlp256", "adamw",
                lambda c: 0.5 if math.isclose(c["lr"], 2 * d) else 1.0)
    # inherited track: 0.5x the default lr wins Stage A and B
    _finish_stage_a(tmp_path, "mlp256", "adamw",
                    lambda c: 0.6 if math.isclose(c["lr"], 0.5 * d) else 1.0)
    for _, c in studies._stage_b_cells("mlp256", "adamw", view_of(tmp_path))[0]:
        plant(tmp_path, c, 0.7)
    v = view_of(tmp_path)
    assert studies._phase2_best_trial("mlp256", "adamw", v)["lr"] == pytest.approx(0.5 * d)
    # headline = CD winner
    best = studies.best_config("mlp256", "adamw", v)
    assert best["lr"] == pytest.approx(2 * d) and best["phase"] == "cd"
    fresh = studies.fresh_seed_configs("mlp256", "adamw", v)
    assert [c["seed"] for c in fresh] == [1, 2, 3]
    assert all(c["lr"] == pytest.approx(2 * d) for c in fresh)
    # each track's muon runs use AdamW's winner on THAT track
    rep = studies.cd_replay("mlp256", "muon", v)
    assert not rep.blocked
    assert all(c["adamw_group_lr"] == pytest.approx(2 * d)
               for c in rep.sweeps["p1_lr"].configs)
    st = studies._stage_cells("mlp256", "muon", v)
    assert st is not None
    assert all(c["adamw_group_lr"] == pytest.approx(0.5 * d) for _, c in st[0])


def test_full_retune_cd_matches_mlp128_protocol(tmp_path):
    """Same CD order, grids (relative to the defaults), phase label and
    cd_sweep labels as mlp128; only the study (width) differs."""
    for st in ("mlp128", "mlp256"):
        plant_pilot(tmp_path, st)
    a = studies.cd_replay("mlp128", "muonall", view_of(tmp_path)).sweeps["p1_lr"].configs
    b = studies.cd_replay("mlp256", "muonall", view_of(tmp_path)).sweeps["p1_lr"].configs
    strip = lambda c: {k: v for k, v in c.items() if k not in ("study", "arch", "width")}
    assert [strip(c) for c in a] == [strip(c) for c in b]


def test_full_retune_all_cd_sweeps_match_mlp128(tmp_path):
    """Every CD sweep (lr, wd, momentum/beta1, beta2, clip; both passes; all
    optimizers) generates the same cells at mlp256 as at mlp128, up to the
    study/width fields."""
    def cells(study):
        root = tmp_path / study
        plant_pilot(root, study)
        for opt in studies.OPTIMIZERS:
            # adamw's 2x-lr cell wins pass 1 so a second pass runs too
            complete_cd(root, study, opt,
                        lambda c: 0.5 if (c["optimizer"] == "adamw" and
                                          math.isclose(c["lr"], 2e-3)) else 1.0)
        out = set()
        for p in (root / "results" / study).glob("*.json"):
            c = json.loads(p.read_text())["config"]
            if c["phase"] == "cd":
                out.add(json.dumps({k: v for k, v in c.items()
                                    if k not in ("study", "arch", "width")},
                                   sort_keys=True))
        return out
    a, b = cells("mlp128"), cells("mlp256")
    assert a == b and len(a) > 64     # pass 2 ran for adamw


def test_full_retune_threshold_is_own_pilot_std(tmp_path):
    """Acceptance uses mlp256's OWN pilot std: an improvement of 0.05 is
    accepted where the pilot std is 0.01 (threshold 0.02) and rejected where
    it is 0.05 (threshold 0.10)."""
    d = 1e-3
    win = lambda c: 0.95 if math.isclose(c["lr"], 2 * d) else 1.0
    plant_pilot(tmp_path, "mlp128", (1.0, 1.01, 1.02))     # std 0.01
    plant_pilot(tmp_path, "mlp256", (1.0, 1.05, 1.10))     # std 0.05
    assert complete_cd(tmp_path, "mlp128", "adamw", win).state["lr"] == pytest.approx(2 * d)
    assert complete_cd(tmp_path, "mlp256", "adamw", win).state["lr"] == pytest.approx(d)


def test_trial_counts_tracks_never_summed(tmp_path):
    finish_phase1(tmp_path, "mlp128")
    plant_pilot(tmp_path, "mlp256")
    complete_cd(tmp_path, "mlp256", "adamw")
    _finish_stage_a(tmp_path, "mlp256", "adamw")
    v = view_of(tmp_path)
    n_cd = len([p for p in (tmp_path / "results" / "mlp256").glob("*.json")
                if json.loads(p.read_text())["config"]["phase"] == "cd"])
    head = studies.trial_counts(v, "mlp256")
    inh = studies.trial_counts(v, "mlp256", track="inherited")
    assert head["adamw"] == n_cd
    assert inh["adamw"] == 6          # Stage A only so far
    with pytest.raises(ValueError):
        studies.trial_counts(v, "mlp256", track="both")
