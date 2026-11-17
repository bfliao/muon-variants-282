"""Tests for muonarch/runner.py — the run-identity / path layer only
(design.md §4 "Run-config schema and hashing" + §3.9 paths; the Runner class
itself is a later change and is not tested here).

Contract under test:
- HASHED_FIELDS is exactly the §4 starred fields.
- config_hash is stable under key order (top-level and nested) and ignores
  every logged-only field.
- EVERY hashed field (and nested keys inside hashed dict fields) changes the
  hash.
- Path layout: results/<study>/<hash>.json, logs/<study>/<hash>.jsonl,
  checkpoints/<study>/<hash>@<step>.pt.
- is_done is true iff the result JSON exists (any status).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from muonarch import runner


# ---------------------------------------------------------------- fixtures

def make_cfg(**over) -> dict:
    """A complete run config carrying every §4 hashed field."""
    cfg = {
        "schema_version": 1,
        "study": "mlp128",
        "phase": "stage_a",
        "arch": "mlp",
        "width": 128,
        "optimizer": "muon",
        "rule": "keller",
        "lr": 0.02,
        "wd": 0.01,
        "momentum": 0.95,
        "adamw_betas": [0.9, 0.95],
        "adamw_eps": 1e-8,
        "adamw_group_lr": 0.0003,
        "adamw_group_wd": 0.1,
        "ns_steps": 5,
        "ns_dtype": "fp32",
        "manifold": None,
        "init_projection": False,
        "seed": 0,
        "budget_multiplier": 1.0,
        "branch": None,
        "budget_base_steps": 3520,
        "budget_unit": "images",
        "batch": 256,
        "seq_len": None,
        "grad_clip": 1.0,
        "schedule": {"kind": "wsd", "warmup_frac_of_1x": 0.05, "decay_frac": 0.2},
        "precision": "fp32",
        "router_aux_loss_coef": None,
        "eval": {"every_steps": 176, "val_small_tokens": None, "val_final": "full"},
        "data": {"dataset": "cifar5m", "version": "cifar5m-p0p1-v1"},
    }
    cfg.update(over)
    return cfg


#: §4 "logged in the result/log files, NEVER hashed".
LOGGED_ONLY = {
    "device": "mps",
    "hostname": "some-mac.local",
    "data_path": "/somewhere/else/data",
    "torch_version": "2.15.0",
    "transformers_version": "4.57.6",
    "git_commit": "deadbeef",
    "started_at": "2026-11-16T10:00:00",
    "finished_at": "2026-11-16T11:00:00",
    "wall_clock_s": 3600.0,
    "notes": "anything",
}


def _perturbed(v):
    """A value guaranteed to differ from v under canonical JSON."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 1
    if isinstance(v, str):
        return v + "_perturbed"
    if isinstance(v, (list, tuple)):
        return list(v) + [1]
    if isinstance(v, dict):
        d = dict(v)
        k = sorted(d)[0]
        d[k] = _perturbed(d[k])
        return d
    if v is None:
        return "was_null"
    raise TypeError(f"no perturbation for {type(v)}")


# ---------------------------------------------------------------- HASHED_FIELDS

def test_hashed_fields_are_exactly_the_schema_starred_fields():
    assert runner.HASHED_FIELDS == frozenset(make_cfg().keys())


def test_hash_format():
    h = runner.config_hash(make_cfg())
    assert isinstance(h, str) and len(h) == 12
    int(h, 16)  # valid hex


# ---------------------------------------------------------------- stability

def test_hash_stable_under_key_order():
    cfg = make_cfg()
    # Rebuild with reversed insertion order, nested dicts included.
    def rev(d):
        return {
            k: rev(v) if isinstance(v, dict) else v
            for k, v in reversed(list(d.items()))
        }
    assert runner.config_hash(cfg) == runner.config_hash(rev(cfg))


def test_hash_stable_under_tuple_vs_list():
    cfg = make_cfg()
    cfg2 = make_cfg(adamw_betas=(0.9, 0.95))
    assert runner.config_hash(cfg) == runner.config_hash(cfg2)


def test_logged_only_fields_do_not_change_hash():
    cfg = make_cfg()
    h0 = runner.config_hash(cfg)
    noisy = dict(cfg, **LOGGED_ONLY)
    assert runner.config_hash(noisy) == h0
    # one at a time too
    for k, v in LOGGED_ONLY.items():
        assert runner.config_hash(dict(cfg, **{k: v})) == h0, k


def test_canonical_keeps_only_hashed_fields():
    cfg = dict(make_cfg(), **LOGGED_ONLY)
    canon = runner.canonical(cfg)
    assert set(canon) == set(runner.HASHED_FIELDS)


# ---------------------------------------------------------------- sensitivity

def test_every_hashed_field_changes_the_hash():
    base = make_cfg()
    h0 = runner.config_hash(base)
    for field in sorted(runner.HASHED_FIELDS):
        cfg = make_cfg()
        cfg[field] = _perturbed(cfg[field])
        assert runner.config_hash(cfg) != h0, f"field {field!r} did not change the hash"


def test_nested_keys_change_the_hash():
    h0 = runner.config_hash(make_cfg())
    for field, key, value in [
        ("schedule", "decay_frac", 0.3),
        ("schedule", "warmup_frac_of_1x", 0.1),
        ("eval", "every_steps", 100),
        ("data", "version", "cifar5m-p0p1-v2"),
    ]:
        cfg = make_cfg()
        cfg[field] = dict(cfg[field], **{key: value})
        assert runner.config_hash(cfg) != h0, (field, key)


def test_manifold_and_branch_subdicts_hash():
    man = {"dual_cap": 5, "tol": 1e-6, "dual_lr": 0.01, "msign_steps": 10}
    cfg = make_cfg(optimizer="manifold", rule=None, wd=None, manifold=man)
    h = runner.config_hash(cfg)
    cfg2 = make_cfg(optimizer="manifold", rule=None, wd=None,
                    manifold=dict(man, dual_cap=100))
    assert runner.config_hash(cfg2) != h
    br = {"parent_hash": "ab12cd34ef56", "checkpoint_step": 1408}
    cfg3 = make_cfg(branch=br, phase="budget_branch", budget_multiplier=0.5)
    cfg4 = make_cfg(branch=dict(br, checkpoint_step=704), phase="budget_branch",
                    budget_multiplier=0.5)
    assert runner.config_hash(cfg3) != runner.config_hash(cfg4)


def test_float_representation_distinguishes_values():
    # floats go through repr (shortest round-trip): a 1-ulp change is a new run
    h0 = runner.config_hash(make_cfg(lr=0.02))
    h1 = runner.config_hash(make_cfg(lr=0.020000000000000004))
    assert h0 != h1


# ---------------------------------------------------------------- paths

def test_path_layout(tmp_path):
    cfg = make_cfg(study="dense")
    h = runner.config_hash(cfg)
    assert runner.result_path(tmp_path, cfg) == tmp_path / "results" / "dense" / f"{h}.json"
    assert runner.log_path(tmp_path, cfg) == tmp_path / "logs" / "dense" / f"{h}.jsonl"
    assert runner.ckpt_path(tmp_path, cfg, 1408) == (
        tmp_path / "checkpoints" / "dense" / f"{h}@1408.pt"
    )


def test_ckpt_file_matches_ckpt_path(tmp_path):
    cfg = make_cfg(study="moe")
    h = runner.config_hash(cfg)
    assert runner.ckpt_file(tmp_path, "moe", h, 976) == runner.ckpt_path(tmp_path, cfg, 976)


# ---------------------------------------------------------------- is_done

def test_is_done(tmp_path):
    cfg = make_cfg()
    assert not runner.is_done(tmp_path, cfg)
    p = runner.result_path(tmp_path, cfg)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"status": "completed"}))
    assert runner.is_done(tmp_path, cfg)
    # a diverged result is also done (divergence is a completed run)
    p.write_text(json.dumps({"status": "diverged"}))
    assert runner.is_done(tmp_path, cfg)
    # a different config is not done
    assert not runner.is_done(tmp_path, make_cfg(seed=1))
