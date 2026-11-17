"""Tests for muonarch/schedule.py (design.md section 3.6, section 7; PLAN.md
Decisions-log rows "Budget-curve step rule" and "When training stops";
DEVIATIONS.md #6).

Conventions pinned here:
- warmup_steps(base) = round(0.05 * base), Python round(), ABSOLUTE steps of the
  1x budget, identical for every branch and fixed for every optimizer
  (DEVIATIONS.md #6): 122 for the LM (base 2441), 176 for the MLP (base 3520).
- branch_steps(base, m) = round(m * base); checkpoint_step(base, m) =
  round(0.8 * branch_steps(base, m)).
- wsd_multiplier: `step` is 0-indexed and the value is the lr multiplier
  applied AT that step. decay_start = total - round(decay_frac * total);
  multiplier = (total - step) / (total - decay_start) for
  decay_start <= step < total, i.e. exactly 1.0 at decay_start, 1/decay_len at
  the final executed step (total - 1), and exactly 0.0 at step = total (one
  past the last step): the schedule ends exactly at the budget.
"""

import math

import pytest

from muonarch.schedule import (
    CHECKPOINT_FRACTIONS,
    branch_steps,
    checkpoint_step,
    make_schedule,
    warmup_steps,
    wsd_multiplier,
)

BASE_LM = 2441
BASE_MLP = 3520

# design.md section 7 table: m' -> (branch steps, branch point, decay length)
TABLE_MLP = {0.125: (440, 352, 88), 0.25: (880, 704, 176), 0.5: (1760, 1408, 352)}
TABLE_LM = {0.125: (305, 244, 61), 0.25: (610, 488, 122), 0.5: (1220, 976, 244)}


def test_checkpoint_fractions_constant():
    assert CHECKPOINT_FRACTIONS == (0.125, 0.25, 0.5)


def test_warmup_absolute_and_fixed():
    """Warmup is round(0.05 * 1x base), absolute: 122 LM / 176 MLP."""
    assert warmup_steps(BASE_LM) == 122
    assert warmup_steps(BASE_MLP) == 176
    # Python round() semantics, explicitly.
    assert warmup_steps(BASE_LM) == round(0.05 * BASE_LM)
    assert warmup_steps(BASE_MLP) == round(0.05 * BASE_MLP)


def test_warmup_below_smallest_branch_point():
    """Sanity inequality (design section 7): the checkpoint is never inside
    warmup: warmup (0.05 * base) < smallest branch point (0.8 * 0.125 * base)."""
    for base in (BASE_LM, BASE_MLP):
        smallest = min(checkpoint_step(base, m) for m in CHECKPOINT_FRACTIONS)
        assert warmup_steps(base) < smallest


@pytest.mark.parametrize(
    "base,table", [(BASE_MLP, TABLE_MLP), (BASE_LM, TABLE_LM)]
)
def test_checkpoint_step_arithmetic(base, table):
    """Reproduce design.md section 7's branch table exactly."""
    for mult, (steps, point, decay_len) in table.items():
        assert branch_steps(base, mult) == steps
        assert checkpoint_step(base, mult) == point
        assert steps - point == decay_len


def test_constant_run_decay_start():
    """The 2x constant run decays from round(0.8 * round(2 * base)):
    MLP 5632, LM 3906 (design section 7)."""
    assert branch_steps(BASE_MLP, 2) == 7040
    assert branch_steps(BASE_LM, 2) == 4882
    assert checkpoint_step(BASE_MLP, 2) == 5632
    assert checkpoint_step(BASE_LM, 2) == 3906
    # The wsd decay-start convention must agree with the checkpoint arithmetic.
    for base in (BASE_MLP, BASE_LM):
        total = branch_steps(base, 2)
        decay_start = total - round(0.2 * total)
        assert decay_start == checkpoint_step(base, 2)


def test_wsd_warmup_shape():
    total, warmup = 440, 176
    assert wsd_multiplier(0, total, warmup) == pytest.approx(1 / warmup)
    assert wsd_multiplier(warmup - 1, total, warmup) == pytest.approx(1.0)
    vals = [wsd_multiplier(s, total, warmup) for s in range(warmup)]
    assert all(b > a for a, b in zip(vals, vals[1:]))  # strictly increasing
    # linear: (step + 1) / warmup
    assert wsd_multiplier(10, total, warmup) == pytest.approx(11 / warmup)


def test_wsd_stable_region_is_one():
    total, warmup = 440, 176
    decay_start = total - round(0.2 * total)  # 352
    for s in range(warmup, decay_start):
        assert wsd_multiplier(s, total, warmup) == 1.0


def test_decay_last_20pct_to_zero():
    """Linear decay over the last 20%: 1.0 at decay_start, (total - step) /
    decay_len throughout, 1/decay_len at the last executed step, exactly 0.0
    at step = total (the schedule ends exactly at the budget)."""
    for base, table in ((BASE_MLP, TABLE_MLP), (BASE_LM, TABLE_LM)):
        warmup = warmup_steps(base)
        for mult, (total, point, decay_len) in table.items():
            assert point == total - round(0.2 * total)
            assert wsd_multiplier(point, total, warmup) == pytest.approx(1.0)
            for s in range(point, total):
                expected = (total - s) / decay_len
                assert wsd_multiplier(s, total, warmup) == pytest.approx(expected)
            # strictly decreasing over the decay
            vals = [wsd_multiplier(s, total, warmup) for s in range(point, total + 1)]
            assert all(b < a for a, b in zip(vals, vals[1:]))
            # final executed step approaches 0 but is nonzero
            assert wsd_multiplier(total - 1, total, warmup) == pytest.approx(
                1 / decay_len
            )
            # at the budget itself the schedule is exactly 0
            assert wsd_multiplier(total, total, warmup) == 0.0


def test_make_schedule_matches_wsd():
    for base in (BASE_MLP, BASE_LM):
        for mult in (*CHECKPOINT_FRACTIONS, 1, 2):
            cfg = {"budget_base_steps": base, "budget_multiplier": mult}
            sched = make_schedule(cfg)
            total = branch_steps(base, mult)
            warmup = warmup_steps(base)  # ALWAYS from the 1x base
            for s in (0, warmup - 1, warmup, total // 2, total - 1, total):
                assert sched(s) == wsd_multiplier(s, total, warmup)


@pytest.mark.parametrize("base", [BASE_MLP, BASE_LM])
@pytest.mark.parametrize("frac", CHECKPOINT_FRACTIONS)
def test_branch_prefix_equivalence(base, frac):
    """For global step < checkpoint, the branch's lr multiplier equals the
    parent constant (2x) run's multiplier IDENTICALLY, elementwise."""
    parent = make_schedule({"budget_base_steps": base, "budget_multiplier": 2})
    child = make_schedule({"budget_base_steps": base, "budget_multiplier": frac})
    point = checkpoint_step(base, frac)
    for step in range(point):
        assert child(step) == parent(step), f"step {step}"


def test_schedule_is_finite_and_bounded():
    sched = make_schedule({"budget_base_steps": BASE_LM, "budget_multiplier": 1})
    total = branch_steps(BASE_LM, 1)
    for s in range(total + 1):
        v = sched(s)
        assert 0.0 <= v <= 1.0 and math.isfinite(v)


# ---------------------------------------------------------------- hashed
# cfg["schedule"] fields are honored (design.md section 4: a hashed field must
# change what the run computes; identity hygiene — production values stay
# 0.05/0.2 everywhere per DEVIATIONS.md #6).

def test_make_schedule_honors_decay_frac():
    base = {"budget_base_steps": 100, "budget_multiplier": 1}
    default = make_schedule(dict(base, schedule={"kind": "wsd"}))
    halved = make_schedule(dict(base, schedule={"kind": "wsd", "decay_frac": 0.5}))
    # decay_frac 0.5: decay starts at step 50 instead of 80
    assert default(60) == 1.0
    assert halved(60) < 1.0
    assert halved(50) == 1.0 and halved(51) < 1.0


def test_make_schedule_honors_warmup_frac():
    base = {"budget_base_steps": 100, "budget_multiplier": 1}
    default = make_schedule(dict(base, schedule={"kind": "wsd"}))
    longer = make_schedule(dict(base, schedule={"kind": "wsd",
                                                "warmup_frac_of_1x": 0.10}))
    # default warmup = 5 steps, longer = 10; the pinned convention has step
    # warmup-1 train at exactly 1.0 ((step+1)/warmup), step warmup-2 below it
    assert default(3) < 1.0 and default(4) == 1.0
    assert longer(4) == pytest.approx(0.5)
    assert longer(8) < 1.0 and longer(9) == 1.0


def test_make_schedule_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        make_schedule({"budget_base_steps": 100, "budget_multiplier": 1,
                       "schedule": {"kind": "cosine"}})
