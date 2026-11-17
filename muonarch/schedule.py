"""WSD (warmup-stable-decay) schedule and budget-curve branch arithmetic.

Implements design.md section 3.6 / section 7 and PLAN.md's Decisions-log rows
"Budget-curve step rule" and "When training stops".

Step-count arithmetic (Python built-in ``round()`` in all three functions):

- ``warmup_steps(base) = round(0.05 * base)`` -- an ABSOLUTE step count of the
  1x budget (122 for the LM's 2,441 steps, 176 for the MLP's 3,520), identical
  for every branch and every budget multiplier, and FIXED at 5% for every
  optimizer including the budget curve (Gate-1 decision, DEVIATIONS.md #6 --
  warmup is not tuned; deliberate departure from Wen et al.).
- ``branch_steps(base, m) = round(m * base)`` -- a budget of m-times has that
  many steps.
- ``checkpoint_step(base, m) = round(0.8 * branch_steps(base, m))`` -- the
  exact global step of the CONSTANT run at which the branch for budget ``m``
  leaves it (80% of the branch budget), taken literally and independent of the
  eval grid.

``wsd_multiplier`` convention (pinned by tests/test_schedule.py):

- ``step`` is 0-indexed; the returned value is the lr multiplier applied AT
  that step (before the update).
- warmup: ``(step + 1) / warmup`` for ``step < warmup`` (so step 0 trains at
  ``1/warmup``, step ``warmup - 1`` at exactly 1.0).
- stable: 1.0 for ``warmup <= step < decay_start`` where
  ``decay_start = total - round(decay_frac * total)``. For the project's
  budgets this equals ``round(0.8 * total)`` (0.2 * int never lands on .5
  mathematically), so the decay starts exactly at the checkpoint arithmetic's
  ``round(0.8 * ...)`` step.
- decay: linear from 1.0 at ``decay_start`` to 0.0 at ``total``:
  ``(total - step) / (total - decay_start)``. The last executed step
  (``total - 1``) therefore trains at ``1/decay_len`` (approaching 0), and the
  multiplier is exactly 0.0 at ``step >= total``: the schedule ends exactly at
  the budget.
"""

from __future__ import annotations

from typing import Callable

__all__ = [
    "CHECKPOINT_FRACTIONS",
    "warmup_steps",
    "branch_steps",
    "checkpoint_step",
    "wsd_multiplier",
    "make_schedule",
]

#: Budget multipliers at which the constant (2x) run saves branch checkpoints.
CHECKPOINT_FRACTIONS: tuple[float, ...] = (0.125, 0.25, 0.5)


def warmup_steps(base_steps_1x: int, frac: float = 0.05) -> int:
    """round(frac * base), absolute steps of the 1x budget; fixed at 5%
    (DEVIATIONS.md #6): 122 for base 2441 (LM), 176 for base 3520 (MLP)."""
    return round(frac * base_steps_1x)


def branch_steps(base_steps_1x: int, mult: float) -> int:
    """Total steps of a budget-multiplier ``mult`` run: round(mult * base)."""
    return round(mult * base_steps_1x)


def checkpoint_step(base_steps_1x: int, mult: float) -> int:
    """Global step at which the ``mult`` branch leaves the constant run:
    round(0.8 * branch_steps(base, mult))."""
    return round(0.8 * branch_steps(base_steps_1x, mult))


def wsd_multiplier(
    step: int, total_steps: int, warmup: int, decay_frac: float = 0.2
) -> float:
    """lr multiplier at 0-indexed ``step`` (see module docstring convention)."""
    if step < warmup:
        return (step + 1) / warmup
    decay_start = total_steps - round(decay_frac * total_steps)
    if step < decay_start:
        return 1.0
    if step >= total_steps:
        return 0.0
    return (total_steps - step) / (total_steps - decay_start)


def make_schedule(cfg: dict) -> Callable[[int], float]:
    """step -> lr multiplier for THIS run.

    ``total = branch_steps(cfg["budget_base_steps"], cfg["budget_multiplier"])``;
    warmup is ALWAYS taken from the 1x base, for every branch. For a
    budget_branch run the ``step`` argument is the GLOBAL step (resumed from
    the checkpoint), so for ``step < checkpoint_step(base, mult)`` the
    multiplier equals the parent constant run's multiplier identically (both
    schedules are in warmup, then constant 1.0, until each one's own decay --
    and every checkpoint precedes both decays;
    tests/test_schedule.py::test_branch_prefix_equivalence).

    The hashed ``cfg["schedule"]`` fields are honored (design.md section 4:
    a hashed field must change what the run computes): ``warmup_frac_of_1x``
    and ``decay_frac`` are plumbed through ``warmup_steps`` /
    ``wsd_multiplier`` (defaults 0.05 / 0.2 — the production values, which
    are FIXED for every planned run per DEVIATIONS.md #6; non-default values
    exist for identity hygiene, not tuning), and any ``kind`` other than
    "wsd" raises.
    """
    sched = cfg.get("schedule") or {}
    kind = sched.get("kind", "wsd")
    if kind != "wsd":
        raise ValueError(f"Unknown schedule kind: {kind!r}")
    base = cfg["budget_base_steps"]
    total = branch_steps(base, cfg["budget_multiplier"])
    warmup = warmup_steps(base, sched.get("warmup_frac_of_1x", 0.05))
    decay_frac = sched.get("decay_frac", 0.2)

    def schedule(step: int) -> float:
        return wsd_multiplier(step, total, warmup, decay_frac)

    return schedule
