"""Study definitions, tuning grids, and the per-study phase/gating DAG.

Implements design.md §3.10, §5 and §6, adapted to the final module reality:

* Two DAG shapes. Phase-I studies (mlp128, dense) run the PLAN's coordinate
  descent (CD_ORDER, ≤2 passes, acceptance > 2x pilot seed std); Phase-II
  studies (mlp256, mlp512, moe) inherit each optimizer's non-lr/wd values from
  their donor (PHASE2_DONOR) and run the 12-trial Stage A/B lr x wd recipe.
  Both shapes then share: fresh seeds -> AdamW budget curve -> Muon init
  ablation -> faithful-settings reruns.

* MLP protocol change (Gate 3, DEVIATIONS.md #12): FULL_RETUNE_STUDIES
  (mlp256, mlp512) run BOTH tracks — the full Phase-I coordinate descent
  (headline: fresh seeds and the shared tail use its winner) and the
  inherited Phase-II recipe at seed 0 (for H4). Each track's muon/manifold
  auxiliary AdamW group comes from AdamW's winner on the same track.

* Configs are generated LAZILY: a ``Phase``'s ``configs()`` closure reads the
  ``ResultsView`` it was built with, so later phases' configs depend on
  earlier results (cd acceptance, Stage-B best lrs, edge extensions, half-LR
  promotion). A phase with an empty config list is complete.

* ``Phase`` carries one field beyond the design sketch: ``ready`` — a
  predicate for gates that live OUTSIDE this study's phase list (the Phase-II
  donor inheritance: mlp256/mlp512 wait on mlp128's Phase I, moe on dense's).
  Within-study ordering uses ``needs`` as designed.

* ``cd_sweep`` — every Phase-I coordinate-descent config carries a
  ``cd_sweep`` field naming the swept hyperparameter ("lr", "wd", "momentum",
  "beta1", "beta2", "clip"); analysis._lr_sweep_rows selects the lr sweep by
  ``cd_sweep == "lr"``. DECISION: cd_sweep is NOT hashed (it is not in
  runner.HASHED_FIELDS, which the identity review froze). Consequence: when a
  later sweep or pass revisits an identical configuration (e.g. the incumbent
  cell re-appearing in every subsequent sweep), it hashes to the existing run
  and REUSES its result — no duplicate training — and the stored result keeps
  the FIRST sweep's cd_sweep label. Analysis therefore sees each unique
  config exactly once, labeled by the sweep that first ran it; a revisited
  incumbent never shows up twice under two labels.

* Grids. The 6-point factor-2 lr grid is current x 2^k, k in LR_GRID_KS =
  (-3, -2, -1, 0, +1, +2): consecutive points differ by 2, max/min = 32x, and
  the centre value is one of the two middle points. The extra low-side point
  is deliberate (an lr too high diverges; the edge rule self-corrects a
  slightly-off centre — DEVIATIONS.md #8). Because the ks are fixed constants
  applied to the same yaml centre, Stage-A grid positions relative to the
  default are identical at every MLP width (H4 for free).

* Edge rule (Gate-0 answer 5): LEARNING-RATE grids only, ≤2 trials beyond an
  edge, generated one at a time (the second only if the first extension wins
  again on the new edge). Ties in a sweep prefer the cell generated first —
  cells are ordered incumbent/centre first — so an all-tied sweep never fires
  the edge rule and never accepts a change. Phase-I lr edge trials carry
  phase="cd" and cd_sweep="lr" (they ARE lr-sweep cells); Phase-II edge
  trials carry phase="edge". For Manifold's quarter-octave Stage B, a cell
  only counts as an edge if NO evaluated lr (Stage A, edge, or Stage B) lies
  beyond it — the quarter-octave grid is interior to the factor-2 bracket, so
  this fires only when the Stage-A winner was itself a capped edge extension.
  Every edge-extension config carries a logged-only ``edge_trial: true``
  field (NOT hashed, exactly like ``cd_sweep``), so edge trials are visible
  in stored result configs for analysis.edge_rule_report; ordinary grid
  cells never carry it.

* Selection (design §6): Phase-I studies use the coordinate-descent accepted
  state directly as the best config; Phase-II studies take the min seed-0
  final validation loss over Stage A, edge and Stage B trials. Diverged runs
  count as +inf. Fresh seeds are 1..final_seed_count() — a module-level
  setting, default 3, set to 5 only by an explicit set_final_seed_count(5)
  after Gate 4 (Gate-0 answer 6). The init ablation stays at seeds 1-3
  (PLAN confounder 10: "3 seeds"; answer 6 names only fresh seeds and the
  budget curve as growing to 5).

This module performs no training and writes nothing: it only reads result
JSONs through ``ResultsView``. The Runner (muonarch/runner.py) executes.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import yaml

from muonarch import runner
from muonarch.schedule import CHECKPOINT_FRACTIONS, checkpoint_step

__all__ = [
    "STUDIES",
    "OPTIMIZERS",
    "PHASE1_STUDIES",
    "PHASE2_DONOR",
    "PILOT_STUDIES",
    "PILOT_SEEDS",
    "CD_ORDER",
    "LR_GRID_KS",
    "QUARTER_OCTAVE_KS",
    "MAX_EDGE_TRIALS",
    "Phase",
    "ResultsView",
    "load_study",
    "set_final_seed_count",
    "final_seed_count",
    "cd_hypers",
    "cd_defaults",
    "run_config",
    "pilot_configs",
    "pilot_seed_std",
    "cd_replay",
    "phase1_values",
    "inherited_values",
    "stage_a_grid",
    "best_config",
    "fresh_seed_configs",
    "budget_constant_configs",
    "budget_branch_configs",
    "budget_halflr_configs",
    "init_ablation_configs",
    "faithful_configs",
    "build_dag",
    "planned_jobs",
    "trial_counts",
]

SCHEMA_VERSION = 1

STUDIES: tuple[str, ...] = ("mlp128", "mlp256", "mlp512", "dense", "moe")
OPTIMIZERS: tuple[str, ...] = ("adamw", "muon", "moonlight", "manifold", "muonall")

PHASE1_STUDIES: tuple[str, ...] = ("mlp128", "dense")
PHASE2_DONOR: dict[str, str] = {"mlp256": "mlp128", "mlp512": "mlp128", "moe": "dense"}
#: MLP protocol change (Paul, Gate 3; DEVIATIONS.md #12): these Phase-II
#: studies ALSO run the full Phase-I coordinate descent independently (same
#: CD_ORDER, grids, pass limit and acceptance rule as mlp128, starting from
#: literature defaults, acceptance threshold from their own pilot). The
#: re-tuned CD winner is the HEADLINE (fresh seeds, budget curve, init
#: ablation, faithful reruns, main table); the inherited Phase-II track is
#: kept at seed 0 only, for H4 (optimal lr inherited vs fully re-tuned).
#: dense/moe are unchanged.
FULL_RETUNE_STUDIES: tuple[str, ...] = ("mlp256", "mlp512")
#: Studies that run the 3-seed AdamW pilot (PLAN "Seed-spread pilot");
#: mlp256 added with the full re-tune (it needs its own acceptance threshold).
PILOT_STUDIES: tuple[str, ...] = ("mlp128", "mlp256", "mlp512", "dense")
PILOT_SEEDS: tuple[int, ...] = (101, 102, 103)  # Gate-0 answer 6

#: Phase-I coordinate-descent sweep order. NO warmup — fixed at 5% of the 1x
#: budget for every optimizer and every run (Gate 1, DEVIATIONS.md #6).
CD_ORDER: tuple[str, ...] = ("lr", "wd", "momentum", "beta2", "clip")

LR_GRID_KS: tuple[int, ...] = (-3, -2, -1, 0, 1, 2)   # 6 points, spans 32x
WD_FACTORS: tuple[float, ...] = (0.0, 0.3, 1.0, 3.0)  # x plain default d
STAGE_B_WD_FACTORS: tuple[float, ...] = (0.0, 0.3, 3.0)  # d already ran in Stage A
MOMENTUM_GRID: tuple[float, ...] = (0.9, 0.95, 0.98)  # Muon family
BETA1_GRID: tuple[float, ...] = (0.8, 0.9, 0.95)      # AdamW
BETA2_GRID: tuple[float, ...] = (0.95, 0.99, 0.999)   # AdamW only
CLIP_GRID: tuple[Optional[float], ...] = (None, 0.5, 1.0)
QUARTER_OCTAVE_KS: tuple[float, ...] = (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75)
MAX_EDGE_TRIALS = 2

_RULES = {"adamw": None, "muon": "keller", "moonlight": "moonlight",
          "manifold": None, "muonall": "keller"}

# ---------------------------------------------------------------------------
# final-seed count (Gate-0 answer 6: ONE count for all five studies)

_FINAL_SEED_COUNT = 3


def set_final_seed_count(n: int) -> None:
    """3 by default; the lead calls set_final_seed_count(5) after Gate 4 iff
    the pilot seed spread demands it. Applies to fresh seeds AND the budget
    curve in every study."""
    if n not in (3, 5):
        raise ValueError("final seed count must be 3 or 5 (Gate-0 answer 6)")
    global _FINAL_SEED_COUNT
    _FINAL_SEED_COUNT = n


def final_seed_count() -> int:
    return _FINAL_SEED_COUNT


def _fresh_seeds() -> tuple[int, ...]:
    return tuple(range(1, _FINAL_SEED_COUNT + 1))


# ---------------------------------------------------------------------------
# study yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@lru_cache(maxsize=None)
def _load_study_cached(name: str, config_dir: str) -> str:
    return (Path(config_dir) / f"{name}.yaml").read_text()


def load_study(name: str, config_dir: Path | str | None = None) -> dict:
    """Parsed configs/<name>.yaml (study constants + per-optimizer defaults)."""
    if name not in STUDIES:
        raise ValueError(f"unknown study {name!r}; expected one of {STUDIES}")
    text = _load_study_cached(name, str(config_dir or _CONFIG_DIR))
    sv = yaml.safe_load(text)
    assert sv["study"] == name, f"{name}.yaml declares study {sv['study']!r}"
    excl = sv.get("exclude_optimizers") or []
    unknown = set(excl) - set(OPTIMIZERS)
    if unknown:
        raise ValueError(f"{name}.yaml exclude_optimizers: unknown {sorted(unknown)}")
    if "adamw" in excl:
        raise ValueError(
            f"{name}.yaml: adamw cannot be excluded — it is the baseline and "
            f"gates the muon/manifold chains and the budget curve")
    return sv


def _included(sv: dict) -> tuple[str, ...]:
    """OPTIMIZERS minus the study's exclude_optimizers — the PLAN cut-order
    lever (drop Manifold from moe first, then mlp512) as a config edit, no
    code change: the excluded optimizer's whole chain (tuning, fresh,
    faithful — and the init ablation if it is muon) leaves the DAG; nothing
    else moves."""
    excl = set(sv.get("exclude_optimizers") or [])
    return tuple(o for o in OPTIMIZERS if o not in excl)


# ---------------------------------------------------------------------------
# results view

class ResultsView:
    """Read-only view over results/<study>/<hash>.json under one root.

    ``loss`` returns None while a run is missing (not done), +inf for a
    diverged/lossless result, and the final validation loss otherwise.
    Results are cached per view instance; the Runner builds a fresh view for
    every DAG rebuild.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._cache: dict[str, dict | None] = {}

    def result(self, cfg: dict) -> dict | None:
        h = runner.config_hash(cfg)
        if h not in self._cache:
            p = runner.result_path(self.root, cfg)
            res = None
            if p.is_file():
                try:
                    res = json.loads(p.read_text())
                except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                    res = None  # corrupt: counts as NOT done; train.run repairs
            self._cache[h] = res
        return self._cache[h]

    def done(self, cfg: dict) -> bool:
        return self.result(cfg) is not None

    def loss(self, cfg: dict) -> float | None:
        r = self.result(cfg)
        if r is None:
            return None
        v = r.get("final_val_loss")
        if v is None or not math.isfinite(v):
            return math.inf
        return float(v)


# ---------------------------------------------------------------------------
# config construction

def cd_defaults(sv: dict, opt: str) -> dict:
    """The literature-default tunable-value state for one optimizer:
    {"lr", "wd", "momentum", "beta1", "beta2", "clip"}."""
    d = sv["optimizers"][opt]
    return {
        "lr": float(d["lr"]),
        "wd": None if d.get("wd") is None else float(d["wd"]),
        "momentum": float(d.get("momentum", 0.95)) if opt != "adamw" else None,
        "beta1": float(d.get("beta1", 0.9)),
        "beta2": float(d.get("beta2", 0.95)),
        "clip": sv.get("grad_clip", 1.0),
    }


def run_config(sv: dict, opt: str, phase: str, values: dict, *, seed: int = 0,
               adamw_group: tuple[float, float] | None = None,
               cd_sweep: str | None = None, **over) -> dict:
    """One full run config carrying every runner.HASHED_FIELD, with the study
    constants copied from the yaml and the tunable ``values`` applied.

    ``adamw_group`` = (lr, wd) of AdamW's tuned winner for this study — REQUIRED
    for muon and manifold (their auxiliary AdamW group, Gate-0 answer 12),
    forbidden otherwise. ``over`` overrides top-level fields (faithful reruns,
    budgets, branches, smoke worlds...).
    """
    if opt in ("muon", "manifold"):
        if adamw_group is None:
            raise ValueError(f"{opt} configs need adamw_group=(lr, wd)")
        group_lr, group_wd = float(adamw_group[0]), float(adamw_group[1])
    else:
        if adamw_group is not None:
            raise ValueError(f"{opt} has no separate AdamW group values")
        group_lr = group_wd = None
    manifold = None
    if opt == "manifold":
        m = sv["optimizers"]["manifold"]
        manifold = {
            "dual_cap": int(m.get("dual_cap", 5)),
            "tol": float(m.get("tol", 1e-6)),
            "dual_lr": float(m.get("dual_lr", 0.01)),
            "msign_steps": int(m.get("msign_steps", 10)),
        }
    cfg = {
        "schema_version": SCHEMA_VERSION,
        "study": sv["study"],
        "phase": phase,
        "arch": sv["arch"],
        "width": sv.get("width"),
        "optimizer": opt,
        "rule": _RULES[opt],
        "lr": float(values["lr"]),
        "wd": None if opt == "manifold" else float(values["wd"]),
        "momentum": None if opt == "adamw" else float(values["momentum"]),
        "adamw_betas": ([float(values["beta1"]), float(values["beta2"])]
                        if opt == "adamw" else [0.9, 0.95]),
        "adamw_eps": float(sv["optimizers"]["adamw"].get("eps", 1e-8)),
        "adamw_group_lr": group_lr,
        "adamw_group_wd": group_wd,
        "ns_steps": 5,
        "ns_dtype": "fp32",
        "manifold": manifold,
        "init_projection": opt == "manifold",
        "seed": seed,
        "budget_multiplier": 1.0,
        "branch": None,
        "budget_base_steps": sv["budget_base_steps"],
        "budget_unit": sv["budget_unit"],
        "batch": sv["batch"],
        "seq_len": sv.get("seq_len"),
        "grad_clip": values["clip"],
        "schedule": dict(sv["schedule"]),
        "precision": sv["precision"],
        "router_aux_loss_coef": sv.get("router_aux_loss_coef"),
        "eval": dict(sv["eval"]),
        "data": dict(sv["data"]),
    }
    cfg.update(over)
    if cd_sweep is not None:
        cfg["cd_sweep"] = cd_sweep  # logged in the result config, NOT hashed
    return cfg


# ---------------------------------------------------------------------------
# pilot

def pilot_configs(study: str, config_dir=None) -> list[dict]:
    """AdamW default config, seeds 101-103; PILOT_STUDIES only (mlp128, mlp256
    — added with the full re-tune, DEVIATIONS.md #12 — mlp512, dense)."""
    if study not in PILOT_STUDIES:
        return []
    sv = load_study(study, config_dir)
    return [run_config(sv, "adamw", "pilot", cd_defaults(sv, "adamw"), seed=s)
            for s in PILOT_SEEDS]


def pilot_seed_std(study: str, view: ResultsView, config_dir=None) -> float | None:
    """Sample std (ddof=1) of the pilot final val losses; None until every
    pilot run is done; +inf if any pilot run diverged (nothing is ever
    accepted then — a diverged pilot is a red flag the lead must see)."""
    losses = [view.loss(c) for c in pilot_configs(study, config_dir)]
    if any(v is None for v in losses):
        return None
    if any(math.isinf(v) for v in losses):
        return math.inf
    return statistics.stdev(losses)


# ---------------------------------------------------------------------------
# sweeps, winners, edge rule

def cd_hypers(opt: str) -> tuple[str, ...]:
    """CD_ORDER restricted to the hypers this optimizer tunes. ``momentum``
    means beta1 for adamw; manifold skips wd (no wd on constrained matrices);
    beta2 is AdamW-only."""
    if opt == "adamw":
        return ("lr", "wd", "momentum", "beta2", "clip")
    if opt == "manifold":
        return ("lr", "momentum", "clip")
    return ("lr", "wd", "momentum", "clip")


def _grid_values(hp: str, opt: str, state: dict, sv: dict) -> list:
    if hp == "lr":
        return [state["lr"] * 2.0 ** k for k in LR_GRID_KS]
    if hp == "wd":
        d = float(sv["optimizers"][opt]["wd"])  # plain default (answer 3)
        return [f * d for f in WD_FACTORS]
    if hp == "momentum":
        return list(BETA1_GRID if opt == "adamw" else MOMENTUM_GRID)
    if hp == "beta2":
        return list(BETA2_GRID)
    if hp == "clip":
        return list(CLIP_GRID)
    raise ValueError(hp)


_STATE_KEY = {"lr": "lr", "wd": "wd", "beta2": "beta2", "clip": "clip"}


def _state_key(hp: str, opt: str) -> str:
    if hp == "momentum":
        return "beta1" if opt == "adamw" else "momentum"
    return _STATE_KEY[hp]


def _sweep_label(hp: str, opt: str) -> str:
    """The cd_sweep label: the hyper actually swept (beta1 for adamw's
    momentum slot)."""
    return _state_key(hp, opt) if hp == "momentum" else hp


def _centre_first(values: list, incumbent) -> list:
    """Sweep evaluation order: incumbent cell first (ties prefer it), the
    rest in grid order."""
    out = [v for v in values if _same(v, incumbent)]
    out += [v for v in values if not _same(v, incumbent)]
    return out


def _same(a, b, rtol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=rtol, abs_tol=0.0) or a == b


def _winner(cells: list[tuple[object, dict]], view: ResultsView):
    """(value, cfg, loss) of the lowest-loss cell; ties keep the earliest
    cell in the list (incumbent/centre first, extensions last). All cells
    must be done."""
    best = None
    for value, cfg in cells:
        loss = view.loss(cfg)
        if loss is None:
            raise RuntimeError("winner asked before the sweep completed")
        if best is None or loss < best[2]:
            best = (value, cfg, loss)
    if best is None:
        raise RuntimeError("empty sweep")
    return best


def _lr_edge(cells: list[tuple[float, dict]], view: ResultsView,
             make_cfg: Callable[[float], dict], factor: float,
             known_lrs: tuple[float, ...] = ()) -> tuple[list[dict], bool]:
    """Lazy ≤2-trial edge extensions for an lr sweep (lr grids ONLY).

    ``cells`` are the sweep's grid cells (centre first). An extension is
    generated only once all current cells are done AND the winner sits on the
    outer boundary of every evaluated lr (grid + extensions + ``known_lrs``,
    e.g. Stage-A lrs for Manifold's Stage B). Returns (extension configs so
    far, complete) — complete means no further extension is wanted; the
    caller still checks the returned configs are done.
    """
    cells = list(cells)
    exts: list[tuple[float, dict]] = []
    while True:
        all_cells = cells + exts
        if not all(view.done(c) for _, c in all_cells):
            return [c for _, c in exts], False
        value, _, _ = _winner(all_cells, view)
        lrs = sorted({lr for lr, _ in all_cells} | set(known_lrs))
        at_low = _same(value, lrs[0])
        at_high = _same(value, lrs[-1])
        if not (at_low or at_high) or len(exts) >= MAX_EDGE_TRIALS:
            return [c for _, c in exts], True
        nxt = value * factor if at_high else value / factor
        if any(_same(nxt, lr) for lr in lrs):   # already evaluated: not an edge
            return [c for _, c in exts], True
        ext = make_cfg(nxt)
        # Logged-only marker (NOT hashed, like cd_sweep): makes edge trials —
        # Phase-I cd lr extensions AND Phase-II edge trials — visible to
        # analysis.edge_rule_report via the stored result config.
        ext["edge_trial"] = True
        exts.append((nxt, ext))


# ---------------------------------------------------------------------------
# Phase I: coordinate descent (mlp128, dense)

@dataclass
class _SweepState:
    configs: list[dict] = field(default_factory=list)       # grid cells
    edge_configs: list[dict] = field(default_factory=list)  # lr extensions
    complete: bool = False
    edge_complete: bool = False


@dataclass
class CDReplay:
    """The deterministic replay of one optimizer's coordinate descent from
    the results on disk. ``sweeps`` is keyed "p<pass>_<hp>"."""
    blocked: bool                      # pilot / AdamW gate not available yet
    state: dict                        # current accepted values
    sweeps: dict[str, _SweepState]
    done: bool                         # whole chain finished


def _cd_spec(opt: str) -> list[tuple[int, str]]:
    return [(p, hp) for p in (1, 2) for hp in cd_hypers(opt)]


def _has_cd(study: str) -> bool:
    """Studies that run the Phase-I coordinate descent (and select on it)."""
    return study in PHASE1_STUDIES or study in FULL_RETUNE_STUDIES


def _adamw_group_for(study: str, opt: str, view: ResultsView,
                     config_dir=None, track: str = "auto") -> tuple[float, float] | None:
    """(lr, wd) the muon/manifold auxiliary AdamW group must use — AdamW's
    tuned winner for this study ON THE SAME TRACK (Phase I: cd state; Phase
    II: Stage winner). ``track``: "cd", "phase2", or "auto" (= cd in
    Phase-I studies, phase2 otherwise). In FULL_RETUNE_STUDIES the CD track
    uses AdamW's CD values and the inherited Phase-II track keeps using
    AdamW's Phase-II winner, so each track is internally consistent. None
    while AdamW's tuning on that track is incomplete. For other optimizers:
    None (and unused)."""
    if opt not in ("muon", "manifold"):
        return None
    if track == "auto":
        track = "cd" if study in PHASE1_STUDIES else "phase2"
    if track == "cd":
        vals = phase1_values(study, "adamw", view, config_dir)
        return None if vals is None else (vals["lr"], vals["wd"])
    best = _phase2_best_trial(study, "adamw", view, config_dir)
    return None if best is None else (best["lr"], best["wd"])


def cd_replay(study: str, opt: str, view: ResultsView,
              config_dir=None) -> CDReplay:
    """Walk the coordinate descent for (study, opt) against the results on
    disk: at most 2 passes over cd_hypers(opt), acceptance only when the
    sweep winner improves on the incumbent by > 2x the study's pilot seed
    std, early stop when pass 1 accepted nothing, lr edge rule inside every
    lr sweep. Stops at the first sweep that is not fully run (its grid is the
    frontier; later sweeps have empty configs and are incomplete)."""
    sv = load_study(study, config_dir)
    sweeps = {f"p{p}_{hp}": _SweepState() for p, hp in _cd_spec(opt)}
    state = cd_defaults(sv, opt)

    std = pilot_seed_std(study, view, config_dir)
    group = _adamw_group_for(study, opt, view, config_dir, track="cd")
    if std is None or (opt in ("muon", "manifold") and group is None):
        return CDReplay(blocked=True, state=state, sweeps=sweeps, done=False)

    pass_changed = {1: False, 2: False}
    frontier = False
    for p, hp in _cd_spec(opt):
        ss = sweeps[f"p{p}_{hp}"]
        if frontier:
            continue
        if p == 2 and not pass_changed[1]:
            ss.complete = ss.edge_complete = True   # early stop: empty sweep
            continue
        key = _state_key(hp, opt)
        label = _sweep_label(hp, opt)
        incumbent = state[key]
        values = _centre_first(_grid_values(hp, opt, state, sv), incumbent)

        def cfg_for(v, _key=key, _label=label):
            vals = dict(state)
            vals[_key] = v
            return run_config(sv, opt, "cd", vals, seed=0,
                              adamw_group=group, cd_sweep=_label)

        cells = [(v, cfg_for(v)) for v in values]
        ss.configs = [c for _, c in cells]
        if not all(view.done(c) for c in ss.configs):
            frontier = True
            continue
        ss.complete = True
        if hp == "lr":
            ss.edge_configs, ss.edge_complete = _lr_edge(
                cells, view, cfg_for, factor=2.0)
            if not (ss.edge_complete and all(view.done(c) for c in ss.edge_configs)):
                frontier = True
                continue
            cells = cells + [(c["lr"], c) for c in ss.edge_configs]
        else:
            ss.edge_complete = True   # edge rule is lr-only
        value, _, loss = _winner(cells, view)
        inc_loss = next(view.loss(c) for v, c in cells if _same(v, incumbent))
        if not _same(value, incumbent) and inc_loss - loss > 2 * std:
            state[key] = value
            pass_changed[p] = True
    done = not frontier
    return CDReplay(blocked=False, state=state, sweeps=sweeps, done=done)


def phase1_values(study: str, opt: str, view: ResultsView,
                  config_dir=None) -> dict | None:
    """The accepted value set after the full coordinate descent — for
    Phase-I studies this IS the best config (design §6 Selection). None
    until the descent is complete."""
    rep = cd_replay(study, opt, view, config_dir)
    return rep.state if (not rep.blocked and rep.done) else None


# ---------------------------------------------------------------------------
# Phase II: 12-trial lr x wd recipe (mlp256, mlp512, moe)

def inherited_values(study: str, opt: str, view: ResultsView,
                     config_dir=None) -> dict | None:
    """Non-lr/wd values (momentum, beta1, beta2, clip) copied from the donor
    study's Phase-I result for this optimizer; lr and wd reset to this
    study's literature defaults (they are what Phase II re-tunes). None while
    the donor's Phase I is incomplete."""
    donor = PHASE2_DONOR[study]
    donor_vals = phase1_values(donor, opt, view, config_dir)
    if donor_vals is None:
        return None
    sv = load_study(study, config_dir)
    vals = cd_defaults(sv, opt)
    for key in ("momentum", "beta1", "beta2", "clip"):
        vals[key] = donor_vals[key]
    return vals


def stage_a_grid(study: str, opt: str, config_dir=None) -> list[float]:
    """The 6 Stage-A lrs: literature default x 2^k, k in LR_GRID_KS — the
    same positions relative to the default at every width."""
    sv = load_study(study, config_dir)
    centre = float(sv["optimizers"][opt]["lr"])
    return [centre * 2.0 ** k for k in LR_GRID_KS]


def _stage_cells(study: str, opt: str, view: ResultsView, config_dir=None):
    """(stage_a cells, edge configs, edge_complete, make_cfg, vals, group) or
    None while blocked (donor Phase I, or AdamW's own Phase II for
    muon/manifold, incomplete)."""
    sv = load_study(study, config_dir)
    vals = inherited_values(study, opt, view, config_dir)
    group = _adamw_group_for(study, opt, view, config_dir, track="phase2")
    if vals is None or (opt in ("muon", "manifold") and group is None):
        return None
    centre = float(sv["optimizers"][opt]["lr"])

    def cfg_for(lr, phase="stage_a"):
        v = dict(vals)
        v["lr"] = lr
        return run_config(sv, opt, phase, v, seed=0, adamw_group=group)

    lrs = _centre_first([centre * 2.0 ** k for k in LR_GRID_KS], centre)
    cells = [(lr, cfg_for(lr)) for lr in lrs]
    edge, edge_complete = _lr_edge(
        cells, view, lambda lr: cfg_for(lr, "edge"), factor=2.0)
    return cells, edge, edge_complete, cfg_for, vals, group


def _stage_b_cells(study: str, opt: str, view: ResultsView, config_dir=None):
    """Stage-B cells: 2 best lrs x wd {0, 0.3d, 3d} (d already ran in
    Stage A); Manifold instead best lr x 2^k, quarter octaves. None until
    Stage A + edge are complete."""
    st = _stage_cells(study, opt, view, config_dir)
    if st is None:
        return None
    cells, edge, edge_complete, cfg_for, vals, group = st
    all_cells = cells + [(c["lr"], c) for c in edge]
    if not (edge_complete and all(view.done(c) for _, c in all_cells)):
        return None
    sv = load_study(study, config_dir)
    if opt == "manifold":
        best_lr, _, _ = _winner(all_cells, view)
        out = []
        for k in QUARTER_OCTAVE_KS:
            out.append((best_lr * 2.0 ** k, cfg_for(best_lr * 2.0 ** k, "stage_b")))
        return out, all_cells, vals, group
    # rank lrs by their best loss, centre-first order breaking ties
    ranked = []
    for lr, cfg in all_cells:
        loss = view.loss(cfg)
        if not any(_same(lr, r[0]) for r in ranked):
            ranked.append((lr, loss))
        else:
            ranked = [(l, min(lo, loss)) if _same(l, lr) else (l, lo)
                      for l, lo in ranked]
    ranked.sort(key=lambda t: t[1])
    top2 = [lr for lr, _ in ranked[:2]]
    d = float(sv["optimizers"][opt]["wd"])
    out = []
    for lr in top2:
        for f in STAGE_B_WD_FACTORS:
            v = dict(vals)
            v["lr"] = lr
            v["wd"] = f * d
            out.append((lr, run_config(sv, opt, "stage_b", v, seed=0,
                                       adamw_group=group)))
    return out, all_cells, vals, group


def _stage_b_edge(study: str, opt: str, view: ResultsView, config_dir=None
                  ) -> tuple[list[dict], bool]:
    """Post-Stage-B lr edge trials — lr grids only, so non-manifold Stage B
    (2 fixed lrs x wd) never fires. For Manifold the quarter-octave grid is
    checked against EVERY evaluated lr, so it fires only when no Stage-A/edge
    lr lies beyond the winner."""
    if opt != "manifold":
        return [], True
    sb = _stage_b_cells(study, opt, view, config_dir)
    if sb is None:
        return [], False
    b_cells, a_cells, vals, group = sb
    if not all(view.done(c) for _, c in b_cells):
        return [], False
    sv = load_study(study, config_dir)

    def cfg_for(lr):
        v = dict(vals)
        v["lr"] = lr
        return run_config(sv, opt, "edge", v, seed=0, adamw_group=group)

    known = tuple(lr for lr, _ in a_cells)
    return _lr_edge(b_cells, view, cfg_for, factor=2.0 ** 0.25, known_lrs=known)


def _phase2_best_trial(study: str, opt: str, view: ResultsView,
                       config_dir=None) -> dict | None:
    """Min seed-0 final val loss over Stage A, edge and Stage B trials
    (diverged = +inf). None until all those phases are complete."""
    sb = _stage_b_cells(study, opt, view, config_dir)
    if sb is None:
        return None
    b_cells, a_cells, _, _ = sb
    edge_b, edge_b_complete = _stage_b_edge(study, opt, view, config_dir)
    trials = [c for _, c in a_cells] + [c for _, c in b_cells] + edge_b
    if not (edge_b_complete and all(view.done(c) for c in trials)):
        return None
    pairs = [(None, c) for c in trials]
    _, cfg, _ = _winner(pairs, view)
    return cfg


# ---------------------------------------------------------------------------
# selection and shared phases (both DAG shapes)

def best_config(study: str, opt: str, view: ResultsView,
                config_dir=None) -> dict | None:
    """The optimizer's winning seed-0 config for this study, or None while
    tuning is incomplete. Phase-I studies AND FULL_RETUNE_STUDIES: the
    coordinate-descent accepted state, directly (the re-tuned headline);
    other Phase-II studies: the min-loss tuning trial (design §6).
    """
    sv = load_study(study, config_dir)
    if _has_cd(study):
        vals = phase1_values(study, opt, view, config_dir)
        if vals is None:
            return None
        group = _adamw_group_for(study, opt, view, config_dir, track="cd")
        return run_config(sv, opt, "cd", vals, seed=0, adamw_group=group)
    return _phase2_best_trial(study, opt, view, config_dir)


def _rerun_of(best: dict, phase: str, seed: int, **over) -> dict:
    cfg = {k: v for k, v in best.items() if k not in ("cd_sweep", "edge_trial")}
    cfg.update({"phase": phase, "seed": seed})
    cfg.update(over)
    return cfg


def fresh_seed_configs(study: str, opt: str, view: ResultsView,
                       config_dir=None) -> list[dict]:
    """Winner rerun on fresh seeds 1..final_seed_count() (never seed 0:
    winner's curse — PLAN TL;DR 9)."""
    best = best_config(study, opt, view, config_dir)
    if best is None:
        return []
    return [_rerun_of(best, "fresh_seed", s) for s in _fresh_seeds()]


def budget_constant_configs(study: str, view: ResultsView,
                            config_dir=None) -> list[dict]:
    """AdamW's best config run at constant lr to 2x the budget (decay over
    its own last 20%), one per curve seed; saves the branch checkpoints."""
    best = best_config(study, "adamw", view, config_dir)
    if best is None:
        return []
    return [_rerun_of(best, "budget_constant", s, budget_multiplier=2.0)
            for s in _fresh_seeds()]


def budget_branch_configs(study: str, view: ResultsView,
                          config_dir=None) -> list[dict]:
    """Decay branches at 0.125x/0.25x/0.5x off each curve seed's constant
    run: branch.parent_hash = that constant run's hash, branch.checkpoint_step
    from schedule.checkpoint_step."""
    constants = budget_constant_configs(study, view, config_dir)
    if not constants:
        return []
    base = constants[0]["budget_base_steps"]
    out = []
    for const in constants:
        parent = runner.config_hash(const)
        for m in CHECKPOINT_FRACTIONS:
            out.append(_rerun_of(
                const, "budget_branch", const["seed"],
                budget_multiplier=m,
                branch={"parent_hash": parent,
                        "checkpoint_step": checkpoint_step(base, m)},
            ))
    return out


def budget_halflr_configs(study: str, view: ResultsView,
                          config_dir=None) -> list[dict]:
    """The half-LR 2x guard: seed 1 ALWAYS; seeds 2..n only if seed 1's
    half-LR final val loss beats seed 1's constant-run final (the branched
    2x) — Gate-0 answer 4. wd unchanged (Clarification 5)."""
    constants = budget_constant_configs(study, view, config_dir)
    if not constants:
        return []
    by_seed = {c["seed"]: c for c in constants}
    half1 = _rerun_of(by_seed[1], "budget_halflr", 1, lr=by_seed[1]["lr"] * 0.5)
    out = [half1]
    l_half, l_const = view.loss(half1), view.loss(by_seed[1])
    if l_half is not None and l_const is not None and l_half < l_const:
        for s in _fresh_seeds()[1:]:
            out.append(_rerun_of(by_seed[s], "budget_halflr", s,
                                 lr=by_seed[s]["lr"] * 0.5))
    return out


def init_ablation_configs(study: str, view: ResultsView,
                          config_dir=None) -> list[dict]:
    """Muon's best config with the Stiefel-projected init, seeds 1-3
    (confounder 10; fixed at 3 — Gate-0 answer 6 grows only fresh seeds and
    the budget curve)."""
    best = best_config(study, "muon", view, config_dir)
    if best is None:
        return []
    return [_rerun_of(best, "init_ablation", s, init_projection=True)
            for s in (1, 2, 3)]


def faithful_configs(study: str, view: ResultsView,
                     config_dir=None) -> list[dict]:
    """THREE official-settings reruns, all seed 1 (Gate-0 answer 2): Muon
    with bf16 NS (covers Moonlight), MuonAll at 6 NS steps in bf16, Manifold
    at 100 dual steps. Only the configs whose optimizer finished tuning are
    returned (the per-optimizer faithful phases gate on the fresh phases)."""
    out = []
    muon = best_config(study, "muon", view, config_dir)
    if muon is not None:
        out.append(_rerun_of(muon, "faithful", 1, ns_dtype="bf16"))
    muonall = best_config(study, "muonall", view, config_dir)
    if muonall is not None:
        out.append(_rerun_of(muonall, "faithful", 1, ns_steps=6, ns_dtype="bf16"))
    manifold = best_config(study, "manifold", view, config_dir)
    if manifold is not None:
        out.append(_rerun_of(
            manifold, "faithful", 1,
            manifold=dict(manifold["manifold"], dual_cap=100)))
    return out


# ---------------------------------------------------------------------------
# the DAG

@dataclass(frozen=True)
class Phase:
    """One schedulable unit. ``configs`` is lazy (may read results); a phase
    with an empty config list is complete. ``ready`` (adaptation to design
    §3.10) expresses gates outside this study's phase list — the Phase-II
    donor inheritance; None = always ready."""
    name: str
    needs: tuple[str, ...]
    configs: Callable[[], list[dict]]
    ready: Callable[[], bool] | None = None

    def is_ready(self) -> bool:
        return self.ready is None or bool(self.ready())


def _cd_phase_names(opt: str) -> list[str]:
    names = []
    for p, hp in _cd_spec(opt):
        names.append(f"{opt}_cd_p{p}_{hp}")
        if hp == "lr":
            names.append(f"{opt}_cd_p{p}_lr_edge")
    return names


def _build_dag_phase1(study: str, view: ResultsView, config_dir=None) -> list[Phase]:
    phases = [Phase("pilot", (), lambda: pilot_configs(study, config_dir))]
    phases += _cd_phases(study, view, config_dir)
    phases += _shared_phases(study, view, config_dir)
    return phases


def _cd_phases(study: str, view: ResultsView, config_dir=None) -> list[Phase]:
    """Per optimizer: the coordinate-descent sweep phases (+ lr edge phases)
    then ``{opt}_fresh`` on the CD winner. Needs the study's "pilot" phase."""
    sv = load_study(study, config_dir)
    phases = []
    adamw_end = _cd_phase_names("adamw")[-1]
    for opt in _included(sv):
        first_needs = ("pilot",) if opt not in ("muon", "manifold") \
            else ("pilot", adamw_end)
        prev = None
        for p, hp in _cd_spec(opt):
            key = f"p{p}_{hp}"

            def sweep_cfgs(o=opt, k=key):
                return cd_replay(study, o, view, config_dir).sweeps[k].configs

            name = f"{opt}_cd_{key}"
            phases.append(Phase(name, first_needs if prev is None else (prev,),
                                sweep_cfgs))
            prev = name
            if hp == "lr":
                def edge_cfgs(o=opt, k=key):
                    return cd_replay(study, o, view, config_dir).sweeps[k].edge_configs

                name = f"{opt}_cd_{key}_edge"
                phases.append(Phase(name, (prev,), edge_cfgs))
                prev = name
        phases.append(Phase(
            f"{opt}_fresh", (prev,),
            lambda o=opt: fresh_seed_configs(study, o, view, config_dir)))
    return phases


def _build_dag_full_retune(study: str, view: ResultsView, config_dir=None) -> list[Phase]:
    """FULL_RETUNE_STUDIES: pilot -> CD track (headline, ends in fresh seeds)
    and, in parallel, the inherited Phase-II track (seed 0 only, no fresh
    seeds) -> shared tail on the CD winners."""
    phases = [Phase("pilot", (), lambda: pilot_configs(study, config_dir))]
    phases += _cd_phases(study, view, config_dir)
    phases += _build_dag_phase2(study, view, config_dir, tail=False)
    phases += _shared_phases(study, view, config_dir)
    return phases


def _build_dag_phase2(study: str, view: ResultsView, config_dir=None,
                      tail: bool = True) -> list[Phase]:
    """``tail=False`` (FULL_RETUNE_STUDIES): only the tuning phases — no
    pilot (the caller adds it), no fresh seeds, no shared tail."""
    sv = load_study(study, config_dir)
    phases = []
    if tail and study in PILOT_STUDIES:
        phases.append(Phase("pilot", (), lambda: pilot_configs(study, config_dir)))

    def donor_ready(o):
        return lambda: inherited_values(study, o, view, config_dir) is not None

    for opt in _included(sv):
        needs_a = ("adamw_edge_b",) if opt in ("muon", "manifold") else ()

        def a_cfgs(o=opt):
            st = _stage_cells(study, o, view, config_dir)
            return [] if st is None else [c for _, c in st[0]]

        def e_cfgs(o=opt):
            st = _stage_cells(study, o, view, config_dir)
            return [] if st is None else st[1]

        def b_cfgs(o=opt):
            sb = _stage_b_cells(study, o, view, config_dir)
            return [] if sb is None else [c for _, c in sb[0]]

        def eb_cfgs(o=opt):
            return _stage_b_edge(study, o, view, config_dir)[0]

        phases += [
            Phase(f"{opt}_stage_a", needs_a, a_cfgs, ready=donor_ready(opt)),
            Phase(f"{opt}_edge", (f"{opt}_stage_a",), e_cfgs),
            Phase(f"{opt}_stage_b", (f"{opt}_edge",), b_cfgs),
            Phase(f"{opt}_edge_b", (f"{opt}_stage_b",), eb_cfgs),
        ]
        if tail:
            phases.append(Phase(
                f"{opt}_fresh", (f"{opt}_edge_b",),
                lambda o=opt: fresh_seed_configs(study, o, view, config_dir)))
    if tail:
        phases += _shared_phases(study, view, config_dir)
    return phases


def _shared_phases(study: str, view: ResultsView, config_dir=None) -> list[Phase]:
    """Fresh-seed-gated tail common to both shapes: AdamW budget curve, Muon
    init ablation, per-variant faithful reruns. Chains of excluded
    optimizers (and the init ablation, if muon is excluded) are dropped."""
    included = _included(load_study(study, config_dir))

    def faithful_for(o):
        def cfgs():
            return [c for c in faithful_configs(study, view, config_dir)
                    if c["optimizer"] == o]
        return cfgs

    phases = [
        Phase("budget_constant", ("adamw_fresh",),
              lambda: budget_constant_configs(study, view, config_dir)),
        Phase("budget_branch", ("budget_constant",),
              lambda: budget_branch_configs(study, view, config_dir)),
        Phase("budget_halflr", ("budget_constant",),
              lambda: budget_halflr_configs(study, view, config_dir)),
    ]
    if "muon" in included:
        phases.append(Phase("init_ablation", ("muon_fresh",),
                            lambda: init_ablation_configs(study, view, config_dir)))
    for opt in ("muon", "muonall", "manifold"):
        if opt in included:
            phases.append(Phase(f"{opt}_faithful", (f"{opt}_fresh",),
                                faithful_for(opt)))
    return phases


def build_dag(study: str, view: ResultsView, config_dir=None) -> list[Phase]:
    """The study's phase list in topological order (design §6: the two
    shapes). Rebuilt from the results view every runner pass."""
    if study in PHASE1_STUDIES:
        return _build_dag_phase1(study, view, config_dir)
    if study in FULL_RETUNE_STUDIES:
        return _build_dag_full_retune(study, view, config_dir)
    if study in PHASE2_DONOR:
        return _build_dag_phase2(study, view, config_dir)
    raise ValueError(f"unknown study {study!r}")


# ---------------------------------------------------------------------------
# planning (no results needed) and reporting

def planned_jobs(study: str, config_dir=None) -> list[dict]:
    """The full-plan job list for this study under the PLANNING SCENARIO:
    coordinate descent accepts no change (single pass, early stop), no edge
    trial fires, the half-LR 2x run is not promoted beyond seed 1, and
    Phase-II inheritance keeps the literature defaults. Duplicate cells
    (the incumbent re-appearing in later sweeps) are deduplicated by config
    hash, exactly as the runner would skip them. Used by --dry-run totals and
    scripts/timing.py; NOT by the runner's own scheduling (which is lazy)."""
    sv = load_study(study, config_dir)
    jobs: dict[str, dict] = {}

    def add(cfg):
        jobs.setdefault(runner.config_hash(cfg), cfg)

    for c in pilot_configs(study, config_dir):
        add(c)
    group = {"muon": (float(sv["optimizers"]["adamw"]["lr"]),
                      float(sv["optimizers"]["adamw"]["wd"]))}
    group["manifold"] = group["muon"]

    for opt in _included(sv):
        g = group.get(opt)
        vals = cd_defaults(sv, opt)
        if _has_cd(study):
            for hp in cd_hypers(opt):
                key = _state_key(hp, opt)
                for v in _grid_values(hp, opt, vals, sv):
                    vv = dict(vals)
                    vv[key] = v
                    add(run_config(sv, opt, "cd", vv, seed=0, adamw_group=g,
                                   cd_sweep=_sweep_label(hp, opt)))
        if study not in PHASE1_STUDIES:
            lrs = stage_a_grid(study, opt, config_dir)
            for lr in lrs:
                add(run_config(sv, opt, "stage_a", dict(vals, lr=lr),
                               seed=0, adamw_group=g))
            if opt == "manifold":
                best = vals["lr"]
                for k in QUARTER_OCTAVE_KS:
                    add(run_config(sv, opt, "stage_b",
                                   dict(vals, lr=best * 2.0 ** k),
                                   seed=0, adamw_group=g))
            else:
                d = float(sv["optimizers"][opt]["wd"])
                for lr in lrs[:2]:  # scenario stand-in for the 2 best lrs
                    for f in STAGE_B_WD_FACTORS:
                        add(run_config(sv, opt, "stage_b",
                                       dict(vals, lr=lr, wd=f * d),
                                       seed=0, adamw_group=g))
        winner = run_config(sv, opt, "cd", vals, seed=0, adamw_group=g)
        for s in _fresh_seeds():
            add(_rerun_of(winner, "fresh_seed", s))
        if opt == "muon":
            for s in (1, 2, 3):
                add(_rerun_of(winner, "init_ablation", s, init_projection=True))
            add(_rerun_of(winner, "faithful", 1, ns_dtype="bf16"))
        if opt == "muonall":
            add(_rerun_of(winner, "faithful", 1, ns_steps=6, ns_dtype="bf16"))
        if opt == "manifold":
            add(_rerun_of(winner, "faithful", 1,
                          manifold=dict(winner["manifold"], dual_cap=100)))
        if opt == "adamw":
            base = winner["budget_base_steps"]
            for s in _fresh_seeds():
                const = _rerun_of(winner, "budget_constant", s,
                                  budget_multiplier=2.0)
                add(const)
                parent = runner.config_hash(const)
                for m in CHECKPOINT_FRACTIONS:
                    add(_rerun_of(const, "budget_branch", s,
                                  budget_multiplier=m,
                                  branch={"parent_hash": parent,
                                          "checkpoint_step": checkpoint_step(base, m)}))
            const1 = _rerun_of(winner, "budget_constant", 1, budget_multiplier=2.0)
            add(_rerun_of(const1, "budget_halflr", 1, lr=winner["lr"] * 0.5))
    return list(jobs.values())


def trial_counts(view: ResultsView, study: str, config_dir=None,
                 track: str = "headline") -> dict[str, int]:
    """Completed tuning trials per optimizer, for the report (PLAN Phase II:
    'Report the number of trials each optimizer used'). ``track``:
    "headline" — the trials behind the selected configs (phase cd in studies
    with coordinate descent, else stage_a/stage_b/edge); "inherited" — the
    inherited Phase-II track (stage_a/stage_b/edge) of mlp256/mlp512/moe.
    The two tracks of a FULL_RETUNE study are never summed (DEVIATIONS.md
    #12)."""
    if track == "headline":
        phases = ("cd",) if _has_cd(study) else ("stage_a", "stage_b", "edge")
    elif track == "inherited":
        phases = ("stage_a", "stage_b", "edge")
    else:
        raise ValueError(f"unknown track {track!r}")
    counts = dict.fromkeys(OPTIMIZERS, 0)
    results_dir = Path(view.root) / "results" / study
    if not results_dir.is_dir():
        return counts
    for p in sorted(results_dir.glob("*.json")):
        try:
            res = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        cfg = res.get("config", {})
        if cfg.get("phase") in phases:
            opt = cfg.get("optimizer")
            if opt in counts:
                counts[opt] += 1
    return counts
