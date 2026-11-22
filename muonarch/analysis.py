"""Analysis: DataFrames, budget-curve fits, speedups, seed rule, figures, table.

Design: docs/design.md §3.11.  This module consumes ONLY the on-disk artifacts
(`results/<study>/*.json` and `logs/<study>/*.jsonl`, schemas in design §3.8);
it imports no other muonarch module, so it can run on a laptop that holds just
the result files.

Conventions implemented here (PLAN "Protocol — Metrics" and Decisions log):

* Token speedup (Decisions "Tokens to target"): per curve seed s, the AdamW
  loss-vs-data curve L(D) = alpha * D^-beta + gamma is fitted to 5 points —
  the 0.125x/0.25x/0.5x decay branches, the ordinary 1x fresh-seed run, and
  the 2x point, where the 2x point is min(branched 2x loss, half-LR 2x loss if
  a half-LR run exists for that seed).  The half-LR run executes on seed 1
  first and is promoted to other seeds only if it wins, so other seeds may or
  may not have one on disk; `curve_points` simply uses whatever exists.
  speedup = invert_curve(fit, L_opt) / D_budget, with D_budget the optimizer's
  own 1x token/image budget; the result is the (min, max) range plus the full
  matrix over {fitted curves per curve seed} x {optimizer fresh seeds}.
  Values < 1 mean slower than AdamW.

* FLOP speedup: see `flop_speedup_range` for the precise convention.

* Divergence (+0.5 nats criterion) is analysis-time only (design §9 answer 10);
  training stops runs only on NaN / theoretical-initial-loss violations.

* The seed rule (all paired diffs same sign AND |mean| > 2*max(seed stds)) is a
  3-seed HEURISTIC, not a statistical test; every table or figure that uses it
  says so.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module constants

STUDIES: tuple[str, ...] = ("mlp128", "mlp256", "mlp512", "dense", "moe")
STUDY_A: tuple[str, ...] = ("mlp128", "mlp256", "mlp512")
STUDY_B: tuple[str, ...] = ("dense", "moe")
#: Every optimizer of the comparison.  Lion-Muon ("lionmuon") joined after the
#: protocol freeze (DEVIATIONS.md #15) and is appended LAST, so the order of
#: the five original optimizers — and every table row built from it — is
#: unchanged.  Generic per-optimizer computations iterate over this tuple.
OPTIMIZERS: tuple[str, ...] = ("adamw", "muon", "moonlight", "manifold", "muonall",
                               "lionmuon")
#: The five optimizers of the pre-registered protocol.  The hypotheses H1-H7
#: were written about these and ONLY these: ``hypothesis_verdicts`` never lets
#: a later addition (Lion-Muon) enter a verdict or its text.
PREREGISTERED_OPTIMIZERS: tuple[str, ...] = ("adamw", "muon", "moonlight",
                                             "manifold", "muonall")

#: One fixed colour per optimizer, used by every figure.
OPTIMIZER_COLORS: dict[str, str] = {
    "adamw": "#444444",
    "muon": "#1f77b4",
    "moonlight": "#2ca02c",
    "manifold": "#d62728",
    "muonall": "#9467bd",
    "lionmuon": "#8c564b",
}

#: One fixed marker per optimizer, so coinciding series stay distinguishable.
OPTIMIZER_MARKERS: dict[str, str] = {
    "adamw": "o",
    "muon": "s",
    "moonlight": "^",
    "manifold": "D",
    "muonall": "v",
    "lionmuon": "P",
}

#: figure_5 horizontal dodge between adjacent optimizers, in log2-width units.
_FIG5_DODGE = 0.06

SEED_RULE_CAVEAT = "3-seed sign+2*std seed rule: a heuristic, not a test"

_RESULT_CONFIG_FIELDS = (
    "study", "optimizer", "phase", "seed", "lr", "wd", "momentum",
    "budget_multiplier", "arch", "width", "batch", "seq_len",
    "budget_base_steps", "budget_unit", "rule", "init_projection",
    "cd_sweep", "grad_clip", "adamw_betas", "edge_trial",
)

#: Phases that count as tuning trials (Phase-I coordinate descent, Phase-II
#: Stage A/B, lr-grid edge trials).  Pilot, budget-curve, init-ablation,
#: faithful and smoke runs are not trials.
TUNING_PHASES: tuple[str, ...] = ("cd", "stage_a", "stage_b", "edge")
PHASE2_TUNING_PHASES: tuple[str, ...] = ("stage_a", "stage_b", "edge")


def _has_cd_rows(df: pd.DataFrame, study: str) -> bool:
    return bool(((df["study"] == study) & (df["phase"] == "cd")).any())


def headline_tuning_phases(df: pd.DataFrame, study: str) -> tuple[str, ...]:
    """The tuning phases behind the study's headline (selected) configs: the
    coordinate descent ("cd") wherever the study ran one — mlp128/dense, and
    mlp256/mlp512 since the Gate-3 MLP protocol change (DEVIATIONS.md #12),
    which ALSO keep an inherited Phase-II track (stage_a/stage_b/edge) at
    seed 0 for H4 only — else the Phase-II phases. Headline quantities
    (basin, divergence rate, figures) never mix the two tracks."""
    return ("cd",) if _has_cd_rows(df, study) else PHASE2_TUNING_PHASES
#: Top-level ``logs/`` subdirectories that load_logs never reads.
_EXCLUDED_LOG_DIRS: frozenset[str] = frozenset({"diagnostics"})

_RESULT_TOP_FIELDS = (
    "hash", "status", "diverged_at_step", "final_val_loss", "final_val_acc",
    "final_test_loss", "final_test_acc", "train_val_gap", "dual_steps_mean",
    "msign_calls_mean", "edge_rule_fired", "device", "precision",
    "wall_clock_s",
)
#: Lion-Muon's two tuned extra hyperparameters, flattened out of the run
#: config's ``lionmuon`` dict ({"period", "alpha", "beta1", "beta2"}; only
#: lionmuon configs carry it) into scalar columns.  NaN for every other
#: optimizer.  alpha = Muon-step lr / sign-step lr; beta2 = the EMA decay.
LIONMUON_COLS: tuple[str, ...] = ("lionmuon_alpha", "lionmuon_beta2")


def _lionmuon_fields(cfg: dict) -> dict:
    lm = cfg.get("lionmuon")
    lm = lm if isinstance(lm, dict) else {}
    return {"lionmuon_alpha": lm.get("alpha"), "lionmuon_beta2": lm.get("beta2")}


_NUMERIC_COLS = (
    "lionmuon_alpha", "lionmuon_beta2",
    "seed", "lr", "wd", "momentum", "budget_multiplier", "width", "batch",
    "seq_len", "budget_base_steps", "final_val_loss", "final_val_acc",
    "final_test_loss", "final_test_acc", "train_val_gap", "dual_steps_mean",
    "msign_calls_mean", "wall_clock_s", "flops_model", "flops_optimizer",
    "flops_total", "diverged_at_step",
)


# ---------------------------------------------------------------------------
# Loading

def load_results(root: Path | str) -> pd.DataFrame:
    """One row per ``results/<study>/<hash>.json``, config fields flattened.

    Flattened columns: the config fields listed in ``_RESULT_CONFIG_FIELDS``,
    Lion-Muon's ``lionmuon_alpha`` / ``lionmuon_beta2`` (``LIONMUON_COLS``;
    NaN for the other optimizers), the top-level result fields, and
    ``flops_model/optimizer/total``.
    Missing values become NaN/None.  ``df.attrs["root"]`` records the root.
    """
    root = Path(root)
    rows = []
    for p in sorted((root / "results").rglob("*.json")):
        res = json.loads(p.read_text())
        cfg = res.get("config", {})
        row = {k: cfg.get(k) for k in _RESULT_CONFIG_FIELDS}
        row.update(_lionmuon_fields(cfg))
        row.update({k: res.get(k) for k in _RESULT_TOP_FIELDS})
        flops = res.get("flops") or {}
        row["flops_model"] = flops.get("model")
        row["flops_optimizer"] = flops.get("optimizer")
        row["flops_total"] = flops.get("total")
        row["path"] = str(p)
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.attrs["root"] = str(root)
    return df


def load_logs(root: Path | str) -> pd.DataFrame:
    """One row per JSONL eval/final record from ``logs/<study>/*.jsonl``.

    Each row carries the run's flattened config fields (study, optimizer,
    phase, seed, lr, budget_multiplier, ...) taken from the file's ``start``
    record, plus the record's own fields.  The eval record's instantaneous
    learning rate is renamed to ``lr_step`` so it does not clobber the config
    ``lr``.  ``final`` records keep ``spectral_norms`` as a dict cell.

    Only TOP-LEVEL study directories are read: ``logs/<study>/*.jsonl``.
    Nothing nested deeper is loaded, and ``logs/diagnostics/`` is skipped
    entirely — it holds parked, unused runs (e.g. the stopped seed-4/5
    fresh_seed runs in ``diagnostics/discarded_seeds45`` and the Manifold
    audit runs, phase "cd" seed 0, in ``diagnostics/manifold_audit``) that
    must never enter any analysis.
    """
    root = Path(root)
    rows = []
    for p in sorted((root / "logs").glob("*/*.jsonl")):
        if p.parent.name in _EXCLUDED_LOG_DIRS:
            continue
        base: dict = {"hash": None}
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = rec.get("t")
            if t == "start":
                cfg = rec.get("config", {})
                base = {k: cfg.get(k) for k in _RESULT_CONFIG_FIELDS}
                base.update(_lionmuon_fields(cfg))
                base["hash"] = rec.get("hash")
                continue
            if t not in ("eval", "final"):
                continue
            row = dict(base)
            for k, v in rec.items():
                if k == "t":
                    continue
                row["lr_step" if (t == "eval" and k == "lr") else k] = v
            row["t"] = t
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in ("seed", "step", "images_or_tokens", "val_loss", "train_loss",
              "max_attn_logit", "expert_entropy", "budget_multiplier",
              "budget_base_steps", "batch", "seq_len", "width",
              *LIONMUON_COLS):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.attrs["root"] = str(root)
    return df


# ---------------------------------------------------------------------------
# Budget-curve fit

#: beta search bounds for fit_budget_curve; a fitted beta pinned to either
#: bound means the data carry no usable curvature (e.g. flat losses) and the
#: fit is treated as degenerate by invert_curve.
BETA_SCAN_LO: float = 1e-3
BETA_SCAN_HI: float = 3.0


@dataclass
class FitResult:
    alpha: float
    beta: float
    gamma: float

    def __call__(self, D: float) -> float:
        return self.alpha * D ** -self.beta + self.gamma


def _lstsq_for_beta(D: np.ndarray, L: np.ndarray, beta: float):
    """For fixed beta the model is linear in (alpha, gamma): solve + SSE."""
    X = np.column_stack([D ** -beta, np.ones_like(D)])
    coef, *_ = np.linalg.lstsq(X, L, rcond=None)
    sse = float(np.sum((X @ coef - L) ** 2))
    return sse, float(coef[0]), float(coef[1])


def fit_budget_curve(points: list[tuple[float, float]]) -> FitResult:
    """Least-squares fit of L(D) = alpha * D^-beta + gamma to one seed's points.

    Five (D, L) points, three parameters (Decisions "Tokens to target").
    numpy-only: for each candidate beta the model is linear in (alpha, gamma),
    so beta is found by a log-spaced scan followed by ternary refinement, with
    alpha and gamma from linear least squares at each beta.
    """
    pts = sorted(points)
    D = np.asarray([p[0] for p in pts], dtype=float)
    L = np.asarray([p[1] for p in pts], dtype=float)
    if len(pts) < 3:
        raise ValueError("need at least 3 (D, L) points for a 3-parameter fit")
    betas = np.geomspace(BETA_SCAN_LO, BETA_SCAN_HI, 600)
    sses = [_lstsq_for_beta(D, L, b)[0] for b in betas]
    i = int(np.argmin(sses))
    lo = math.log(betas[max(i - 1, 0)])
    hi = math.log(betas[min(i + 1, len(betas) - 1)])
    for _ in range(100):  # ternary search on log(beta)
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if _lstsq_for_beta(D, L, math.exp(m1))[0] <= _lstsq_for_beta(D, L, math.exp(m2))[0]:
            hi = m2
        else:
            lo = m1
    beta = math.exp((lo + hi) / 2)
    _, alpha, gamma = _lstsq_for_beta(D, L, beta)
    return FitResult(alpha=alpha, beta=beta, gamma=gamma)


def invert_curve(fit: FitResult, loss: float) -> float:
    """D such that fit(D) == loss, i.e. AdamW's data to reach ``loss``.

    If ``loss <= gamma`` the fitted curve never reaches it at any finite
    budget, so the function returns ``inf`` (the optimizer beat AdamW's fitted
    asymptote; the speedup is unbounded and reported as inf).  A degenerate
    fit also returns inf: alpha or beta <= 0, beta pinned to the fit's scan
    boundary (``BETA_SCAN_LO``/``BETA_SCAN_HI`` — what a flat set of points
    produces), or an inversion that overflows a float.  Downstream consumers
    (figure_1, main_table, the verdict table) must therefore tolerate inf
    cells; they do.
    """
    if (loss <= fit.gamma or fit.alpha <= 0 or fit.beta <= 0
            or fit.beta <= BETA_SCAN_LO * 1.01 or fit.beta >= BETA_SCAN_HI * 0.99):
        return float("inf")
    try:
        return float((fit.alpha / (loss - fit.gamma)) ** (1.0 / fit.beta))
    except OverflowError:
        return float("inf")


# ---------------------------------------------------------------------------
# Point assembly and speedups

def _unit_size(rows: pd.DataFrame) -> float:
    """Data units per 1x budget: base_steps * batch * (seq_len or 1)."""
    r = rows.iloc[0]
    seq = r.get("seq_len")
    seq = 1.0 if seq is None or (isinstance(seq, float) and math.isnan(seq)) else float(seq)
    return float(r["budget_base_steps"]) * float(r["batch"]) * seq


def curve_points(study: str, seed: int, df: pd.DataFrame) -> list[tuple[float, float]]:
    """The 5 (D, L) AdamW budget-curve points for one seed.

    Implements the Decisions-log assembly: the three decay branches
    (0.125x/0.25x/0.5x), the ordinary 1x fresh-seed run, and the 2x point,
    where 2x = min(branched 2x loss from the constant run's own decay,
    half-LR 2x loss if a half-LR run exists for this seed).  The seed-1-first
    promotion rule (design §9 answer 4) means seeds other than 1 may lack a
    half-LR run; whatever exists is used.

    An incomplete assembly (fewer or more than 5 points, e.g. a missing
    branch) is reported with ``warnings.warn`` and whatever exists is
    returned; only fewer than 3 points — too few for the 3-parameter fit —
    raises ValueError.
    """
    a = df[(df["study"] == study) & (df["optimizer"] == "adamw")
           & (df["status"] == "completed") & (df["seed"] == seed)]
    if a.empty:
        raise ValueError(f"no completed AdamW runs for {study} seed {seed}")
    d1 = _unit_size(a)
    pts: list[tuple[float, float]] = []
    br = a[a["phase"] == "budget_branch"]
    for m, g in br.groupby("budget_multiplier"):
        pts.append((float(m) * d1, float(g["final_val_loss"].min())))
    one = a[(a["phase"] == "fresh_seed") & (a["budget_multiplier"] == 1.0)]
    if not one.empty:
        pts.append((d1, float(one["final_val_loss"].iloc[0])))
    const = a[a["phase"] == "budget_constant"]
    half = a[a["phase"] == "budget_halflr"]
    if not const.empty or not half.empty:
        cands = []
        if not const.empty:
            cands.append(float(const["final_val_loss"].iloc[0]))
        if not half.empty:
            cands.append(float(half["final_val_loss"].min()))
        pts.append((2.0 * d1, min(cands)))
    if len(pts) < 3:
        raise ValueError(
            f"only {len(pts)} budget-curve points for {study} seed {seed}; "
            "need at least 3 for the 3-parameter fit")
    if len(pts) != 5:
        warnings.warn(
            f"budget curve for {study} seed {seed} has {len(pts)} points "
            "instead of the expected 5 (branches 0.125/0.25/0.5, 1x fresh, "
            "2x); fitting what exists", stacklevel=2)
    return sorted(pts)


def _curve_seeds(study: str, df: pd.DataFrame) -> list[int]:
    rows = df[(df["study"] == study) & (df["optimizer"] == "adamw")
              & (df["phase"] == "budget_constant") & (df["status"] == "completed")]
    return sorted(int(s) for s in rows["seed"].dropna().unique())


def _fresh(df: pd.DataFrame, study: str, opt: str | None = None) -> pd.DataFrame:
    m = ((df["study"] == study) & (df["phase"] == "fresh_seed")
         & (df["status"] == "completed"))
    if "budget_multiplier" in df.columns:
        m &= df["budget_multiplier"] == 1.0
    if opt is not None:
        m &= df["optimizer"] == opt
    return df[m]


_SPEEDUP_COLS = ["speedup_min", "speedup_max", "matrix"]


def _speedup_frame(study: str, df: pd.DataFrame, kind: str) -> pd.DataFrame:
    seeds = _curve_seeds(study, df)
    if not seeds:
        return pd.DataFrame(columns=_SPEEDUP_COLS).rename_axis("optimizer")
    fits = {s: fit_budget_curve(curve_points(study, s, df)) for s in seeds}
    adamw = _fresh(df, study, "adamw")
    d1 = _unit_size(adamw)
    # AdamW total-FLOPs-per-unit rate per curve seed, from its 1x fresh run
    rate = {}
    for s in seeds:
        r = adamw[adamw["seed"] == s]
        if not r.empty:
            rate[s] = float(r["flops_total"].iloc[0]) / d1
    records = []
    for opt in [o for o in OPTIMIZERS if not _fresh(df, study, o).empty]:
        fr = _fresh(df, study, opt).sort_values("seed")
        losses = fr["final_val_loss"].to_numpy(dtype=float)
        flops_opt = fr["flops_total"].to_numpy(dtype=float)
        mat = np.empty((len(seeds), len(losses)))
        for i, s in enumerate(seeds):
            for j, L in enumerate(losses):
                d_eq = invert_curve(fits[s], L)
                if kind == "tokens":
                    mat[i, j] = d_eq / d1
                else:
                    mat[i, j] = d_eq * rate[s] / flops_opt[j]
        records.append({"optimizer": opt, "speedup_min": float(np.nanmin(mat)),
                        "speedup_max": float(np.nanmax(mat)), "matrix": mat})
    return pd.DataFrame(records).set_index("optimizer")


def token_speedup_range(study: str, df: pd.DataFrame) -> pd.DataFrame:
    """Token speedup vs AdamW per optimizer: (min, max) range + full matrix.

    ``matrix[i, j] = invert_curve(fit_i, L_opt_seed_j) / D_budget`` over the
    {fitted curves per curve seed} x {optimizer fresh seeds} grid, where
    ``D_budget`` is the optimizer's own 1x token/image budget (same for every
    optimizer in a study).  Values < 1 mean slower than AdamW; inf means the
    optimizer's loss lies below the fitted AdamW asymptote.
    Empty DataFrame if the study has no budget-curve runs on disk.
    """
    return _speedup_frame(study, df, "tokens")


def flop_speedup_range(study: str, df: pd.DataFrame) -> pd.DataFrame:
    """FLOP speedup vs AdamW per optimizer: (min, max) range + full matrix.

    Convention (documented here, used nowhere else):

    ``matrix[i, j] = F_AdamW(D_eq) / F_opt(seed_j)`` where

    * ``D_eq = invert_curve(fit_i, L_opt_seed_j)`` is the AdamW tokens/images
      needed to reach the optimizer's final loss, from curve seed i's fit;
    * ``F_AdamW(D_eq) = D_eq * rate_i`` converts AdamW tokens to AdamW total
      training FLOPs at AdamW's per-token rate ``rate_i = flops.total of the
      AdamW 1x fresh-seed run on curve seed i / D_1x`` — the TOTAL rate, i.e.
      model forward+backward plus AdamW's own (negligible) optimizer FLOPs,
      assumed linear in tokens (per-step cost is constant over a run);
    * ``F_opt(seed_j)`` is the optimizer run's own ``flops.total`` from its 1x
      fresh-seed result JSON — model plus its optimizer FLOPs (NS/msign at the
      iteration counts the run actually performed; Manifold Muon at its logged
      msign calls).

    Values < 1 mean the optimizer spends more total FLOPs than AdamW needs to
    reach the same loss.
    """
    return _speedup_frame(study, df, "flops")


_EXTRAP_COLS = ["max_budget", "n_extrapolated", "n_cells", "extrapolated"]


def speedup_extrapolation(study: str, df: pd.DataFrame) -> pd.DataFrame:
    """Which speedups are EXTRAPOLATIONS of the fitted AdamW curve.

    A cell of the {curve seed} x {optimizer seed} token-speedup matrix is an
    extrapolation when its AdamW-equivalent data exceeds the largest budget
    AdamW was actually run at on that curve seed (the largest D among that
    seed's ``curve_points`` — read from the data, 2x in the frozen protocol),
    or is inf (below the fitted asymptote).  The FLOP speedup is derived from
    the same equivalent data, so the flag covers it too.

    Returns a frame indexed by optimizer: ``max_budget`` (largest measured
    AdamW budget as a multiple of 1x; the smallest over curve seeds),
    ``n_extrapolated``, ``n_cells`` and ``extrapolated`` (any cell).  Empty
    without budget-curve runs.
    """
    seeds = _curve_seeds(study, df)
    tok = token_speedup_range(study, df)
    if not seeds or tok.empty:
        return pd.DataFrame(columns=_EXTRAP_COLS).rename_axis("optimizer")
    d1 = _unit_size(_fresh(df, study, "adamw"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        top = np.array([max(d for d, _ in curve_points(study, s, df)) / d1
                        for s in seeds])
    records = []
    for opt in tok.index:
        mat = np.asarray(tok.loc[opt, "matrix"], dtype=float)
        beyond = ~np.isfinite(mat) | (mat > top[:, None] * (1 + 1e-9))
        records.append({"optimizer": opt, "max_budget": float(top.min()),
                        "n_extrapolated": int(beyond.sum()),
                        "n_cells": int(mat.size),
                        "extrapolated": bool(beyond.any())})
    return pd.DataFrame(records).set_index("optimizer")[_EXTRAP_COLS]


# ---------------------------------------------------------------------------
# Scalar metrics

#: The non-lr tuned hypers that define a coordinate-descent context.  The two
#: Lion-Muon columns are NaN (-> None) on every other optimizer's rows, so
#: they never change those optimizers' contexts.
_CD_OTHER_HYPERS = ("wd", "momentum", "grad_clip", "adamw_betas") + LIONMUON_COLS


def _hyper_norm(v):
    """Hashable/comparable form of a hyper value (lists -> tuples, NaN -> None)."""
    if isinstance(v, (list, tuple)):
        return tuple(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _hyper_key(v):
    """``_hyper_norm`` with floats rounded to 10 significant digits, so the
    same value written twice (0.3 vs 0.30000000000000004) compares equal."""
    v = _hyper_norm(v)
    if isinstance(v, tuple):
        return tuple(_hyper_key(x) for x in v)
    if isinstance(v, (float, np.floating)):
        return float(f"{float(v):.10g}")
    return v


def _lr_sweep_rows(df: pd.DataFrame, study: str, opt: str,
                   track: str = "headline") -> pd.DataFrame:
    """Rows forming the optimizer's learning-rate sweep in a study.

    ``track``: "headline" (default) — the sweep behind the study's selected
    configs: the cd lr sweep wherever the study has cd rows, else Stage A;
    "inherited" — the Stage-A grid of the inherited Phase-II track (mlp256 /
    mlp512 / moe; DEVIATIONS.md #12), for H4's inherited-vs-re-tuned
    comparison. The two tracks are never mixed.

    Phase-II studies tune via the 12-trial recipe, so the lr sweep is the
    Stage-A grid (``phase == "stage_a"``).  Phase-I studies (mlp128, dense)
    tune via coordinate descent instead, so their lr sweep lives in
    ``phase == "cd"`` rows (design §4): the cells where only lr varies while
    the other swept hypers sit at the FINAL accepted values.

    Selection of the cd lr-sweep cells:

    1. Context = the ACCEPTED values of the non-lr tuned hypers (wd,
       momentum, grad_clip, adamw_betas — and Lion-Muon's alpha and beta2 —
       where present).  These are read off
       the optimizer's completed fresh-seed runs, which are run at exactly
       the accepted config — so the context is the accepted one whatever
       ``cd_sweep`` labels the rows carry.  (Config dedup gives a re-run cell
       its FIRST label: the pass-2 lr sweep's centre cell was first run
       inside the wd / momentum sweep and stays labeled "wd" / "momentum" on
       disk, and it usually BEATS every lr-labeled cell of pass 1, so the
       labels alone cannot find the context.)
    2. Fallback, only when there are no fresh-seed runs yet (or none of the
       cd rows sits at their config): among the contexts that contain an
       lr-labeled row (every context if the rows carry no labels), the one
       containing the best (lowest final val loss) completed cd row of ANY
       label.
    3. Membership: every cd row — whatever its ``cd_sweep`` label — whose
       non-lr hypers match that context.  This re-admits the deduped centre
       cell and excludes lr cells run under an earlier context (e.g. the
       pass-1 sweep at the pre-acceptance wd).

    ``muonarch.figures.lr_sweep`` is a thin view of this function; there is
    no second implementation.

    Returns both completed and diverged rows (figure_2 marks the diverged
    ones); callers that need losses filter on status themselves.
    """
    rows = df[(df["study"] == study) & (df["optimizer"] == opt)]
    stage = rows[rows["phase"] == "stage_a"]
    if track == "inherited":
        return stage
    if track != "headline":
        raise ValueError(f"unknown track {track!r}")
    cd = rows[rows["phase"] == "cd"]
    if not _has_cd_rows(df, study):
        return stage
    if not cd.empty:
        keys = [k for k in _CD_OTHER_HYPERS if k in cd.columns]

        def _key(r):
            return tuple(_hyper_key(r.get(k)) for k in keys)

        cd_keys = cd.apply(_key, axis=1)
        ctx = None
        fresh = _fresh(df, study, opt)
        if not fresh.empty:
            fresh_key = _key(fresh.iloc[0])
            if (cd_keys == fresh_key).any():
                ctx = fresh_key
        if ctx is None:
            if "cd_sweep" in cd.columns and cd["cd_sweep"].notna().any():
                swept = set(cd_keys[cd["cd_sweep"] == "lr"])
            else:
                swept = set(cd_keys)
            pool = cd[cd_keys.isin(swept) & (cd["status"] == "completed")]
            pool = pool.dropna(subset=["final_val_loss"])
            if pool.empty:
                return cd.iloc[0:0]
            ctx = cd_keys[pool["final_val_loss"].idxmin()]
        cd = cd[cd_keys == ctx]
    return cd


def inherited_best_lr(df: pd.DataFrame, study: str, opt: str) -> float | None:
    """lr of the min seed-0 final-loss trial of the INHERITED Phase-II track
    (stage_a/stage_b/edge) — H4's inherited optimum in mlp256/mlp512, to set
    against the fully re-tuned optimum (``_best_lr``). None if no completed
    trials."""
    rows = df[(df["study"] == study) & (df["optimizer"] == opt)
              & df["phase"].isin(PHASE2_TUNING_PHASES)
              & (df["status"] == "completed")].dropna(subset=["final_val_loss"])
    if rows.empty:
        return None
    return float(rows.loc[rows["final_val_loss"].idxmin(), "lr"])


def lr_basin_width(study: str, opt: str, df: pd.DataFrame, tol: float,
                   track: str = "headline") -> float:
    """Width, in factors of 2, of the lr-sweep range within ``tol`` of best.

    The sweep rows are Stage-A trials, or the Phase-I coordinate-descent
    lr-sweep cells for cd-tuned studies (see ``_lr_sweep_rows``).  Tolerance
    semantics follow design §9 answer 9: for the LM studies (dense/moe)
    ``tol`` is ABSOLUTE nats (0.02); for the MLP studies it is RELATIVE final
    validation loss (0.01 = 1%).  The study name decides which (``mlp*`` ->
    relative).  Diverged trials are excluded.

    The width is CONTIGUOUS around the best lr: the grid cells are sorted by
    lr and the maximal contiguous run of within-tolerance cells containing
    the best lr is taken; its log2(max lr / min lr) is returned.  A lucky
    far-away cell separated from the best by out-of-tolerance cells does not
    widen the basin (it is not a basin).  Returns NaN if no completed sweep
    trials exist; 0.0 if only the best lr itself is within tolerance.
    """
    rows = _lr_sweep_rows(df, study, opt, track)
    rows = rows[rows["status"] == "completed"].dropna(subset=["final_val_loss", "lr"])
    if rows.empty:
        return float("nan")
    cells = rows.groupby("lr")["final_val_loss"].min().sort_index()
    losses = cells.to_numpy(dtype=float)
    lrs = cells.index.to_numpy(dtype=float)
    best_i = int(np.argmin(losses))
    best = losses[best_i]
    thr = best * (1.0 + tol) if study.startswith("mlp") else best + tol
    lo = best_i
    while lo - 1 >= 0 and losses[lo - 1] <= thr:
        lo -= 1
    hi = best_i
    while hi + 1 < len(losses) and losses[hi + 1] <= thr:
        hi += 1
    return float(math.log2(lrs[hi] / lrs[lo]))


def divergence_rate(study: str, opt: str, df: pd.DataFrame) -> float:
    """Share of the optimizer's trials in the study counted as diverged.

    A trial counts if ``status == "diverged"`` OR its final validation loss
    exceeds (the optimizer's best final loss among those trials + 0.5 nats).
    The +0.5-nats criterion is analysis-time only (design §9 answer 10): it
    never stopped a run.

    "Trials" means the study's HEADLINE tuning trials
    (``headline_tuning_phases``: cd, or stage_a / stage_b / edge — never both
    tracks, DEVIATIONS.md #12) plus the fresh-seed 1x runs — both numerator
    and denominator.
    Budget-curve runs (branches, constant 2x, half-LR), pilot, init-ablation,
    faithful and smoke runs are excluded entirely: a 0.125x decay branch ends
    ~1 nat above the 2x loss BY DESIGN, and counting it against the +0.5-nats
    criterion would read AdamW's own curve as divergence.
    """
    rows = df[(df["study"] == study) & (df["optimizer"] == opt)
              & df["phase"].isin(headline_tuning_phases(df, study) + ("fresh_seed",))]
    if rows.empty:
        return float("nan")
    comp = rows[rows["status"] == "completed"]["final_val_loss"].dropna()
    best = float(comp.min()) if not comp.empty else float("inf")
    bad = (rows["status"] == "diverged") | (rows["final_val_loss"] > best + 0.5)
    return float(bad.sum()) / float(len(rows))


def seed_rule(diffs: np.ndarray, stds: tuple[float, float]) -> bool:
    """The "a difference is real" HEURISTIC (PLAN Protocol — Metrics).

    True iff all paired per-seed differences share one sign AND the absolute
    mean difference exceeds twice the larger of the two seed stds.  With 3
    seeds this is a heuristic, not a statistical test; every table or figure
    that uses it must label it as such (``SEED_RULE_CAVEAT``).
    """
    d = np.asarray(diffs, dtype=float)
    if d.size == 0 or np.any(np.isnan(d)):
        return False
    same_sign = bool(np.all(d > 0) or np.all(d < 0))
    return bool(same_sign and abs(d.mean()) > 2.0 * max(stds))


def _loss_by_seed(df: pd.DataFrame, study: str, opt: str) -> dict[int, float]:
    fr = _fresh(df, study, opt)
    return {int(s): float(l) for s, l in zip(fr["seed"], fr["final_val_loss"])}


def h3_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """H3 interaction (Decisions log "H3 test"): per variant, seed-paired
    (gap on MoE - gap on dense), gap = L_variant - L_adamw.

    Returns a DataFrame indexed by variant with ``mean_gap_diff``,
    ``std_gap_diff`` (ddof=1), ``n_seeds`` and ``real`` — real iff at least
    2 paired seeds exist AND |mean| > 2 * seed std of the difference (if the
    std is exactly 0, real iff the mean is nonzero).  A single paired seed
    has no spread estimate, so it is never "real".  Variants missing on
    either architecture are omitted.
    """
    a_dense = _loss_by_seed(df, "dense", "adamw")
    a_moe = _loss_by_seed(df, "moe", "adamw")
    records = []
    for opt in OPTIMIZERS:
        if opt == "adamw":
            continue
        v_dense = _loss_by_seed(df, "dense", opt)
        v_moe = _loss_by_seed(df, "moe", opt)
        seeds = sorted(set(a_dense) & set(a_moe) & set(v_dense) & set(v_moe))
        if not seeds:
            continue
        d = np.array([(v_moe[s] - a_moe[s]) - (v_dense[s] - a_dense[s]) for s in seeds])
        std = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        mean = float(d.mean())
        if len(d) < 2:
            real = False  # no spread estimate from a single paired seed
        elif std > 0:
            real = abs(mean) > 2 * std
        else:
            real = mean != 0.0
        records.append({"optimizer": opt, "mean_gap_diff": mean,
                        "std_gap_diff": std, "n_seeds": len(seeds), "real": real})
    return pd.DataFrame(records).set_index("optimizer") if records else pd.DataFrame(
        columns=["mean_gap_diff", "std_gap_diff", "n_seeds", "real"]).rename_axis("optimizer")


def _paired_vs(df: pd.DataFrame, study: str, opt_a: str, opt_b: str):
    """Paired per-seed diffs (a - b) and the two seed stds, or None."""
    la, lb = _loss_by_seed(df, study, opt_a), _loss_by_seed(df, study, opt_b)
    seeds = sorted(set(la) & set(lb))
    if not seeds:
        return None
    diffs = np.array([la[s] - lb[s] for s in seeds])
    sa = np.array([la[s] for s in seeds]).std(ddof=1) if len(seeds) > 1 else 0.0
    sb = np.array([lb[s] for s in seeds]).std(ddof=1) if len(seeds) > 1 else 0.0
    return diffs, (float(sa), float(sb))


def null_result_rows(df: pd.DataFrame) -> pd.DataFrame:
    """"Indistinguishable after equal tuning" rows (Decisions log "Null
    results"): per study, each variant vs AdamW; where the seed rule FAILS the
    comparison is reported as a null result together with the measured seed
    spread, so the reader sees how large an effect could have been detected.
    """
    records = []
    for study in [s for s in STUDIES if s in set(df["study"])]:
        for opt in OPTIMIZERS:
            if opt == "adamw":
                continue
            pair = _paired_vs(df, study, opt, "adamw")
            if pair is None:
                continue
            diffs, stds = pair
            if not seed_rule(diffs, stds):
                records.append({
                    "study": study, "optimizer": opt, "vs": "adamw",
                    "mean_diff": float(diffs.mean()),
                    "seed_spread": float(max(stds)),
                    "verdict": "indistinguishable after equal tuning",
                    "caveat": SEED_RULE_CAVEAT,
                })
    return pd.DataFrame(records, columns=["study", "optimizer", "vs", "mean_diff",
                                          "seed_spread", "verdict", "caveat"])


def edge_rule_report(df: pd.DataFrame) -> str:
    """How often the lr-grid edge rule fired, per study x optimizer.

    Edge-extension trials are identified by the logged-only config field
    ``edge_trial`` (set by studies.py on EVERY edge extension: Phase-II
    trials carry phase "edge", while Phase-I lr-edge extensions carry phase
    "cd", so the phase alone misses them).  Rows without the field fall back
    to ``phase == "edge"``.  Counts are reported per optimizer x study.
    """
    fired = (df[df["edge_rule_fired"] == True]  # noqa: E712 (may be object col)
             if "edge_rule_fired" in df.columns else df.iloc[0:0])
    if "edge_trial" in df.columns:
        has_field = df["edge_trial"].notna()
        truthy = df["edge_trial"].fillna(False).astype(bool)
    else:
        has_field = pd.Series(False, index=df.index)
        truthy = pd.Series(False, index=df.index)
    phase_edge = (df["phase"] == "edge") if "phase" in df.columns else pd.Series(
        False, index=df.index)
    extra = df[truthy | (~has_field & phase_edge)]
    lines = [f"Edge rule fired {len(fired)} time(s); "
             f"{len(extra)} extra edge trial(s) run."]
    for (study, opt), g in fired.groupby(["study", "optimizer"]):
        lines.append(f"  fired — {study}/{opt}: {len(g)}")
    for (study, opt), g in extra.groupby(["study", "optimizer"]):
        lines.append(f"  edge trials — {study}/{opt}: {len(g)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Log selection (headline fresh-seed runs only)

def _completed_fresh_hashes(df: pd.DataFrame | None) -> set | None:
    """Hashes of completed fresh-seed 1x results, or None when no usable
    results frame is given (no filtering possible)."""
    if df is None or df.empty or not {"hash", "study"} <= set(df.columns):
        return None
    m = (df["phase"] == "fresh_seed") & (df["status"] == "completed")
    if "budget_multiplier" in df.columns:
        m &= df["budget_multiplier"] == 1.0
    return set(df.loc[m, "hash"].dropna())


def _fresh_logs(logs: pd.DataFrame | None, t: str | None = None,
                df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Log rows of the headline fresh-seed 1x runs (phase "fresh_seed",
    budget_multiplier 1 where the column exists), optionally one record
    type ``t`` ("eval"/"final").  Smoke, pilot, tuning, budget-curve,
    faithful and init-ablation runs are excluded: diagnostics and H7 describe
    the tuned headline runs only.

    Defence in depth: when a results frame ``df`` is given, only logs whose
    hash belongs to a COMPLETED fresh-seed 1x result are kept, so a stopped
    or partial fresh_seed run (no completed result JSON) never counts as a
    seed."""
    if logs is None or logs.empty or "phase" not in logs.columns:
        return pd.DataFrame()
    m = logs["phase"] == "fresh_seed"
    keep = _completed_fresh_hashes(df)
    if keep is not None:
        m &= logs["hash"].isin(keep) if "hash" in logs.columns else False
    if "budget_multiplier" in logs.columns:
        m &= logs["budget_multiplier"] == 1.0
    if t is not None:
        m &= logs["t"] == t
    return logs[m]


def _eval_curves(logs: pd.DataFrame | None, metric: str,
                 studies: tuple[str, ...],
                 df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (study, optimizer, images_or_tokens): ``metric`` averaged over the
    fresh seeds.  Columns: study, optimizer, images_or_tokens, <metric>,
    n_seeds."""
    cols = ["study", "optimizer", "images_or_tokens", metric, "n_seeds"]
    ev = _fresh_logs(logs, "eval", df)
    if ev.empty or metric not in ev.columns:
        return pd.DataFrame(columns=cols)
    ev = ev[ev["study"].isin(studies) & ev[metric].notna()]
    if ev.empty:
        return pd.DataFrame(columns=cols)
    out = (ev.groupby(["study", "optimizer", "images_or_tokens"])
           .agg(**{metric: (metric, "mean"), "n_seeds": ("seed", "nunique")})
           .reset_index())
    return out[cols]


def attn_logit_curves(logs: pd.DataFrame | None,
                      df: pd.DataFrame | None = None) -> pd.DataFrame:
    """figure_8 (a) data: fresh-seed max attention logit, mean over seeds,
    per study (dense, moe) x optimizer x token count.  With ``df``, only
    runs with a completed fresh-seed result count (``_fresh_logs``)."""
    return _eval_curves(logs, "max_attn_logit", STUDY_B, df)


def expert_entropy_curves(logs: pd.DataFrame | None,
                          df: pd.DataFrame | None = None) -> pd.DataFrame:
    """figure_8 (b) data: fresh-seed MoE expert-share entropy, mean over
    seeds, per optimizer x token count.  ``df`` as in attn_logit_curves."""
    return _eval_curves(logs, "expert_entropy", ("moe",), df)


_MODULE_ORDER = ("q", "k", "v", "o", "gate", "up", "down")


def short_layer_name(name: str) -> str:
    """'model.layers.3.mlp.gate_proj.weight' -> 'L3.gate';
    'model.layers.0.self_attn.q_proj.weight' -> 'L0.q';
    'model.layers.0.mlp.experts.10.up_proj.weight' -> 'L0.e10.up';
    'model.embed_tokens.weight' -> 'embed'."""
    s = name
    if s.endswith(".weight"):
        s = s[: -len(".weight")]
    s = s.replace("model.layers.", "L")
    for tok in ("self_attn.", "mlp."):
        s = s.replace(tok, "")
    s = s.replace("experts.", "e")
    s = s.replace("_proj", "")
    if s.startswith("model."):
        s = s[len("model."):]
    if s == "embed_tokens":
        s = "embed"
    return s


def layer_sort_key(name: str):
    """Sort key putting matrices in network order: embeddings first, then
    layers by NUMERIC index (L2 before L10), within a layer q/k/v/o then
    gate/up/down (dense MLP), then MoE experts by NUMERIC expert index
    (e2 before e10) each gate/up/down, then anything else (e.g. lm_head)
    last."""
    m = re.search(r"layers?\.?(\d+)", name)
    short = short_layer_name(name)
    if m is None:
        group = -1 if "embed" in name else 10 ** 9
        return (group, 0, -1, 0, short)
    mod = short.split(".", 1)[1] if "." in short else ""
    e = re.match(r"e(\d+)\.(.*)$", mod)
    expert = -1
    if e is not None:
        expert, mod = int(e.group(1)), e.group(2)
    rank = _MODULE_ORDER.index(mod) if mod in _MODULE_ORDER else len(_MODULE_ORDER)
    # attention (q/k/v/o) precedes the MLP; dense MLP precedes experts
    attn = rank < 4
    return (int(m.group(1)), 0 if attn else 1, expert if not attn else -1,
            rank, short)


def spectral_norm_means(logs: pd.DataFrame | None,
                        study: str | None = None,
                        df: pd.DataFrame | None = None) -> tuple[str | None, pd.DataFrame]:
    """figure_8 (c) data: final per-layer spectral norms of the fresh-seed
    runs, MEAN over seeds.  Returns (study, frame) where the frame is indexed
    by the full layer name in network order (``layer_sort_key``) over the
    UNION of layers across optimizers, one column per optimizer, NaN where an
    optimizer has no value (e.g. the embedding outside MuonAll).  Optimizers
    with no spectral norms at all (AdamW logs an empty dict) are dropped.
    ``study`` defaults to dense if present, else the first study with data."""
    fin = _fresh_logs(logs, "final", df)
    if fin.empty or "spectral_norms" not in fin.columns:
        return None, pd.DataFrame()
    fin = fin[fin["spectral_norms"].apply(lambda d: isinstance(d, dict) and len(d) > 0)]
    if fin.empty:
        return None, pd.DataFrame()
    if study is None:
        present = list(fin["study"].dropna().unique())
        study = "dense" if "dense" in present else (
            [s for s in STUDIES if s in present] or present)[0]
    fin = fin[fin["study"] == study]
    if fin.empty:
        return study, pd.DataFrame()
    cols = {}
    for opt in [o for o in OPTIMIZERS if o in set(fin["optimizer"])]:
        per_seed = pd.DataFrame(list(fin[fin["optimizer"] == opt]["spectral_norms"]))
        cols[opt] = per_seed.apply(pd.to_numeric, errors="coerce").mean(axis=0)
    out = pd.DataFrame(cols)
    out = out.loc[sorted(out.index, key=layer_sort_key)]
    return study, out


# ---------------------------------------------------------------------------
# Hypothesis verdicts

def _best_variant(df: pd.DataFrame, study: str,
                  optimizers: tuple[str, ...] = OPTIMIZERS) -> str | None:
    """Non-AdamW optimizer with the lowest mean fresh-seed loss, among
    ``optimizers`` (H1 passes the pre-registered five)."""
    means = {o: _fresh(df, study, o)["final_val_loss"].mean()
             for o in optimizers if o != "adamw" and not _fresh(df, study, o).empty}
    return min(means, key=means.get) if means else None


def _best_lr(df: pd.DataFrame, study: str, opt: str) -> float | None:
    fr = _fresh(df, study, opt)
    if fr.empty:
        return None
    return float(fr["lr"].mode().iloc[0])


def hypothesis_verdicts(df: pd.DataFrame, logs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Data-driven verdict table for H1-H7 per PLAN's falsification column.

    Each row: hypothesis id, study, the test quantity computed from ``df``
    (as text, with numbers), a verdict in {supported, falsified,
    indistinguishable}, and the seed-rule caveat.  Missing data never raises:
    the verdict degrades to "indistinguishable" with a note.

    The hypotheses were pre-registered about five optimizers
    (``PREREGISTERED_OPTIMIZERS``).  Rows of any optimizer added later
    (Lion-Muon, DEVIATIONS.md #15) are dropped on entry, so no verdict, test
    quantity or text can change when its results appear: "best variant" (H1),
    "every variant" (H2a), "at least one variant" (H3) and "the curves
    separate" (H7) all keep meaning the variants the hypothesis named.
    """
    if "optimizer" in df.columns:
        df = df[df["optimizer"].isin(PREREGISTERED_OPTIMIZERS)]
    if logs is not None and "optimizer" in logs.columns:
        # defence in depth: H7 also hash-matches its logs against ``df``
        logs = logs[logs["optimizer"].isin(PREREGISTERED_OPTIMIZERS)]
    rows = []

    def add(h, study, quantity, verdict):
        rows.append({"hypothesis": h, "study": study, "quantity": quantity,
                     "verdict": verdict, "caveat": SEED_RULE_CAVEAT})

    # H1: best dense variant's token speedup range in [1.0, 1.4]
    try:
        tok = token_speedup_range("dense", df)
        best = _best_variant(df, "dense", PREREGISTERED_OPTIMIZERS)
        lo, hi = float(tok.loc[best, "speedup_min"]), float(tok.loc[best, "speedup_max"])
        if hi < 1.0 or lo > 1.4:
            v = "falsified"
        elif lo >= 1.0 and hi <= 1.4:
            v = "supported"
        else:
            v = "indistinguishable"
        add("H1", "dense", f"best variant {best} token speedup range "
            f"[{lo:.3f}, {hi:.3f}] vs [1.0, 1.4]", v)
    except Exception as e:  # missing curve etc.
        add("H1", "dense", f"insufficient data ({e})", "indistinguishable")

    # H2a: MLP width >= 256 — token speedup must stay below FLOP break-even
    try:
        studies = [s for s in ("mlp256", "mlp512") if s in set(df["study"])]
        if not studies:
            raise ValueError("no mlp256/mlp512 results")
        parts, violated = [], False
        for study in studies:
            tok = token_speedup_range(study, df)
            for opt in tok.index:
                if opt == "adamw":
                    continue
                fr = _fresh(df, study, opt)
                fa = _fresh(df, study, "adamw")
                be = float(fr["flops_total"].mean() / fa["flops_total"].mean())
                mat = np.asarray(tok.loc[opt, "matrix"], dtype=float)
                fin = mat[np.isfinite(mat)]
                # finite-only max: an inf cell means the fitted asymptote was
                # beaten, which certainly exceeds any break-even ratio
                hi = float(fin.max()) if fin.size else float("inf")
                parts.append(f"{study}/{opt}: speedup<= {hi:.2f} vs break-even {be:.2f}")
                violated |= hi > be or fin.size < mat.size
        add("H2a", "A", "; ".join(parts), "falsified" if violated else "supported")
    except Exception as e:
        add("H2a", "A", f"insufficient data ({e})", "indistinguishable")

    # H2b: Manifold FLOP speedup < 1; Muon/Moonlight within 10% of token speedup
    try:
        fl, tok = flop_speedup_range("dense", df), token_speedup_range("dense", df)
        mlo, mhi = float(fl.loc["manifold", "speedup_min"]), float(fl.loc["manifold", "speedup_max"])
        ratios = {o: float(np.nanmean(np.asarray(fl.loc[o, "matrix"], dtype=float))
                           / np.nanmean(np.asarray(tok.loc[o, "matrix"], dtype=float)))
                  for o in ("muon", "moonlight") if o in fl.index and o in tok.index}
        within = all(abs(r - 1) <= 0.10 for r in ratios.values())
        if mlo >= 1.0:
            v = "falsified"
        elif mhi < 1.0 and within:
            v = "supported"
        else:
            v = "indistinguishable"
        add("H2b", "dense", f"manifold FLOP speedup [{mlo:.3f}, {mhi:.3f}]; "
            f"flop/token ratios {ratios}", v)
    except Exception as e:
        add("H2b", "dense", f"insufficient data ({e})", "indistinguishable")

    # H3: interaction real for at least one variant
    try:
        h3 = h3_interaction(df)
        if h3.empty:
            raise ValueError("no paired dense+moe fresh seeds")
        reals = h3[h3["real"]].index.tolist()
        q = "; ".join(f"{o}: {r.mean_gap_diff:+.4f} (2*std {2 * r.std_gap_diff:.4f})"
                      for o, r in h3.iterrows())
        add("H3", "B", q, "supported" if reals else "falsified")
    except Exception as e:
        add("H3", "B", f"insufficient data ({e})", "indistinguishable")

    # H4: AdamW optimal lr shifts >= 2x from mlp128 to mlp512; Muon's within one grid step
    try:
        la1, la5 = _best_lr(df, "mlp128", "adamw"), _best_lr(df, "mlp512", "adamw")
        lm1, lm5 = _best_lr(df, "mlp128", "muon"), _best_lr(df, "mlp512", "muon")
        if None in (la1, la5, lm1, lm5):
            raise ValueError("missing mlp128/mlp512 best configs")
        sa, sm = abs(math.log2(la5 / la1)), abs(math.log2(lm5 / lm1))
        if sa >= 1.0 and sm <= 1.0 and sm < sa:
            v = "supported"
        elif sm >= sa:
            v = "falsified"
        else:
            v = "indistinguishable"
        add("H4", "A", f"|log2 lr shift| adamw {sa:.2f}, muon {sm:.2f}", v)
    except Exception as e:
        add("H4", "A", f"insufficient data ({e})", "indistinguishable")

    # H5: Muon vs Moonlight distinguishable at some MLP width
    try:
        parts, any_real, any_data = [], False, False
        for study in [s for s in STUDY_A if s in set(df["study"])]:
            pair = _paired_vs(df, study, "muon", "moonlight")
            if pair is None:
                continue
            any_data = True
            real = seed_rule(*pair)
            any_real |= real
            parts.append(f"{study}: mean diff {pair[0].mean():+.4f} real={real}")
        if not any_data:
            raise ValueError("no paired muon/moonlight MLP seeds")
        add("H5", "A", "; ".join(parts), "supported" if any_real else "falsified")
    except Exception as e:
        add("H5", "A", f"insufficient data ({e})", "indistinguishable")

    # H6: MuonAll no better than Muon, and narrower lr basin.  PLAN's
    # falsification column: "MuonAll beats Muon, OR its basin is at least as
    # wide" — so an everywhere-at-least-as-wide basin also falsifies.
    try:
        parts, beats, any_data = [], False, False
        basin_pairs = []
        for study in [s for s in STUDIES if s in set(df["study"])]:
            pair = _paired_vs(df, study, "muonall", "muon")
            if pair is None:
                continue
            any_data = True
            diffs, stds = pair
            if seed_rule(diffs, stds) and diffs.mean() < 0:
                beats = True
            tol = 0.01 if study.startswith("mlp") else 0.02
            bw_all = lr_basin_width(study, "muonall", df, tol)
            bw_muon = lr_basin_width(study, "muon", df, tol)
            if not (math.isnan(bw_all) or math.isnan(bw_muon)):
                basin_pairs.append((bw_all, bw_muon))
            parts.append(f"{study}: diff {diffs.mean():+.4f}, basins "
                         f"muonall {bw_all:.2f} vs muon {bw_muon:.2f}")
        if not any_data:
            raise ValueError("no paired muonall/muon seeds")
        wider_everywhere = bool(basin_pairs) and all(a >= m for a, m in basin_pairs)
        narrower_somewhere = any(a < m for a, m in basin_pairs)
        if beats or wider_everywhere:
            v = "falsified"
        elif narrower_somewhere:
            v = "supported"
        else:
            v = "indistinguishable"
        add("H6", "A+B", "; ".join(parts), v)
    except Exception as e:
        add("H6", "A+B", f"insufficient data ({e})", "indistinguishable")

    # H7 (exploratory): expert-share entropy curves separate on the MoE
    try:
        if logs is None or logs.empty:
            raise ValueError("no logs provided")
        # headline (fresh_seed) runs only: smoke/tuning/budget runs must
        # never decide H7 (200-step smoke runs gave std 0 -> "supported")
        # completed fresh-seed runs only (hash-matched against df), so a
        # stopped run's last eval never stands in for a final value
        ev = _fresh_logs(logs, "eval", df)
        if "expert_entropy" not in ev.columns:
            raise ValueError("no fresh_seed MoE expert-entropy logs")
        ev = ev[(ev["study"] == "moe") & ev["expert_entropy"].notna()]
        if ev.empty:
            raise ValueError("no fresh_seed MoE expert-entropy logs")
        last = ev.sort_values("step").groupby(["optimizer", "seed"]).tail(1)
        g = last.groupby("optimizer")["expert_entropy"]
        means, stds = g.mean(), g.std(ddof=1).fillna(0.0)
        base = means.get("adamw", means.mean())
        sep = any(abs(m - base) > 2 * max(stds.max(), 1e-12)
                  for o, m in means.items() if o != "adamw")
        add("H7", "moe", "final expert entropy " +
            ", ".join(f"{o}: {m:.3f}" for o, m in means.items()),
            "supported" if sep else "falsified")
    except Exception as e:
        add("H7", "moe", f"insufficient data ({e})", "indistinguishable")

    return pd.DataFrame(rows, columns=["hypothesis", "study", "quantity",
                                       "verdict", "caveat"])


# ---------------------------------------------------------------------------
# Figures

def _optimizers_present(df: pd.DataFrame) -> tuple[str, ...]:
    """OPTIMIZERS that have any row in ``df`` (legacy figures 5-7 lay out one
    slot per optimizer; an optimizer without data gets no slot)."""
    present = set(df["optimizer"].dropna())
    return tuple(o for o in OPTIMIZERS if o in present)


def _studies_present(df: pd.DataFrame) -> list[str]:
    present = set(df["study"].dropna())
    return [s for s in STUDIES if s in present]


def _save(fig, outdir: Path | str, name: str) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _no_data(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def figure_1(df: pd.DataFrame, outdir: Path | str) -> Path:
    """Headline speedup: dot + min-max whisker per optimizer, one panel per
    architecture; separate markers for tokens (circle) and FLOPs (square);
    reference line at 1; log y (PLAN Plots #1)."""
    studies = [s for s in _studies_present(df) if _curve_seeds(s, df)]
    n = max(len(studies), 1)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4), squeeze=False)
    if not studies:
        _no_data(axes[0][0], "no budget-curve runs")
    for ax, study in zip(axes[0], studies):
        tok, fl = token_speedup_range(study, df), flop_speedup_range(study, df)
        opts = [o for o in OPTIMIZERS if o in tok.index]
        for x, opt in enumerate(opts):
            c = OPTIMIZER_COLORS[opt]
            for dx, frame, marker in ((-0.12, tok, "o"), (0.12, fl, "s")):
                lo = float(frame.loc[opt, "speedup_min"])
                hi = float(frame.loc[opt, "speedup_max"])
                # inf cells (loss below the fitted asymptote / degenerate
                # fit) are drawn as an open-ended whisker with an inf label
                unbounded = not np.isfinite(hi)
                if not np.isfinite(lo) or lo <= 0:
                    lo = 10.0
                if unbounded:
                    hi = max(10.0 * lo, 10.0)
                mid = math.sqrt(lo * hi)
                ax.errorbar([x + dx], [mid], yerr=[[mid - lo], [hi - mid]],
                            fmt=marker, color=c, capsize=3, markersize=6)
                if unbounded:
                    ax.annotate("inf", (x + dx, hi), fontsize=7, color=c,
                                ha="center", va="bottom")
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xticks(range(len(opts)))
        ax.set_xticklabels(opts, rotation=45, ha="right")
        ax.set_title(study)
        ax.set_ylabel("speedup vs AdamW (log)")
    fig.suptitle("Speedup vs AdamW — circle: tokens, square: FLOPs", fontsize=10)
    return _save(fig, outdir, "fig1_speedup.png")


def figure_2(df: pd.DataFrame, outdir: Path | str) -> Path:
    """LR sensitivity: final loss vs lr-sweep lr (Stage A, or the Phase-I
    coordinate-descent lr sweep — see ``_lr_sweep_rows``), one line per
    optimizer, x markers for diverged trials, one panel per architecture
    (PLAN Plots #2)."""
    sweeps: dict[str, dict[str, pd.DataFrame]] = {}
    for s in _studies_present(df):
        per_opt = {o: _lr_sweep_rows(df, s, o) for o in OPTIMIZERS}
        if any(not g.empty for g in per_opt.values()):
            sweeps[s] = per_opt
    studies = list(sweeps)
    n = max(len(studies), 1)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4), squeeze=False)
    if not studies:
        _no_data(axes[0][0], "no lr-sweep trials")
    for ax, study in zip(axes[0], studies):
        per_opt = sweeps[study]
        comps = [g[g["status"] == "completed"].dropna(subset=["final_val_loss"])
                 for g in per_opt.values()]
        ymax = max((c["final_val_loss"].max() for c in comps if not c.empty),
                   default=1.0)
        for opt in OPTIMIZERS:
            g = per_opt[opt]
            go = g[g["status"] == "completed"].dropna(
                subset=["final_val_loss"]).sort_values("lr")
            if not go.empty:
                ax.plot(go["lr"], go["final_val_loss"], "o-", ms=4,
                        color=OPTIMIZER_COLORS[opt], label=opt)
            div = g[g["status"] == "diverged"]
            if not div.empty:
                ax.scatter(div["lr"], [ymax] * len(div), marker="x", s=60,
                           color=OPTIMIZER_COLORS[opt], zorder=5)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("learning rate (log2)")
        ax.set_ylabel("final val loss")
        ax.set_title(f"{study} (x = diverged)")
    axes[0][0].legend(fontsize=7)
    return _save(fig, outdir, "fig2_lr_sensitivity.png")


def figure_3(df: pd.DataFrame, outdir: Path | str,
             logs: pd.DataFrame | None = None) -> Path:
    """Val-loss curves of the tuned configs: mean line with min-max band over
    fresh seeds; top row x = tokens/images, bottom row x = estimated FLOPs;
    inset on each top panel zoomed on the last 20% (PLAN Plots #3)."""
    have = logs is not None and not logs.empty
    if have:
        ev = logs[(logs["t"] == "eval") & (logs["phase"] == "fresh_seed")]
        studies = [s for s in _studies_present(df) if not ev[ev["study"] == s].empty]
    else:
        studies = []
    if not studies:
        fig, ax = plt.subplots(figsize=(5, 3))
        _no_data(ax, "no fresh-seed eval logs")
        return _save(fig, outdir, "fig3_loss_curves.png")
    fig, axes = plt.subplots(2, len(studies), figsize=(4.2 * len(studies), 6),
                             squeeze=False)
    for col, study in enumerate(studies):
        gs = ev[ev["study"] == study]
        top, bot = axes[0][col], axes[1][col]
        rates = {}
        for opt in OPTIMIZERS:
            fr = _fresh(df, study, opt)
            if not fr.empty:
                rates[opt] = float(fr["flops_total"].mean()) / _unit_size(fr)
        for opt in [o for o in OPTIMIZERS if o in set(gs["optimizer"])]:
            go = gs[gs["optimizer"] == opt]
            agg = go.groupby("images_or_tokens")["val_loss"].agg(["mean", "min", "max"])
            c = OPTIMIZER_COLORS[opt]
            top.plot(agg.index, agg["mean"], color=c, label=opt)
            top.fill_between(agg.index, agg["min"], agg["max"], color=c, alpha=0.2)
            rate = rates.get(opt, 1.0)
            bot.plot(agg.index * rate, agg["mean"], color=c, label=opt)
            bot.fill_between(agg.index * rate, agg["min"], agg["max"], color=c, alpha=0.2)
        xmax = float(gs["images_or_tokens"].max())
        ins = top.inset_axes([0.42, 0.42, 0.55, 0.52])
        tail = gs[gs["images_or_tokens"] >= 0.8 * xmax]
        for opt in [o for o in OPTIMIZERS if o in set(tail["optimizer"])]:
            go = tail[tail["optimizer"] == opt]
            agg = go.groupby("images_or_tokens")["val_loss"].agg(["mean", "min", "max"])
            c = OPTIMIZER_COLORS[opt]
            ins.plot(agg.index, agg["mean"], color=c)
            ins.fill_between(agg.index, agg["min"], agg["max"], color=c, alpha=0.2)
        ins.set_xlim(0.8 * xmax, xmax)
        ins.tick_params(labelsize=6)
        top.set_title(study)
        top.set_xlabel("tokens / images")
        bot.set_xlabel("total FLOPs (est.)")
        top.set_ylabel("val loss")
        bot.set_ylabel("val loss")
    axes[0][0].legend(fontsize=7, loc="lower left")
    return _save(fig, outdir, "fig3_loss_curves.png")


def figure_4(df: pd.DataFrame, outdir: Path | str) -> Path:
    """Rank chart across studies; lines connect the same optimizer; hollow
    markers where the gap to the study's best is within seed spread — the
    seed-rule heuristic (PLAN Plots #4)."""
    studies = [s for s in _studies_present(df) if not _fresh(df, s).empty]
    fig, ax = plt.subplots(figsize=(1.6 * max(len(studies), 3) + 2, 3.6))
    if not studies:
        _no_data(ax, "no fresh-seed runs")
        return _save(fig, outdir, "fig4_ranks.png")
    for opt in OPTIMIZERS:
        xs, ys, hollow = [], [], []
        for i, study in enumerate(studies):
            means = {o: _fresh(df, study, o)["final_val_loss"].mean()
                     for o in OPTIMIZERS if not _fresh(df, study, o).empty}
            if opt not in means:
                continue
            order = sorted(means, key=means.get)
            xs.append(i)
            ys.append(order.index(opt) + 1)
            best = order[0]
            if opt == best:
                hollow.append(False)
            else:
                pair = _paired_vs(df, study, opt, best)
                hollow.append(pair is None or not seed_rule(*pair))
        if xs:
            c = OPTIMIZER_COLORS[opt]
            ax.plot(xs, ys, "-", color=c, label=opt)
            for x, y, h in zip(xs, ys, hollow):
                ax.plot([x], [y], "o", color=c, mfc="none" if h else c, ms=8)
    ax.set_xticks(range(len(studies)))
    ax.set_xticklabels(studies)
    ax.set_ylabel("rank by final loss (1 = best)")
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Hollow = within seed spread (seed-rule heuristic)", fontsize=9)
    return _save(fig, outdir, "fig4_ranks.png")


def figure_5(df: pd.DataFrame, outdir: Path | str) -> Path:
    """Width transfer (MLP): optimal LR vs width per optimizer, plus best loss
    vs width (PLAN Plots #5)."""
    studies = [s for s in STUDY_A if s in set(df["study"]) and not _fresh(df, s).empty]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.4))
    if not studies:
        _no_data(ax1, "no MLP studies")
        _no_data(ax2, "no MLP studies")
        return _save(fig, outdir, "fig5_width_transfer.png")
    widths = {s: float(_fresh(df, s)["width"].iloc[0]) for s in studies}
    # identical optima (e.g. muon == muonall at every width) would hide lines
    # under each other: each optimizer gets its own marker and a small
    # horizontal dodge in log2-width space (plotted values are unchanged)
    optimizers = _optimizers_present(df)
    n_opt = len(optimizers)
    for k, opt in enumerate(optimizers):
        w, lr, loss = [], [], []
        for s in studies:
            fr = _fresh(df, s, opt)
            if fr.empty:
                continue
            w.append(widths[s])
            lr.append(_best_lr(df, s, opt))
            loss.append(fr["final_val_loss"].mean())
        if w:
            c = OPTIMIZER_COLORS[opt]
            mk = OPTIMIZER_MARKERS[opt]
            wd = [x * 2 ** (_FIG5_DODGE * (k - (n_opt - 1) / 2)) for x in w]
            ax1.plot(wd, lr, "-", marker=mk, color=c, label=opt, ms=6,
                     mfc="none", mew=1.4, alpha=0.9)
            ax2.plot(wd, loss, "-", marker=mk, color=c, label=opt, ms=6,
                     mfc="none", mew=1.4, alpha=0.9)
    for ax, yl in ((ax1, "optimal LR"), (ax2, "best final val loss")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(widths.values()))
        ax.set_xticklabels([f"{v:g}" for v in sorted(widths.values())])
        ax.minorticks_off()
        ax.set_xlabel("width (log2; optimizers dodged horizontally)")
        ax.set_ylabel(yl)
    ax1.set_yscale("log")
    ax1.legend(fontsize=7)
    return _save(fig, outdir, "fig5_width_transfer.png")


def figure_6(df: pd.DataFrame, outdir: Path | str) -> Path:
    """Optimizer cost: optimizer FLOPs as a multiple of model (fwd+bwd) FLOPs
    per architecture/width, log scale; Manifold at its logged dual steps
    (PLAN Plots #6)."""
    studies = [s for s in _studies_present(df) if not _fresh(df, s).empty]
    fig, ax = plt.subplots(figsize=(1.4 * max(len(studies), 3) + 2.5, 3.4))
    if not studies:
        _no_data(ax, "no fresh-seed runs")
        return _save(fig, outdir, "fig6_optimizer_cost.png")
    optimizers = _optimizers_present(df)
    width = 0.8 / len(optimizers)
    for k, opt in enumerate(optimizers):
        xs, ys = [], []
        for i, s in enumerate(studies):
            fr = _fresh(df, s, opt)
            if fr.empty:
                continue
            xs.append(i + (k - (len(optimizers) - 1) / 2) * width)
            ys.append(float((fr["flops_optimizer"] / fr["flops_model"]).mean()))
        if xs:
            ax.bar(xs, ys, width=width * 0.9, color=OPTIMIZER_COLORS[opt], label=opt)
    ax.set_yscale("log")
    ax.set_xticks(range(len(studies)))
    ax.set_xticklabels(studies)
    ax.set_ylabel("optimizer / model FLOPs (log)")
    ax.set_title("Manifold Muon at its logged dual steps", fontsize=9)
    ax.legend(fontsize=7)
    return _save(fig, outdir, "fig6_optimizer_cost.png")


def figure_7(df: pd.DataFrame, outdir: Path | str) -> Path:
    """lr x wd heatmaps, one per optimizer and architecture (PLAN Plots #7).

    Phase-II studies use their Stage-B trials.  Phase-I studies have no Stage
    B; their panels fall back to the coordinate-descent trials (phase "cd"),
    whose lr and wd sweeps form a cross in the lr x wd plane (cells away from
    the cross stay blank).  Manifold (no wd) shows its lr-only trials.
    Studies with a cd headline (any cd rows, incl. mlp256/mlp512 since
    DEVIATIONS.md #12) always show the cd (headline) trials, never their
    inherited Stage B."""
    sb = df[(df["phase"] == "stage_b") & (df["status"] == "completed")]
    cd = df[(df["phase"] == "cd") & (df["status"] == "completed")]
    per_study: dict[str, pd.DataFrame] = {}
    for s in _studies_present(df):
        g = cd[cd["study"] == s] if _has_cd_rows(df, s) else sb[sb["study"] == s]
        if not g.empty:
            per_study[s] = g
    studies = list(per_study)
    if not studies:
        fig, ax = plt.subplots(figsize=(4, 3))
        _no_data(ax, "no Stage-B or coordinate-descent trials")
        return _save(fig, outdir, "fig7_lr_wd.png")
    optimizers = _optimizers_present(df)
    fig, axes = plt.subplots(len(optimizers), len(studies),
                             figsize=(3.4 * len(studies), 2.4 * len(optimizers)),
                             squeeze=False)
    for col, study in enumerate(studies):
        for row, opt in enumerate(optimizers):
            ax = axes[row][col]
            g = per_study[study]
            g = g[g["optimizer"] == opt].dropna(subset=["final_val_loss", "lr"])
            if g.empty:
                _no_data(ax, "n/a")
                continue
            if g["wd"].notna().any():
                piv = g.pivot_table(index="wd", columns="lr",
                                    values="final_val_loss", aggfunc="min")
                ax.pcolormesh(np.arange(piv.shape[1] + 1),
                              np.arange(piv.shape[0] + 1), piv.values,
                              cmap="viridis")
                ax.set_xticks(np.arange(piv.shape[1]) + 0.5)
                ax.set_xticklabels([f"{v:g}" for v in piv.columns], fontsize=6)
                ax.set_yticks(np.arange(piv.shape[0]) + 0.5)
                ax.set_yticklabels([f"{v:g}" for v in piv.index], fontsize=6)
                ax.set_xlabel("lr", fontsize=7)
                ax.set_ylabel("wd", fontsize=7)
            else:  # manifold: lr-only Stage B
                gg = g.sort_values("lr")
                ax.plot(gg["lr"], gg["final_val_loss"], "o-",
                        color=OPTIMIZER_COLORS[opt], ms=4)
                ax.set_xscale("log", base=2)
                ax.set_xlabel("lr (no wd)", fontsize=7)
            ax.set_title(f"{study} / {opt}", fontsize=8)
    fig.tight_layout()
    return _save(fig, outdir, "fig7_lr_wd.png")


def figure_8(df: pd.DataFrame, outdir: Path | str,
             logs: pd.DataFrame | None = None) -> Path:
    """Diagnostics of the headline fresh-seed runs only (PLAN Plots #8):
    (a) max attention logit over training, dense solid / MoE dashed, mean over
    seeds; (b) expert-share entropy over training (MoE, H7), mean over seeds;
    (c) final per-layer spectral norms, mean over seeds, log y, on its own
    full-width row.  Smoke, tuning, budget-curve and other non-headline runs
    are excluded, as are runs without a completed fresh-seed result in
    ``df`` (``_fresh_logs``)."""
    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    # (a) max attention logit, dense + moe
    attn = attn_logit_curves(logs, df)
    if not attn.empty:
        for study, ls in (("dense", "-"), ("moe", "--")):
            g = attn[attn["study"] == study]
            for opt in [o for o in OPTIMIZERS if o in set(g["optimizer"])]:
                go = g[g["optimizer"] == opt].sort_values("images_or_tokens")
                ax1.plot(go["images_or_tokens"], go["max_attn_logit"], ls,
                         color=OPTIMIZER_COLORS[opt],
                         label=opt if study == "dense" or not (
                             attn["study"] == "dense").any() else None)
        ax1.set_xlabel("tokens")
        ax1.set_ylabel("max attention logit")
        ax1.legend(fontsize=6)
        ax1.set_title("fresh seeds (mean): dense solid / MoE dashed", fontsize=8)
    else:
        _no_data(ax1, "no fresh-seed attention-logit logs")
    # (b) expert entropy, moe
    ent = expert_entropy_curves(logs, df)
    if not ent.empty:
        for opt in [o for o in OPTIMIZERS if o in set(ent["optimizer"])]:
            go = ent[ent["optimizer"] == opt].sort_values("images_or_tokens")
            ax2.plot(go["images_or_tokens"], go["expert_entropy"],
                     color=OPTIMIZER_COLORS[opt], label=opt)
        ax2.set_xlabel("tokens")
        ax2.set_ylabel("expert-share entropy")
        ax2.legend(fontsize=6)
        ax2.set_title("MoE expert load (H7), fresh seeds (mean)", fontsize=8)
    else:
        _no_data(ax2, "no fresh-seed MoE entropy logs")
    # (c) final spectral norms per layer, mean over fresh seeds
    study, sn = spectral_norm_means(logs, df=df)
    if not sn.empty:
        opts = list(sn.columns)
        layers = list(sn.index)
        w = 0.8 / max(len(opts), 1)
        for k, opt in enumerate(opts):
            xs = [i + (k - (len(opts) - 1) / 2) * w for i in range(len(layers))]
            ax3.bar(xs, sn[opt].to_numpy(dtype=float), width=w * 0.95,
                    color=OPTIMIZER_COLORS[opt], label=opt)
        ax3.set_xticks(range(len(layers)))
        ax3.set_xticklabels([short_layer_name(l) for l in layers],
                            rotation=90, fontsize=7)
        ax3.set_xlim(-0.6, len(layers) - 0.4)
        ax3.set_yscale("log")
        # bars start at the axis floor: keep it below the smallest value
        # (Manifold Muon sits at ~1.0) so every bar is visible
        vmin = float(np.nanmin(sn.to_numpy(dtype=float)))
        ax3.set_ylim(bottom=vmin / 2)
        ax3.set_ylabel("final spectral norm (log)")
        ax3.set_title(f"{study}: per-layer sigma_max, mean over fresh seeds "
                      "(AdamW logs none)", fontsize=8)
        ax3.legend(fontsize=7, ncol=len(opts))
    else:
        _no_data(ax3, "no fresh-seed spectral-norm logs")
    fig.tight_layout()
    return _save(fig, outdir, "fig8_diagnostics.png")


# ---------------------------------------------------------------------------
# Main table

EXTRAPOLATION_MARK = "†"


def _fmt_range(frame: pd.DataFrame, opt: str, mark: bool = False) -> str:
    if frame.empty or opt not in frame.index:
        return "—"
    lo, hi = frame.loc[opt, "speedup_min"], frame.loc[opt, "speedup_max"]
    return f"[{lo:.2f}, {hi:.2f}]" + (EXTRAPOLATION_MARK if mark else "")


def main_table(df: pd.DataFrame, outdir: Path | str) -> Path:
    """tables/main.md: final loss mean±std, token and FLOP speedup ranges,
    LR-basin width and divergence rate per optimizer x study.  Speedup
    ranges that rely on extrapolating the fitted AdamW curve beyond its
    largest measured budget (``speedup_extrapolation``) carry a dagger and a
    one-line footnote.  Study A and
    Study B are separate sections with NO cross-study ranking.  The per-study
    winner (lowest mean loss) is bolded only where the seed rule passes
    against the runner-up — and the seed rule is labeled a heuristic.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Main results",
        "",
        f"Winners are **bold** only where the difference to the runner-up "
        f"passes the seed rule ({SEED_RULE_CAVEAT}).",
        "Study A and Study B are reported separately; no ranking is compared "
        "across the two studies.",
        "",
    ]
    extrap_budgets: list[float] = []
    sections = [("Study A — MLP width study (CIFAR-5m)", STUDY_A),
                ("Study B — dense vs MoE transformer (FineWeb-Edu)", STUDY_B)]
    for title, group in sections:
        studies = [s for s in group if s in set(df["study"])]
        if not studies:
            continue
        lines += [f"## {title}", ""]
        for study in studies:
            opts = [o for o in OPTIMIZERS if not _fresh(df, study, o).empty]
            if not opts:
                continue
            means = {o: float(_fresh(df, study, o)["final_val_loss"].mean()) for o in opts}
            stds = {o: float(_fresh(df, study, o)["final_val_loss"].std(ddof=1))
                    if len(_fresh(df, study, o)) > 1 else 0.0 for o in opts}
            order = sorted(means, key=means.get)
            winner, bold = order[0], False
            if len(order) > 1:
                pair = _paired_vs(df, study, winner, order[1])
                bold = pair is not None and seed_rule(*pair)
            tok, fl = token_speedup_range(study, df), flop_speedup_range(study, df)
            ext = speedup_extrapolation(study, df)
            tol = 0.01 if study.startswith("mlp") else 0.02
            lines += [
                f"### {study}", "",
                "| optimizer | final val loss (mean±std) | token speedup | "
                "FLOP speedup | LR basin (factors of 2) | divergence rate |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for o in opts:
                name = f"**{o}**" if (o == winner and bold) else o
                bw = lr_basin_width(study, o, df, tol)
                bw_s = "—" if math.isnan(bw) else f"{bw:.2f}"
                dr = divergence_rate(study, o, df)
                dr_s = "—" if math.isnan(dr) else f"{dr:.1%}"
                mark = bool(o in ext.index and ext.loc[o, "extrapolated"])
                if mark:
                    extrap_budgets.append(float(ext.loc[o, "max_budget"]))
                lines.append(
                    f"| {name} | {means[o]:.4f} ± {stds[o]:.4f} | "
                    f"{_fmt_range(tok, o, mark)} | {_fmt_range(fl, o, mark)} | "
                    f"{bw_s} | {dr_s} |")
            lines.append("")
    if extrap_budgets:
        b = sorted(set(extrap_budgets))
        b_s = f"{b[0]:g}×" if len(b) == 1 else f"{b[0]:g}×–{b[-1]:g}×"
        lines += [f"{EXTRAPOLATION_MARK} Extrapolated: for at least one (AdamW curve "
                  f"seed, optimizer seed) pair the AdamW-equivalent data exceeds the "
                  f"largest measured AdamW budget ({b_s}), so the speedup is read off "
                  f"the fitted curve beyond the data.", ""]
    nulls = null_result_rows(df)
    if not nulls.empty:
        lines += ["## Null results (indistinguishable after equal tuning)", "",
                  "Reported with the measured seed spread, so the reader sees "
                  "how large an effect could have been detected "
                  f"({SEED_RULE_CAVEAT}).", ""]
        lines += ["| study | optimizer | vs | mean diff | seed spread |",
                  "| --- | --- | --- | --- | --- |"]
        for _, r in nulls.iterrows():
            lines.append(f"| {r['study']} | {r['optimizer']} | {r['vs']} | "
                         f"{r['mean_diff']:+.5f} | {r['seed_spread']:.5f} |")
        lines.append("")
    lines += ["", edge_rule_report(df), ""]
    path = outdir / "main.md"
    path.write_text("\n".join(lines))
    return path
