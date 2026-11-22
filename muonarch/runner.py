"""Run identity and on-disk layout (design.md §4 and §3.9).

This module is ONLY the identity/path layer: which config fields define a
run, how they hash, and where a run's result/log/checkpoint files live.

# Runner class added in a later change (together with studies.py).

Identity contract (design.md §4 "Why this split"): two configs agreeing on
every HASHED_FIELD must produce statistically identical results and share one
result file; two configs differing in any of them must not collide. Fields
that vary with *where or when* a run executes (device, hostname, data_path,
library versions, timestamps, notes, ...) are logged in the result/log files
and NEVER hashed, so a run has the same identity on every laptop and after a
lockfile refresh.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = [
    "HASHED_FIELDS",
    "OMITTED_WHEN_NULL",
    "canonical",
    "config_hash",
    "result_path",
    "log_path",
    "ckpt_path",
    "ckpt_file",
    "is_done",
]

#: Exactly the starred fields of the §4 run-config schema — the identity of a
#: run. Dict-valued fields (manifold, branch, schedule, eval, data, lionmuon)
#: are hashed whole, nested keys included. "lionmuon" is a later addition and
#: is hashed only when set (see OMITTED_WHEN_NULL / ``canonical``).
HASHED_FIELDS: frozenset[str] = frozenset({
    "schema_version",
    "study",
    "phase",
    "arch",
    "width",
    "optimizer",
    "rule",
    "lr",
    "wd",
    "momentum",
    "adamw_betas",
    "adamw_eps",
    "adamw_group_lr",
    "adamw_group_wd",
    "ns_steps",
    "ns_dtype",
    "manifold",
    "init_projection",
    "seed",
    "budget_multiplier",
    "branch",
    "budget_base_steps",
    "budget_unit",
    "batch",
    "seq_len",
    "grad_clip",
    "schedule",
    "precision",
    "router_aux_loss_coef",
    "eval",
    "data",
    "lionmuon",
})

#: Hashed fields added AFTER results existed on disk. ``canonical`` leaves
#: such a field OUT of the hashed view when it is absent/None, so every run
#: that predates the field (and every run it does not apply to) keeps its
#: hash byte for byte. ANY future hashed field must be added here too.
#:   "lionmuon": {"period", "alpha", "beta1", "beta2"} — Lion-Muon runs only.
OMITTED_WHEN_NULL: frozenset[str] = frozenset({"lionmuon"})


def canonical(cfg: dict) -> dict:
    """The hashed view of ``cfg``: HASHED_FIELDS only, in sorted key order.

    An absent field is identical to an explicit ``null`` (the schema's way of
    saying "not applicable", e.g. ``grad_clip: null`` = clipping off). Values
    must be JSON-serializable: numbers, strings, booleans, null, and
    lists/dicts of those (tuples serialize as lists, so tuple/list spelling
    does not fork identities).

    Exception — OMITTED_WHEN_NULL fields (currently "lionmuon"): these were
    added to the schema after ~600 results were already keyed by this hash.
    Emitting ``"lionmuon": null`` for the old runs would change the canonical
    JSON and therefore EVERY existing hash, orphaning all results on disk. So
    such a field appears in the hashed view only when it carries a value
    (non-None); absent/None it is omitted and the canonical JSON of every
    pre-existing config is byte-identical to before. Absent == null still
    holds for them. Any hashed field added in the future MUST follow the same
    rule (add it to OMITTED_WHEN_NULL); tests/test_hash_identity.py checks
    every tracked result file against its filename.
    """
    return {k: cfg.get(k) for k in sorted(HASHED_FIELDS)
            if not (k in OMITTED_WHEN_NULL and cfg.get(k) is None)}


def config_hash(cfg: dict) -> str:
    """First 12 hex chars of sha256 over the canonical JSON.

    ``json.dumps(..., sort_keys=True, separators=(",", ":"))`` sorts keys
    recursively and emits no whitespace; floats are serialized via Python's
    ``float.__repr__`` (shortest round-trip representation — json's default
    float encoder), so any representable change of a float value changes the
    hash. ``allow_nan=False``: NaN/Infinity are not valid run identities.
    """
    blob = json.dumps(
        canonical(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def result_path(root: Path | str, cfg: dict) -> Path:
    """results/<study>/<hash>.json (written atomically, only on completion —
    where a diverged run counts as completed)."""
    return Path(root) / "results" / cfg["study"] / f"{config_hash(cfg)}.json"


def log_path(root: Path | str, cfg: dict) -> Path:
    """logs/<study>/<hash>.jsonl (streamed eval/diagnostic records)."""
    return Path(root) / "logs" / cfg["study"] / f"{config_hash(cfg)}.jsonl"


def ckpt_file(root: Path | str, study: str, run_hash: str, step: int) -> Path:
    """checkpoints/<study>/<hash>@<step>.pt — by explicit hash, for branch
    runs that address their PARENT's checkpoint (cfg["branch"]["parent_hash"])."""
    return Path(root) / "checkpoints" / study / f"{run_hash}@{step}.pt"


def ckpt_path(root: Path | str, cfg: dict, step: int) -> Path:
    """checkpoints/<study>/<hash>@<step>.pt for THIS config (budget-constant
    runs saving their own checkpoints)."""
    return ckpt_file(root, cfg["study"], config_hash(cfg), step)


def is_done(root: Path | str, cfg: dict) -> bool:
    """A run is done iff its result JSON exists — any status: divergence is a
    completed run (it must not re-run; it feeds the divergence-rate metric)."""
    return result_path(root, cfg).is_file()


# ===========================================================================
# Runner (design.md §3.9) — added below the identity layer, which is frozen.
# ===========================================================================
#
# Scheduling contract:
#
# * The DAG is REBUILT from the results on disk after every executed run
#   (later phases' configs depend on earlier results), so a pass executes the
#   first ready phase's pending shard-owned configs and then starts over.
#   The loop ends when a full pass executes nothing: either the study is
#   complete or every pending config belongs to another shard / a blocked
#   phase. Everything is resumable: a crash or Ctrl-C loses at most the
#   in-flight run (results are written atomically on completion only).
#
# * Sharding: a config belongs to laptop i of n iff
#   int(shard_key, 16) % n == i, where shard_key is the config's OWN hash —
#   EXCEPT budget_branch runs, which use their PARENT constant run's hash
#   (cfg["branch"]["parent_hash"]): branches resume from
#   checkpoints/<study>/<parent_hash>@<step>.pt, which exists only on the
#   machine that ran the parent, so a branch must always land on its
#   parent's shard (coordinator decision after the train.py review).
#
# * --parallel k runs a phase's ready configs k at a time in spawned worker
#   processes, all on the SAME device as the sequential path would use (a
#   branch must load its parent's checkpoint, so mixing devices within one
#   runner is not supported).

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor


def result_valid(root: Path | str, cfg: dict) -> bool:
    """True iff the run's result JSON exists AND parses. The execution path
    uses this (not bare is_done): train.run repairs corrupt result files
    (warns and re-runs), so a runner that skipped on is_done alone would
    leave a corrupt result unrepaired forever (coordinator decision after
    the train.py re-review). A corrupt result therefore counts as PENDING:
    it is handed to train.run, whose own idempotency check is authoritative.
    """
    p = result_path(root, cfg)
    if not p.is_file():
        return False
    try:
        json.loads(p.read_text())
        return True
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False


def _shard_key(cfg: dict) -> str:
    """The hash that decides which laptop owns this config: the parent
    constant run's hash for budget_branch runs (the checkpoint is local to
    the parent's machine), the config's own hash otherwise."""
    branch = cfg.get("branch")
    if branch:
        return branch["parent_hash"]
    return config_hash(cfg)


def _run_worker(cfg: dict, root: str, device: str | None) -> str:
    """Module-level so ProcessPoolExecutor('spawn') can pickle it."""
    from muonarch import train

    train.run(cfg, Path(root), device=device)
    return config_hash(cfg)


def _default_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class Runner:
    """Executes one study's DAG: skip done, skip other shards, run the rest
    through train.run (the only way a run executes)."""

    def __init__(self, root: Path | str, laptop_index: int = 0,
                 n_laptops: int = 1, device: str | None = None,
                 parallel: int = 1):
        if not (0 <= laptop_index < n_laptops):
            raise ValueError(f"laptop_index {laptop_index} not in [0, {n_laptops})")
        self.root = Path(root)
        self.laptop_index = laptop_index
        self.n_laptops = n_laptops
        self.device = device
        self.parallel = max(1, parallel)

    # -- ownership ---------------------------------------------------------
    def owns(self, cfg: dict) -> bool:
        return int(_shard_key(cfg), 16) % self.n_laptops == self.laptop_index

    # -- one DAG evaluation ------------------------------------------------
    def _survey(self, study: str):
        """[(phase, ready, configs, pending, mine)] over a fresh DAG."""
        from muonarch import studies

        view = studies.ResultsView(self.root)
        dag = studies.build_dag(study, view)
        complete: dict[str, bool] = {}
        rows = []
        for phase in dag:
            ready = (all(complete.get(n, False) for n in phase.needs)
                     and phase.is_ready())
            cfgs = phase.configs() if ready else []
            pending = [c for c in cfgs if not result_valid(self.root, c)]
            mine = [c for c in pending if self.owns(c)]
            complete[phase.name] = ready and not pending
            rows.append((phase, ready, cfgs, pending, mine))
        return rows

    def _execute(self, cfgs: list[dict]) -> None:
        if self.parallel == 1 or len(cfgs) == 1:
            from muonarch import train

            for cfg in cfgs:
                train.run(cfg, self.root, device=self.device)
            return
        with ProcessPoolExecutor(
            max_workers=self.parallel,
            mp_context=__import__("multiprocessing").get_context("spawn"),
        ) as pool:
            futures = [pool.submit(_run_worker, cfg, str(self.root), self.device)
                       for cfg in cfgs]
            try:
                for f in futures:
                    f.result()
            except KeyboardInterrupt:
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    def run_study(self, study: str) -> int:
        """Loop: rebuild the DAG, execute the first ready phase's pending
        shard-owned configs, repeat until no phase advances. Returns the
        number of runs executed. KeyboardInterrupt propagates after a status
        line (in-flight run lost; everything else resumes)."""
        executed = 0
        try:
            while True:
                progressed = False
                for phase, ready, _cfgs, _pending, mine in self._survey(study):
                    if ready and mine:
                        self._execute(mine)
                        executed += len(mine)
                        progressed = True
                        break   # results changed: rebuild the DAG
                if not progressed:
                    return executed
        except KeyboardInterrupt:
            print(f"\nInterrupted: in-flight run lost; {executed} run(s) "
                  f"completed this session are on disk and will be skipped "
                  f"on resume.", file=sys.stderr)
            raise

    # -- reporting ---------------------------------------------------------
    def status(self, study: str) -> str:
        """Phase-completion table (for PROGRESS.md)."""
        rows = [("phase", "ready", "jobs", "done", "pending(mine)", "complete")]
        for phase, ready, cfgs, pending, mine in self._survey(study):
            done = len(cfgs) - len(pending)
            rows.append((
                phase.name,
                "yes" if ready else "no",
                str(len(cfgs)),
                str(done),
                f"{len(pending)}({len(mine)})",
                "yes" if (ready and not pending) else "no",
            ))
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
                 for r in rows]
        lines.insert(1, "  ".join("-" * w for w in widths))
        return "\n".join(lines)

    def dry_run(self, study: str) -> str:
        """The currently-generatable job list — every ready phase's pending
        configs (all shards) — with per-job and total estimated FLOPs via
        flops.py. Executes nothing."""
        lines = [f"dry run — study {study} (shard {self.laptop_index}/"
                 f"{self.n_laptops}); currently generatable jobs:"]
        header = (f"{'hash':12}  {'phase':22}  {'optimizer':9}  {'lr':>10}  "
                  f"{'wd':>8}  {'seed':>4}  {'mult':>5}  {'steps':>5}  "
                  f"{'est FLOPs':>10}  mine")
        lines.append(header)
        total = 0
        per_opt: dict[str, int] = {}
        n = 0
        for phase, ready, _cfgs, pending, _mine in self._survey(study):
            if not ready:
                continue
            for cfg in pending:
                fl = estimate_flops(cfg)
                total += fl
                per_opt[cfg["optimizer"]] = per_opt.get(cfg["optimizer"], 0) + fl
                n += 1
                wd = cfg.get("wd")
                lines.append(
                    f"{config_hash(cfg):12}  {phase.name:22}  "
                    f"{cfg['optimizer']:9}  {cfg['lr']:>10.3g}  "
                    f"{(f'{wd:.3g}' if wd is not None else '-'):>8}  "
                    f"{cfg['seed']:>4}  {cfg['budget_multiplier']:>5.3g}  "
                    f"{_run_steps(cfg):>5}  {fl:>10.3e}  "
                    f"{'yes' if self.owns(cfg) else 'no'}")
        lines.append(f"total: {n} job(s), {total:.3e} FLOPs "
                     f"(per optimizer: "
                     + ", ".join(f"{o} {f:.2e}" for o, f in sorted(per_opt.items()))
                     + ")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FLOP estimation for planning (dry-run and scripts/timing.py)

_MODEL_CACHE: dict[tuple, object] = {}


def _planning_model(arch: str, width):
    key = (arch, width)
    if key not in _MODEL_CACHE:
        import torch

        from muonarch.models import build_model

        torch.manual_seed(0)
        _MODEL_CACHE[key] = build_model({"arch": arch, "width": width})
    return _MODEL_CACHE[key]


def _run_steps(cfg: dict) -> int:
    """Steps this run actually executes: its budget, minus the checkpoint
    prefix for a branch (which resumes at branch.checkpoint_step)."""
    from muonarch.schedule import branch_steps

    total = branch_steps(cfg["budget_base_steps"], cfg["budget_multiplier"])
    if cfg.get("branch"):
        return total - cfg["branch"]["checkpoint_step"]
    return total


def estimate_flops(cfg: dict) -> int:
    """Estimated total FLOPs of one run: steps x (model + optimizer FLOPs per
    step), analytically (flops.py). Manifold's msign_calls are estimated at
    dual_cap + 1 per matrix per step (at most ``cap`` ascent calls plus the
    retraction, measured on the real ManifoldMuon; early stop gives
    taken + 2 <= cap + 1) — an upper bound, so early stopping only makes
    real runs cheaper. Lion-Muon is exact, not a per-step average: its Muon
    steps are the run steps t with t % period == 0
    (flops.lionmuon_optimizer_flops)."""
    from muonarch import flops as flops_mod
    from muonarch import train
    from muonarch.optim import split_params

    model = _planning_model(cfg["arch"], cfg.get("width"))
    model_fl = train._model_flops_per_step(cfg, model)
    groups = split_params(model, cfg["optimizer"], cfg["arch"])
    muon_mats = [tuple(p.shape) for _, p in groups.muon if p.ndim >= 2]
    vec_sizes = [p.numel() for _, p in groups.muon if p.ndim == 1]
    vec_sizes += [p.numel() for _, p in groups.adamw]
    if cfg["optimizer"] == "manifold":
        m = cfg["manifold"]
        opt_fl = flops_mod.optimizer_flops_per_step(
            muon_mats, vec_sizes, "manifold",
            msign_steps=m.get("msign_steps", 10),
            msign_calls=m.get("dual_cap", 5) + 1)
    elif cfg["optimizer"] == "lionmuon":
        n = _run_steps(cfg)
        return n * model_fl + flops_mod.lionmuon_optimizer_flops(
            muon_mats, vec_sizes, n_steps=n,
            period=cfg["lionmuon"]["period"], ns_steps=cfg["ns_steps"])
    else:
        opt_fl = flops_mod.optimizer_flops_per_step(
            muon_mats, vec_sizes, cfg["optimizer"], ns_steps=cfg["ns_steps"])
    return _run_steps(cfg) * (model_fl + opt_fl)


# ---------------------------------------------------------------------------
# CLI: python -m muonarch.runner --study S [--shard i/n] [--parallel k]
#                                [--dry-run] [--status] [--device d]

def _parse_shard(text: str) -> tuple[int, int]:
    try:
        i, n = text.split("/")
        return int(i), int(n)
    except Exception as e:  # noqa: BLE001 — argparse reports it
        raise argparse.ArgumentTypeError(
            f"--shard wants i/n (e.g. 0/4), got {text!r}") from e


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m muonarch.runner",
        description="Execute (or inspect) one study's phase DAG.")
    parser.add_argument("--study", required=True,
                        help="mlp128|mlp256|mlp512|dense|moe")
    parser.add_argument("--shard", type=_parse_shard, default=(0, 1),
                        metavar="i/n", help="this laptop's index / laptop count")
    parser.add_argument("--parallel", type=int, default=1, metavar="k",
                        help="run ready configs k at a time (spawn workers)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the currently-generatable job list with "
                             "estimated FLOPs; run nothing")
    parser.add_argument("--status", action="store_true",
                        help="print the phase-completion table; run nothing")
    parser.add_argument("--root", default=Path(__file__).resolve().parent.parent,
                        help="project root (default: the repo)")
    parser.add_argument("--device", default=None,
                        help="torch device for train.run (default: mps if "
                             "available, else cpu)")
    args = parser.parse_args(argv)

    i, n = args.shard
    device = args.device or _default_device()
    r = Runner(args.root, laptop_index=i, n_laptops=n, device=device,
               parallel=args.parallel)
    if args.dry_run:
        print(r.dry_run(args.study))
        return 0
    if args.status:
        print(r.status(args.study))
        return 0
    try:
        executed = r.run_study(args.study)
    except KeyboardInterrupt:
        return 130
    print(f"run_study({args.study!r}): executed {executed} run(s); "
          f"no phase can advance further on this shard.")
    print(r.status(args.study))
    return 0


__all__ += ["Runner", "estimate_flops", "result_valid", "_shard_key"]

if __name__ == "__main__":
    sys.exit(_main())
