#!/usr/bin/env python
"""Profile one Manifold Muon training step (dense / MoE, cap 5 / 10) on MPS.

Splits wall time into forward+backward(+clip), each optimizer's step, and,
inside ManifoldMuon.step, time in dual-ascent msign calls vs retraction msign
calls vs everything else (momentum, stacking, residual checks, Lambda updates,
copies). Every timed region is bracketed by a device synchronize, so absolute
times include sync overhead; use them as a split, not as s/step.

Diagnostic only: writes nothing under results/, changes no code paths.

  uv run python scripts/profile_manifold.py --study moe --cap 5
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from muonarch import manifold as mm  # noqa: E402
from muonarch import train  # noqa: E402
from smoke import smoke_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=["dense", "moe"])
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sync = torch.mps.synchronize if dev.type == "mps" else (lambda: None)

    cfg = smoke_config(args.study, "manifold", args.warmup + args.steps, dual_cap=args.cap)
    model, opts, _ = train.setup(cfg, dev)
    data = train.open_data(cfg, ROOT, dev)
    batches = train.train_batches(cfg, data, args.warmup + args.steps)
    aux_coef = cfg.get("router_aux_loss_coef")
    clip = cfg.get("clip")

    # --- instrumentation of msign inside the manifold update -------------
    rec = {"on": False, "calls": []}  # (kind, shape, seconds)
    orig_msign, orig_update = mm.msign, mm._manifold_update
    current: list = []

    def timed_msign(X, *a, **k):
        if not rec["on"]:
            return orig_msign(X, *a, **k)
        sync(); t = time.perf_counter()
        out = orig_msign(X, *a, **k)
        sync(); current.append((tuple(X.shape), time.perf_counter() - t))
        return out

    def timed_update(W, G, **kw):
        current.clear()
        sync(); t = time.perf_counter()
        out = orig_update(W, G, **kw)
        sync(); total = time.perf_counter() - t
        if rec["on"]:
            for i, (shape, dt) in enumerate(current):
                kind = "retraction" if i == len(current) - 1 else "ascent"
                rec["calls"].append((kind, shape, dt))
            rec["calls"].append(("update_total", tuple(W.shape), total))
            rec.setdefault("asc_per_update", []).append((tuple(W.shape), len(current) - 1))
        return out

    mm.msign, mm._manifold_update = timed_msign, timed_update

    acc = collections.defaultdict(float)
    for step in range(args.warmup + args.steps):
        measure = step >= args.warmup
        rec["on"] = measure
        x, y = next(batches)
        x, y = x.to(dev), y.to(dev)
        sync(); t0 = time.perf_counter()
        loss, _, _ = train.lm_forward_loss(model, x, y, aux_coef)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        train.clip_gradients(model.parameters(), clip)
        sync(); t1 = time.perf_counter()
        step_times = {}
        for o in opts:
            ts = time.perf_counter()
            o.step()
            sync()
            step_times[type(o).__name__] = time.perf_counter() - ts
        t2 = time.perf_counter()
        if measure:
            acc["fwd_bwd_clip"] += t1 - t0
            for name, dt in step_times.items():
                acc[f"step:{name}"] += dt
            acc["total"] += t2 - t0

    n = args.steps
    calls = rec["calls"]
    asc = sum(dt for k, _, dt in calls if k == "ascent")
    ret = sum(dt for k, _, dt in calls if k == "retraction")
    upd = sum(dt for k, _, dt in calls if k == "update_total")
    man = acc["step:ManifoldMuon"]
    print(f"== {args.study} cap {args.cap} on {dev}: mean over {n} steps (synchronized) ==")
    print(f"total step            {acc['total']/n*1000:9.1f} ms")
    print(f"fwd+bwd+clip          {acc['fwd_bwd_clip']/n*1000:9.1f} ms")
    for k, v in acc.items():
        if k.startswith("step:"):
            print(f"{k:22s}{v/n*1000:9.1f} ms")
    print(f"  manifold: ascent msign      {asc/n*1000:9.1f} ms")
    print(f"  manifold: retraction msign  {ret/n*1000:9.1f} ms")
    print(f"  manifold: other in update   {(upd-asc-ret)/n*1000:9.1f} ms "
          "(residual .item() syncs, Lambda updates, matmuls)")
    print(f"  manifold: outside update    {(man-upd)/n*1000:9.1f} ms "
          "(momentum, stack, copy-back)")
    per = collections.defaultdict(list)
    for shape, a in rec.get("asc_per_update", []):
        per[shape].append(a)
    n_calls = sum(1 for k, _, _ in calls if k in ("ascent", "retraction"))
    print(f"  batched msign calls/step: {n_calls/n:.1f} over {len(per)} shape groups")
    for shape, lst in sorted(per.items()):
        print(f"    stack {shape}: ascent calls/update mean {sum(lst)/len(lst):.2f} "
              f"(min {min(lst)}, max {max(lst)})")

if __name__ == "__main__":
    main()
