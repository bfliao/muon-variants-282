#!/usr/bin/env python
"""Per-step re-timing of the transformer studies (Gate 3).

Replaces smoke.py's (t200 - t20)/180 estimate, which proved unreliable on
the transformers (one-off costs vary between the two runs). For each
(study, optimizer[, Manifold cap]) it builds the real smoke config (literature
default lr, real data), runs `--warmup` untimed steps, then times
`--steps` training steps one by one, each bracketed by a device
synchronize, doing exactly the train.run step work (schedule-free here:
zero_grad, forward incl. MoE aux, backward, global clip, every optimizer's
step, loss.item()).

Eval overhead is optimizer-independent, so it is timed once per study with
the same model: one mid-run eval (evaluate on the 0.5M-token set +
diagnostics) and the final 2M-token eval. The CSV s/step is
    median step time + (mid-run eval time) / every_steps
                     + (final eval time) / 1x steps
i.e. the eval cost amortised per training step, so timing.py's
steps x s/step model includes it. The raw step time and both eval times are
printed and written to logs/retime_lm.jsonl.

Diagnostic only: nothing is written under results/.

  uv run python scripts/retime_lm.py --study moe
"""
from __future__ import annotations

import argparse
import csv
import json
import socket
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from muonarch import studies, train  # noqa: E402
from muonarch.data import lm_val_batches  # noqa: E402
from smoke import CSV_FIELDS, append_csv, smoke_config  # noqa: E402

UNITS = [("adamw", None), ("muon", None), ("moonlight", None),
         ("manifold", 5), ("manifold", 10), ("muonall", None)]


def time_steps(cfg, dev, sync, warmup, steps):
    model, opts, _ = train.setup(cfg, dev)
    data = train.open_data(cfg, ROOT, dev)
    batches = train.train_batches(cfg, data, warmup + steps)
    aux_coef = cfg.get("router_aux_loss_coef")
    clip = cfg.get("clip")
    times = []
    for step in range(warmup + steps):
        x, y = next(batches)
        sync(); t0 = time.perf_counter()
        x, y = x.to(dev), y.to(dev)
        loss, _, _ = train.lm_forward_loss(model, x, y, aux_coef)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        train.clip_gradients(model.parameters(), clip)
        for o in opts:
            o.step()
        float(loss.detach())
        sync()
        if step >= warmup:
            times.append(time.perf_counter() - t0)
    return model, data, times


def time_evals(model, cfg, data, dev, sync, groups):
    sync(); t0 = time.perf_counter()
    train.evaluate(model, cfg, data, dev)
    probe = next(lm_val_batches(data, cfg["eval"]["val_small_tokens"],
                                batch_seqs=cfg["batch"], seq_len=cfg["seq_len"]))[0].to(dev)
    train.diagnostics(model, cfg, probe, groups)
    sync(); t1 = time.perf_counter()
    train.evaluate(model, cfg, data, dev, final=True)
    sync(); t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=["dense", "moe"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--csv", type=Path, default=ROOT / "hardware-results.csv")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sync = torch.mps.synchronize if dev.type == "mps" else (lambda: None)
    sv = studies.load_study(args.study)
    every = int(sv["eval"]["every_steps"])

    rows = []
    eval_mid = eval_final = None
    one_x = None
    for opt, cap in UNITS:
        cfg = smoke_config(args.study, opt, args.warmup + args.steps, dual_cap=cap)
        model, data, times = time_steps(cfg, dev, sync, args.warmup, args.steps)
        if eval_mid is None:  # optimizer-independent: once per study
            _, _, groups = train.setup(cfg, dev)
            eval_mid, eval_final = time_evals(model, cfg, data, dev, sync, groups)
            one_x = studies.run_config(
                sv, "adamw", "fresh_seed", studies.cd_defaults(sv, "adamw"), seed=1)
            one_x = int(one_x.get("budget_base_steps") or 2441)
        med = statistics.median(times)
        s = med + eval_mid / every + eval_final / one_x
        rows.append(dict(study=args.study, optimizer=opt, cap=cap, step_median=med,
                         step_min=min(times), step_max=max(times),
                         eval_mid_s=eval_mid, eval_final_s=eval_final,
                         s_per_step_incl_eval=s, n=len(times)))
        label = opt + (f" cap {cap}" if cap else "")
        print(f"{args.study:5s} {label:14s} step median {med*1000:7.1f} ms "
              f"[{min(times)*1000:.0f}-{max(times)*1000:.0f}]  "
              f"-> {s:.4f} s/step incl. eval", flush=True)
        append_csv(args.csv, {
            "hostname": socket.gethostname(), "device": dev.type,
            "study": args.study, "optimizer": opt,
            "cap": "" if cap is None else cap, "s_per_step": f"{s:.6f}",
            "timestamp": datetime.now(timezone.utc).isoformat()})
        del model
    print(f"{args.study}: mid-run eval+diagnostics {eval_mid:.1f} s (every {every} "
          f"steps), final eval {eval_final:.1f} s, 1x = {one_x} steps")
    with (ROOT / "logs" / "retime_lm.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    assert CSV_FIELDS  # same schema as smoke.py rows


if __name__ == "__main__":
    main()
