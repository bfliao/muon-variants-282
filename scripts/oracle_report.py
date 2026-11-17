"""Summarize the Phase-1 oracle runs from the vendored script's pickles.

Reads logs/oracle/results/update-*.pkl (written by the UNMODIFIED
third_party/manifolds/src/main.py) and prints, per run: train/test accuracy,
mean time per epoch and per step, and for manifold_muon the first layer's
singular-value stats (the Stiefel check). Numbers come only from the pickles.
"""

import glob
import os
import pickle
import re
import statistics
import sys

import torch  # noqa: F401  (needed to unpickle tensors)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "logs", "oracle", "results")
STEPS_PER_EPOCH = 49  # ceil(50000 / 1024), fixed by the vendored DataLoader


def load_all() -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "update-*.pkl"))):
        with open(path, "rb") as f:
            r = pickle.load(f)
        r["_file"] = os.path.basename(path)
        out.append(r)
    return out


def main() -> None:
    runs = load_all()
    if not runs:
        sys.exit(f"no pickles in {RESULTS}")
    print(f"{'update':<16} {'lr':>7} {'seed':>4} {'train%':>7} {'test%':>7} "
          f"{'s/epoch':>8} {'s/step':>7}  first-layer sv (min/mean/max)")
    for r in runs:
        sv = r["singular_values"][0].float().cpu()
        t_epoch = statistics.mean(r["epoch_times"])
        print(f"{r['update']:<16} {r['lr']:>7} {r['seed']:>4} "
              f"{r['train_acc']:>7.2f} {r['test_acc']:>7.2f} "
              f"{t_epoch:>8.1f} {t_epoch / STEPS_PER_EPOCH:>7.2f}  "
              f"{sv.min():.4f} / {sv.mean():.4f} / {sv.max():.4f}")

    # Per-optimizer aggregation over seeds (used for the gate table)
    by: dict[tuple[str, float], list[dict]] = {}
    for r in runs:
        by.setdefault((r["update"], r["lr"]), []).append(r)
    print("\nAggregates over seeds (where >1 seed):")
    for (upd, lr), rs in sorted(by.items()):
        if len(rs) < 2:
            continue
        tr = [x["train_acc"] for x in rs]
        te = [x["test_acc"] for x in rs]
        print(f"{upd} lr={lr} seeds={sorted(x['seed'] for x in rs)}: "
              f"train {statistics.mean(tr):.2f} ± {statistics.stdev(tr):.2f}  "
              f"test {statistics.mean(te):.2f} ± {statistics.stdev(te):.2f}")


if __name__ == "__main__":
    main()
