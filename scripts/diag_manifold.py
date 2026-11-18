"""Manifold Muon audit (Gate-3 follow-up): diagnostic-only ablations on MLP-128 seed 0.

NOT part of the frozen protocol and never written to results/: outputs go to
the directory given as argv[1] (default logs/diagnostics/manifold_audit).
Configs start from the MLP-128 fresh-seed-1 winners with seed set to 0.
Idempotent: experiments already in summary.jsonl are skipped.
Usage: uv run python scripts/diag_manifold.py [outdir]
See docs/manifold_audit.md.
"""
import copy, json, sys, glob
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from muonarch import train

PROJ = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ / "logs/diagnostics/manifold_audit"
OUT.mkdir(parents=True, exist_ok=True); SUMMARY = OUT / "summary.jsonl"
dev = torch.device("mps")

base = muon = None
for f in glob.glob(str(PROJ / "results/mlp128/*.json")):
    c = json.load(open(f))["config"]
    if c["phase"] == "fresh_seed" and c["seed"] == 1:
        if c["optimizer"] == "manifold": base = c
        if c["optimizer"] == "muon": muon = c
for c in (base, muon):
    c["seed"] = 0; c["phase"] = "cd"

class Scaled(nn.Module):
    def __init__(self, lin, s): super().__init__(); self.lin, self.s = lin, s
    def forward(self, x): return F.linear(x, self.s * self.lin.weight, self.lin.bias)

orig_setup = train.setup
SCALE = {"s": None}
def setup(cfg, device):
    model, opts, groups = orig_setup(cfg, device)
    if SCALE["s"] is not None:
        for i in (0, 2, 4): model[i] = Scaled(model[i], SCALE["s"])
    return model, opts, groups
train.setup = setup

def v(c, **kw):
    c = copy.deepcopy(c)
    for k, val in kw.items():
        if k == "dual_cap": c["manifold"]["dual_cap"] = val
        else: c[k] = val
    return c

exps = [
    ("baseline_replica", v(base), None),
    ("cap0_projection_only", v(base, dual_cap=0), None),
    ("momentum0", v(base, momentum=0.0), None),
    ("adamw_group_lr_x4", v(base, adamw_group_lr=0.004), None),
    ("adamw_group_lr_x16", v(base, adamw_group_lr=0.016), None),
    ("adamw_group_wd0", v(base, adamw_group_wd=0.0), None),
    ("scale2.8", v(base, diag_tag="scale2.8"), 2.8),
    ("scale2.8_lr_x4", v(base, diag_tag="scale2.8b", adamw_group_lr=0.004), 2.8),
    ("muon_seed0_reference", v(muon), None),
    ("scale8", v(base), 8.0),
    ("scale0.35", v(base), 0.35),
    ("adamw_group_lr_x16_wd0", v(base, adamw_group_lr=0.016, adamw_group_wd=0.0), None),
    ("adamw_group_lr_x64", v(base, adamw_group_lr=0.064), None),
    ("muon_adamw_group_lr_x16", v(muon, adamw_group_lr=0.016), None),
]
data = train.open_data(base, PROJ, dev)
done = {json.loads(l)["name"] for l in open(SUMMARY)} if SUMMARY.exists() else set()
for name, cfg, s in exps:
    if name in done: continue
    SCALE["s"] = s
    r = train.run(cfg, OUT / name if s is not None else OUT, device=dev, data=data)
    row = {"name": name, "val": r["final_val_loss"], "test_acc": r["final_test_acc"], "status": r["status"],
           "gap": r["train_val_gap"], "wall": r["wall_clock_s"], "hash": r["hash"]}
    print(row, flush=True)
    with open(SUMMARY, "a") as f: f.write(json.dumps(row) + "\n")
print("DIAG DONE", flush=True)
