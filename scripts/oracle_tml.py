"""Thin launcher for the UNMODIFIED Thinking Machines manifolds code.

Phase 1 oracle check (PLAN.md "Manifold Muon oracle check"): runs
third_party/manifolds/src/main.py on real CIFAR-10 on this machine's device.
The vendored code calls .cuda(); this shim maps .cuda() to the local device
(MPS, falling back to CPU if the vendored bf16 msign misbehaves on MPS) and
then executes main.py via runpy, passing CLI arguments through untouched.

Nothing in third_party/ is modified.

Usage (CWD decides where ./data and ./results land):
    uv run python scripts/oracle_tml.py --update adam --lr 1e-3 --wd 0.0 \
        --epochs 3 --seed 42
"""

import os
import sys
import runpy

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "third_party", "manifolds", "src")
sys.path.insert(0, SRC)


def pick_device() -> torch.device:
    forced = os.environ.get("ORACLE_DEVICE")
    if forced:
        return torch.device(forced)
    if not torch.backends.mps.is_available():
        return torch.device("cpu")
    # Probe: the vendored Polar Express msign runs in bf16. On MPS this can
    # either error out or return garbage on some Mac/torch combinations, so
    # check that msign of a random matrix is close to orthogonal.
    try:
        from msign import msign  # vendored module

        g = torch.randn(64, 32, generator=torch.Generator().manual_seed(0))
        x = msign(g.to("mps")).cpu()
        err = (x.mT @ x - torch.eye(32)).norm().item()
        if err < 0.1 and torch.isfinite(x).all():
            return torch.device("mps")
        print(f"[oracle shim] MPS bf16 msign probe off (||XtX-I||={err:.3f}); using CPU")
    except Exception as e:  # noqa: BLE001
        print(f"[oracle shim] MPS bf16 msign probe raised ({e!r}); using CPU")
    return torch.device("cpu")


DEVICE = pick_device()
print(
    f"[oracle shim] device={DEVICE} dtype=vendored(bf16 msign) "
    f"torch={torch.__version__} argv={sys.argv[1:]}"
)

# Map the vendored code's .cuda() calls onto the local device.
torch.Tensor.cuda = lambda self, *a, **k: self.to(DEVICE)  # type: ignore[method-assign]
torch.nn.Module.cuda = lambda self, *a, **k: self.to(DEVICE)  # type: ignore[method-assign]
if not torch.cuda.is_available():
    torch.cuda.manual_seed_all = lambda *a, **k: None  # type: ignore[assignment]

sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path(os.path.join(SRC, "main.py"), run_name="__main__")
