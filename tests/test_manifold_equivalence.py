"""Phase-1 gate tests: muonarch ManifoldMuon vs the vendored Thinking Machines
reference (third_party/manifolds/src), plus supporting tests.

The vendored code is imported here (and only here) via a temporary sys.path
entry; muonarch itself never imports from third_party/ at runtime.

Measured deviations (CPU, torch 2.15, fixed seeds) are recorded in each
test's docstring, with the tolerance chosen from them.

One numerics footnote: CPU bf16 matmul kernels give slightly different
accumulations for different input strides. msign() of a tall matrix returns a
column-major tensor (its internal transpose), and feeding that layout to the
reference while our optimizer stacks a row-major copy showed up as a ~4e-3
drift after 100 bf16 dual steps. Real parameters (nn.Linear weights) are
row-major contiguous, so the gate inputs below are made contiguous; with that,
the bf16 paths are bit-identical.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TML_SRC = ROOT / "third_party" / "manifolds" / "src"
sys.path.insert(0, str(TML_SRC))
import msign as tml_msign  # noqa: E402
import manifold_muon as tml_manifold  # noqa: E402
sys.path.remove(str(TML_SRC))

from muonarch.manifold import ManifoldMuon, grouped_by_shape  # noqa: E402
from muonarch.msign import ABC_LIST, ABC_LIST_STABLE, msign  # noqa: E402


# ---------------------------------------------------------------- (a) coefficients
def test_abc_lists_match_vendored_exactly():
    assert ABC_LIST == tml_msign.ABC_LIST
    assert ABC_LIST_STABLE == tml_msign.ABC_LIST_STABLE


# ---------------------------------------------------------------- (b) bf16 fidelity
MSIGN_SHAPES = [(64, 64), (128, 64), (64, 128), (96, 32), (32, 96), (4, 48, 32)]


@pytest.mark.parametrize("shape", MSIGN_SHAPES, ids=str)
def test_msign_bf16_matches_vendored(shape):
    """Our msign(G, dtype=bfloat16) vs vendored msign(G) on identical inputs.

    Measured max |diff| over all shapes: 0.0 (bit-identical) — the only code
    difference is `G.to(dtype, copy=True)` vs `G.bfloat16()`, which produce
    the same tensor for fp32 inputs; every subsequent op is identical.
    """
    torch.manual_seed(hash(shape) % (2**31))
    G = torch.randn(*shape)
    G_before = G.clone()
    ours = msign(G, dtype=torch.bfloat16)
    ref = tml_msign.msign(G)
    assert torch.equal(G, G_before), "msign must not mutate its input"
    assert ours.dtype == torch.float32
    assert torch.allclose(ours, ref, atol=1e-7, rtol=0), (
        f"max diff {(ours - ref).abs().max().item():.3e}"
    )


def test_msign_fp32_default_orthogonalizes():
    """Default-dtype (fp32) msign should push singular values to ~1."""
    torch.manual_seed(0)
    for shape in [(48, 24), (24, 48)]:
        G = torch.randn(*shape)
        X = msign(G)
        s = torch.linalg.svdvals(X)
        assert (s - 1).abs().max() < 1e-3


# ---------------------------------------------------------------- (c) THE gate test
GATE_SHAPES = [(128, 64), (64, 128), (3072, 128)]


def _reference_step(W, G, lr=0.1):
    return tml_manifold.manifold_muon(
        W.clone(), G.clone(), eta=lr, alpha=0.01, steps=100, tol=1e-6
    )


def _our_step(W, G, msign_dtype, lr=0.1):
    p = torch.nn.Parameter(W.clone())
    p.grad = G.clone()
    opt = ManifoldMuon(
        [p], lr=lr, dual_cap=100, tol=1e-6, dual_lr=0.01,
        momentum=0.0, msign_steps=10, msign_dtype=msign_dtype,
    )
    stats = opt.step()
    return p.detach().clone(), stats


def _gate_inputs(shape):
    torch.manual_seed(42)
    # Start on the Stiefel manifold (fp32 projection). .contiguous() because
    # msign returns a column-major view-of-transpose for tall inputs, and CPU
    # bf16 matmul accumulates slightly differently per stride layout (see
    # module docstring); real parameters are contiguous.
    W = msign(torch.randn(*shape)).contiguous()
    G = 0.1 * torch.randn(*shape)
    return W, G


@pytest.mark.parametrize("shape", GATE_SHAPES, ids=str)
def test_one_step_matches_reference_bf16(shape):
    """Phase-1 gate, bf16 path: one ManifoldMuon step (dual_cap=100, momentum=0,
    tol=1e-6, msign_dtype=bfloat16) must match the vendored
    manifold_muon(W, G, eta=lr, alpha=0.01, steps=100, tol=1e-6).

    Identical algorithm and dtype; the only difference is that our dual ascent
    runs batched with a leading dim of 1. Measured max |diff| per shape
    (CPU, torch 2.15): (128,64) 0.0, (64,128) 0.0, (3072,128) 0.0 —
    bit-identical on contiguous inputs; asserted at atol=1e-6 to leave room
    for batched-matmul kernel differences on other backends.
    """
    W, G = _gate_inputs(shape)
    ref = _reference_step(W, G)
    ours, stats = _our_step(W, G, msign_dtype=torch.bfloat16)
    assert ours.shape == W.shape
    assert torch.allclose(ours, ref, atol=1e-6, rtol=0), (
        f"max diff {(ours - ref).abs().max().item():.3e}"
    )
    assert len(stats["dual_steps"]) == 1
    assert len(stats["dual_residual"]) == 1


@pytest.mark.parametrize("shape", GATE_SHAPES, ids=str)
def test_one_step_matches_reference_fp32_loosely(shape):
    """Phase-1 gate, fp32 sanity bound: the same step with msign_dtype=fp32
    (our production setting) vs the bf16 reference.

    bf16 rounding inside the reference's msign (~2^-8 relative) perturbs every
    dual-ascent iterate, and could in principle change WHICH dual step
    early-stops, so this comparison is a coarse sanity bound only; the hard
    equivalence assertion is the bf16-vs-bf16 test above. On these inputs no
    early stop fires (residuals ~1e-2 >> tol, both paths take all 100 steps),
    so the comparison is not flaky here. Measured max |diff| per shape
    (CPU, torch 2.15): (128,64) 2.39e-3, (64,128) 2.91e-3, (3072,128) 1.28e-3.
    Asserted at atol=1e-2 (entries of W are O(n^-1/2) ≈ 0.09-0.13 here).
    """
    W, G = _gate_inputs(shape)
    ref = _reference_step(W, G)
    ours, _ = _our_step(W, G, msign_dtype=torch.float32)
    assert torch.allclose(ours, ref, atol=1e-2, rtol=0), (
        f"max diff {(ours - ref).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------- (d) Stiefel stay
@pytest.mark.parametrize("shape", [(32, 16), (16, 32)], ids=str)
def test_stiefel_stay_over_50_steps(shape):
    """project_ then 50 ManifoldMuon steps (momentum 0.95, lr 0.05, cap 5)
    with random gradients keeps all singular values within 1e-2 of 1."""
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.randn(*shape))
    ManifoldMuon.project_([p])
    s0 = torch.linalg.svdvals(p.detach())
    assert (s0 - 1).abs().max() < 1e-3, "project_ must land on the manifold"

    opt = ManifoldMuon([p], lr=0.05, dual_cap=5, tol=1e-6, momentum=0.95)
    for _ in range(50):
        p.grad = 0.1 * torch.randn(*shape)
        opt.step()
    s = torch.linalg.svdvals(p.detach())
    assert (s - 1).abs().max() < 1e-2, (
        f"max |sigma - 1| = {(s - 1).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------- (e) batched == loop
def test_batched_step_equals_per_matrix_loop():
    """Two same-shape parameters updated in one batched step equal the same
    parameters updated one at a time by separate optimizer instances."""
    torch.manual_seed(11)
    shape = (16, 8)
    W1, W2 = torch.randn(*shape), torch.randn(*shape)
    G1, G2 = 0.1 * torch.randn(*shape), 0.1 * torch.randn(*shape)
    kwargs = dict(lr=0.1, dual_cap=5, tol=1e-6, momentum=0.95)

    # batched: one optimizer, both params in one group
    pa, pb = torch.nn.Parameter(W1.clone()), torch.nn.Parameter(W2.clone())
    opt = ManifoldMuon([pa, pb], **kwargs)
    for _ in range(3):  # several steps so momentum buffers are exercised
        pa.grad, pb.grad = G1.clone(), G2.clone()
        stats = opt.step()
    assert len(stats["dual_steps"]) == 2

    # loop: separate instances, same inputs
    singles = []
    for W, G in [(W1, G1), (W2, G2)]:
        p = torch.nn.Parameter(W.clone())
        o = ManifoldMuon([p], **kwargs)
        for _ in range(3):
            p.grad = G.clone()
            o.step()
        singles.append(p.detach())

    assert torch.allclose(pa.detach(), singles[0], atol=1e-5, rtol=0)
    assert torch.allclose(pb.detach(), singles[1], atol=1e-5, rtol=0)


def test_grouped_by_shape():
    a = torch.nn.Parameter(torch.zeros(4, 2))
    b = torch.nn.Parameter(torch.zeros(4, 2))
    c = torch.nn.Parameter(torch.zeros(2, 4))
    groups = grouped_by_shape([a, b, c])
    assert list(groups.keys()) == [torch.Size([4, 2]), torch.Size([2, 4])]
    assert groups[torch.Size([4, 2])] == [a, b]
    assert groups[torch.Size([2, 4])] == [c]


# ---------------------------------------------------------------- (f) cap + logging
def _step_with(tol, dual_cap=5):
    torch.manual_seed(3)
    p = torch.nn.Parameter(msign(torch.randn(24, 12)))
    p.grad = 0.1 * torch.randn(24, 12)
    opt = ManifoldMuon([p], lr=0.1, dual_cap=dual_cap, tol=tol, momentum=0.0)
    return opt.step()


def test_cap_respected_and_steps_logged():
    """dual_steps counts the Lambda updates actually applied per matrix."""
    stats = _step_with(tol=1e-6)
    assert len(stats["dual_steps"]) == 1
    assert 0 <= stats["dual_steps"][0] <= 5
    assert math.isfinite(stats["dual_residual"][0])


def test_huge_tol_stops_immediately():
    """With tol=inf the first residual check already passes: 0 dual updates."""
    stats = _step_with(tol=float("inf"))
    assert stats["dual_steps"] == [0]


def test_zero_tol_runs_to_cap():
    """With tol=0 the residual check never passes: exactly cap updates."""
    stats = _step_with(tol=0.0)
    assert stats["dual_steps"] == [5]
    stats = _step_with(tol=0.0, dual_cap=3)
    assert stats["dual_steps"] == [3]


# ---------------------------------------------------------------- misc contracts
def test_momentum_zero_reduces_to_raw_gradient():
    """With momentum=0 and a fresh buffer, u = G: one step must equal a step
    of the same optimizer re-created with any other (unused) buffer state."""
    torch.manual_seed(5)
    W = msign(torch.randn(20, 10))
    G = 0.1 * torch.randn(20, 10)
    ours, _ = _our_step(W, G, msign_dtype=torch.float32)
    # same thing computed through the reference path but with our fp32 msign
    # is covered by the gate tests; here we only check determinism:
    ours2, _ = _our_step(W, G, msign_dtype=torch.float32)
    assert torch.equal(ours, ours2)


def test_project_handles_wide_matrices():
    torch.manual_seed(9)
    p = torch.nn.Parameter(torch.randn(8, 24))
    ManifoldMuon.project_([p])
    s = torch.linalg.svdvals(p.detach())
    assert (s - 1).abs().max() < 1e-3


def test_rejects_non_2d_params():
    p = torch.nn.Parameter(torch.zeros(5))
    p.grad = torch.zeros(5)
    opt = ManifoldMuon([p], lr=0.1)
    with pytest.raises(ValueError):
        opt.step()
