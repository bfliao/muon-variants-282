"""Vectorized dual ascent (Gate 3 speed change) vs the original per-matrix loop.

The vectorized path (manifold._dual_ascent_vectorized) replaces the
per-matrix Python bookkeeping (one .item() sync per matrix per iteration)
with masked tensor ops. It must be a pure speed change: identical updated
parameters, dual variables, per-matrix dual-step counts, residuals, msign
call counts and FLOPs. The loop path stays behind the internal flag
manifold._VECTORIZED_DUAL_ASCENT as the oracle.

Tests (Paul's Gate-3 brief, Step 3):
1. equivalence on random same-shape stacks, G in {1, 4, 16}, several shapes,
   CPU and MPS (max abs/rel error printed with -s);
2. staggered early stop: per-matrix counts identical to the loop AND to
   each matrix run alone (the truly unbatched count);
3. per-step FLOPs identical between paths, including skipped (grad None)
   experts;
4. end to end: ~50-step dense and MoE runs, per-step losses equal;
(5. the full suite is run separately.)
"""

import pytest
import torch

from muonarch import manifold as mm
from muonarch.manifold import ManifoldMuon
from muonarch.msign import msign

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
SHAPES = [(32, 16), (16, 32), (128, 96), (96, 128), (64, 64)]


@pytest.fixture
def use_path(monkeypatch):
    def _set(vectorized: bool):
        monkeypatch.setattr(mm, "_VECTORIZED_DUAL_ASCENT", vectorized)
    return _set


def _tall_inputs(k, shape, device, seed):
    g = torch.Generator().manual_seed(seed)
    W = msign(torch.randn(k, *shape, generator=g)).to(device)
    G = (0.1 * torch.randn(k, *shape, generator=g)).to(device)
    if W.size(-2) < W.size(-1):  # the ascent functions take tall stacks
        W, G = W.mT.contiguous(), G.mT.contiguous()
    return W, G


def _errs(a, b):
    d = (a - b).abs()
    return d.max().item(), (d / b.abs().clamp_min(1e-12)).max().item()


# ---------------------------------------------------------------------------
# 1. equivalence on random stacks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("k", [1, 4, 16])
@pytest.mark.parametrize("cap", [5, 10])
def test_dual_ascent_equivalence(device, shape, k, cap):
    W, G = _tall_inputs(k, shape, device, seed=k * 1000 + shape[0] + cap)
    args = (W, G, cap, 1e-6, 0.01, 10, torch.float32)
    A0, L0, s0, r0, c0 = mm._dual_ascent_loop(*args)
    A1, L1, s1, r1, c1 = mm._dual_ascent_vectorized(*args)
    assert s1 == s0 and c1 == c0 and r1 == r0
    for x, y, name in ((A1, A0, "A"), (L1, L0, "Lambda")):
        abs_e, rel_e = _errs(x, y)
        print(f"{device} k={k} {shape} cap={cap} {name}: max abs {abs_e:.2e} rel {rel_e:.2e}")
        assert torch.equal(x, y), (name, abs_e, rel_e)  # bit-identical


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k", [1, 4, 16])
def test_optimizer_step_equivalence(device, k, use_path):
    """Full ManifoldMuon steps (momentum, both orientations, several shape
    groups) give the same parameters and stats on both paths."""
    shapes = [(32, 16), (16, 32), (64, 64)]
    g = torch.Generator().manual_seed(k)
    init = [torch.randn(s, generator=g) for s in shapes for _ in range(k)]
    grads = [[0.1 * torch.randn(w.shape, generator=g) for w in init] for _ in range(3)]

    def run(vectorized):
        use_path(vectorized)
        ps = [torch.nn.Parameter(w.clone().to(device)) for w in init]
        ManifoldMuon.project_(ps)
        opt = ManifoldMuon(ps, lr=0.1, dual_cap=5)
        stats = []
        for gs in grads:
            for p, gr in zip(ps, gs):
                p.grad = gr.to(device)
            stats.append(opt.step())
        return [p.detach().cpu() for p in ps], stats

    p0, st0 = run(False)
    p1, st1 = run(True)
    for a, b in zip(p1, p0):
        abs_e, rel_e = _errs(a, b)
        assert torch.equal(a, b), (abs_e, rel_e)
    for s1, s0 in zip(st1, st0):
        assert s1 == s0  # dual_steps, dual_residual, msign_calls all exact


# ---------------------------------------------------------------------------
# 2. staggered early stop
# ---------------------------------------------------------------------------

def _staggered_stack():
    """Six tall matrices on the Stiefel manifold whose gradients are tangent
    plus noise of growing size, so they converge at different iterations
    (found by search; asserted below so a silent change cannot hide it)."""
    torch.manual_seed(0)
    k, m, n = 6, 32, 16
    W = msign(torch.randn(k, m, n))
    S = torch.randn(k, n, n)
    S = S - S.mT
    eps = torch.tensor([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]).view(k, 1, 1)
    G = W @ S + eps * torch.randn(k, m, n)
    return W, G


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tol", [3e-4, 3e-3])
def test_staggered_early_stop_counts(tol, device):
    W, G = (t.to(device) for t in _staggered_stack())
    args = (W, G, 20, tol, 0.3, 10, torch.float32)
    A0, L0, s0, r0, c0 = mm._dual_ascent_loop(*args)
    A1, L1, s1, r1, c1 = mm._dual_ascent_vectorized(*args)
    # the case really is staggered: some stop at 0, one strictly between 0 and cap
    assert 0 in s0 and 20 in s0 and any(0 < s < 20 for s in s0), s0
    assert s1 == s0 and c1 == c0 and r1 == r0
    assert torch.equal(L1, L0) and torch.equal(A1, A0)
    # truly unbatched: each matrix alone gets the same count as in the stack
    for i in range(W.size(0)):
        _, Li, si, _, _ = mm._dual_ascent_vectorized(
            W[i:i + 1], G[i:i + 1], 20, tol, 0.3, 10, torch.float32)
        assert si == [s1[i]], (i, si, s1[i])
        assert torch.allclose(Li[0], L1[i], atol=1e-6, rtol=1e-5)


def test_all_converged_breaks_and_counts_calls():
    """When every matrix converges at iteration 0 the loop breaks after one
    ascent call on both paths (the early-stop check call is counted)."""
    W, G = _staggered_stack()
    args = (W[:1], G[:1], 10, 1e-3, 0.01, 10, torch.float32)
    *_, s0, r0, c0 = mm._dual_ascent_loop(*args)
    *_, s1, r1, c1 = mm._dual_ascent_vectorized(*args)
    assert s0 == s1 == [0] and c0 == c1 == 1


def test_whole_stack_converges_at_iteration_zero():
    W, G = _staggered_stack()
    args = (W, G, 10, 1.0, 0.01, 10, torch.float32)
    A0, L0, s0, r0, c0 = mm._dual_ascent_loop(*args)
    A1, L1, s1, r1, c1 = mm._dual_ascent_vectorized(*args)
    assert s0 == s1 == [0] * 6 and c0 == c1 == 1 and r0 == r1
    assert torch.equal(A0, A1) and torch.equal(L0, L1)


def test_cap_zero_fallback_matches():
    W, G = _tall_inputs(4, (32, 16), "cpu", seed=3)
    out0 = mm._dual_ascent_loop(W, G, 0, 1e-6, 0.01, 10, torch.float32)
    out1 = mm._dual_ascent_vectorized(W, G, 0, 1e-6, 0.01, 10, torch.float32)
    assert out0[2] == out1[2] == [0, 0, 0, 0] and out0[4] == out1[4] == 1
    assert torch.equal(out0[0], out1[0])


# ---------------------------------------------------------------------------
# 3. FLOPs identical, including skipped experts
# ---------------------------------------------------------------------------

def test_step_flops_identical_with_skipped_experts(use_path):
    from muonarch.train import _manifold_step_flops

    g = torch.Generator().manual_seed(7)
    shapes = [(96, 128)] * 8 + [(128, 96)] * 8 + [(64, 128)] * 2
    init = [torch.randn(s, generator=g) for s in shapes]
    skip = {1, 4, 9, 15}  # "unrouted experts": grad None this step

    def run(vectorized):
        use_path(vectorized)
        ps = [torch.nn.Parameter(w.clone()) for w in init]
        ManifoldMuon.project_(ps)
        opt = ManifoldMuon(ps, lr=0.1, dual_cap=5)
        flops = []
        for step in range(3):
            for i, p in enumerate(ps):
                p.grad = None if (i in skip and step != 0) else 0.1 * torch.randn(
                    p.shape, generator=torch.Generator().manual_seed(100 * step + i))
            st = opt.step()
            flops.append(_manifold_step_flops(opt, st["msign_calls"], 10))
            assert len(st["msign_calls"]) == sum(p.grad is not None for p in ps)
        return flops, [p.detach() for p in ps]

    f0, p0 = run(False)
    f1, p1 = run(True)
    assert f1 == f0 and f0[1] < f0[0]  # skipped experts really cost nothing
    for a, b in zip(p1, p0):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 4. end to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["dense", "moe"])
def test_end_to_end_losses_match(arch, use_path):
    from muonarch.models import build_dense, build_moe
    from muonarch.optim import split_params
    from muonarch.train import clip_gradients, lm_forward_loss

    steps, batch, seq = 50, 4, 64
    g = torch.Generator().manual_seed(1)
    data = torch.randint(0, 4096, (steps, batch, seq + 1), generator=g)

    def run(vectorized):
        use_path(vectorized)
        torch.manual_seed(0)
        model = (build_dense if arch == "dense" else build_moe)().float()
        groups = split_params(model, "manifold", arch)
        muon = [p for _, p in groups.muon]
        ManifoldMuon.project_(muon)
        opts = [ManifoldMuon(muon, lr=0.05, dual_cap=5),
                torch.optim.AdamW([p for _, p in groups.adamw], lr=1e-3,
                                  betas=(0.9, 0.95), weight_decay=0.1)]
        losses, calls = [], []
        for t in range(steps):
            x, y = data[t, :, :-1], data[t, :, 1:]
            loss, _, _ = lm_forward_loss(model, x, y, 0.01 if arch == "moe" else None)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            clip_gradients(model.parameters(), 1.0)
            st = opts[0].step()
            opts[1].step()
            losses.append(loss.item())
            calls.append(st["msign_calls"])
        return losses, calls

    l0, c0 = run(False)
    l1, c1 = run(True)
    assert c1 == c0
    worst = max(abs(a - b) for a, b in zip(l1, l0))
    print(f"{arch}: max per-step loss diff over {steps} steps = {worst:.2e}")
    assert worst < 1e-4, worst
