"""Tests for Lion-Muon (muonarch.optim.LionMuon, optimizer name "lionmuon").

Reference: brain-lab-research/lion-muon @ e9077a8, src/optim/lion_muon.py,
Algorithm 1 (single process, non-adaptive, nesterov=False), vendored
unmodified under third_party/lion_muon/.

Two oracles:

1.  ``reference_transcription`` — an independent line-by-line fp32
    transcription of the reference update written in this file (its own
    Newton-Schulz, the reference's op order, the reference's global step
    counter and ``lion_lr_ratio``). It shares no code with muonarch. Asserted
    tight (atol 1e-6).

2.  The REAL reference class, imported from third_party/. Two things in it
    cannot run here as written, and are redirected from the outside (the
    vendored files are untouched):
      - ``torch.zeros(..., device="cuda", dtype=torch.bfloat16)`` for its
        flat update buffer: the test gives the reference module a ``torch``
        proxy whose ``zeros`` maps device "cuda" to "cpu" (this Mac has no
        CUDA).
      - ``@torch.compile`` on its Newton-Schulz: made a no-op while the
        module is imported (compilation is not part of the algorithm).
    One mismatch is unavoidable: the reference runs Newton-Schulz in bfloat16
    AND rounds every update through a bfloat16 buffer, the project runs fp32.
    Sign updates are +-1/0, exactly representable in bf16, so SIGN steps are
    compared tightly; MUON steps get a tolerance justified by bf16 (see
    ``test_real_reference_*``).

Gradients in the oracle tests are a fixed pre-drawn sequence (independent of
the parameters), so both sides see bit-identical gradients and EMA buffers
and a sign can never flip between them through a 1e-9 parameter difference.
"""

from __future__ import annotations

import importlib
import io
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from muonarch import flops as flops_mod
from muonarch import runner, train
from muonarch.models import build_dense, build_mlp, build_moe
from muonarch.optim import LionMuon, build_optimizers, ns, scale, split_params

from tests.test_train import mlp_cfg, write_cifar_world

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / "third_party" / "lion_muon" / "src" / "optim"

TALL, WIDE = (24, 10), (8, 20)


# ---------------------------------------------------------------- helpers

def draw_grads(shapes, n_steps, seed=0):
    """grads[t][i]: a fixed gradient sequence, independent of the params."""
    gen = torch.Generator().manual_seed(seed)
    return [[torch.randn(s, generator=gen) for s in shapes] for _ in range(n_steps)]


def init_params(shapes, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return [nn.Parameter(torch.randn(s, generator=gen)) for s in shapes]


def lr_schedule(n_steps, base):
    """A schedule that changes the lr on every step (warmup, plateau, decay)."""
    mults = [0.25, 0.5, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    mults = (mults * (n_steps // len(mults) + 1))[:n_steps]
    return [base * m for m in mults]


def run_ours(params, grads, lrs, **kw):
    """Step LionMuon over the gradient sequence, rewriting group["lr"] before
    each step exactly as train.py's scheduler does. Returns per-step param
    snapshots and the per-step last_step_muon flags."""
    opt = LionMuon(params, lr=lrs[0], **kw)
    snaps, flags = [], []
    for g_t, lr in zip(grads, lrs):
        for group in opt.param_groups:
            group["lr"] = lr
        for p, g in zip(params, g_t):
            p.grad = g.clone()
        opt.step()
        snaps.append([p.detach().clone() for p in params])
        flags.append(opt.last_step_muon)
    return snaps, flags, opt


# ------------------------------------------- oracle 1: transcription (fp32)

def _ref_newtonschulz5_fp32(G, steps, eps=1e-7):
    """reference src/optim/muon.py zeropower_via_newtonschulz5, line by line,
    with the single change X = G.float() instead of G.bfloat16()."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float().clone()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def reference_transcription(params, grads, lrs, *, base_lr, alpha, beta1, beta2,
                            wd, period, ns_steps=5):
    """reference src/optim/lion_muon.py LionMuon.step(), the single-process
    non-adaptive nesterov=False path, line by line on plain tensors."""
    ps = [p.detach().clone() for p in params]
    lion_lr = base_lr / alpha                 # run_experiments.sh: slr = mlr / alpha
    lion_lr_ratio = lion_lr / base_lr         # __init__
    exp_avg = [None] * len(ps)
    step_count = 0
    snaps = []
    for g_t, group_lr in zip(grads, lrs):
        is_muon_global = period <= 1 or (step_count % period) == 0
        step_count += 1
        muon_lr = group_lr
        sign_lr = lion_lr_ratio * group_lr
        lr = muon_lr if is_muon_global else sign_lr
        updates = []
        for i, (p, g) in enumerate(zip(ps, g_t)):
            if exp_avg[i] is None:
                exp_avg[i] = torch.zeros_like(g)
            m = exp_avg[i]
            update = m * beta1 + g * (1 - beta1)
            if is_muon_global:
                update = _ref_newtonschulz5_fp32(update, steps=ns_steps)
                update = update * (0.2 * max(update.size(0), update.size(1)) ** 0.5)
            else:
                update = update.sign()
            updates.append(update)
            m.mul_(beta2).add_(g, alpha=1 - beta2)
        for p, u in zip(ps, updates):
            if wd > 0:
                p.mul_(1 - lr * wd)
            p.add_(u, alpha=-lr)
        snaps.append([p.clone() for p in ps])
    return snaps


@pytest.mark.parametrize("period", [5, 3])
def test_matches_reference_transcription(period):
    """14 steps, tall + wide matrix, wd > 0, lr changing every step: Muon
    steps at t % period == 0 and sign steps in between, all against the
    line-by-line transcription. Measured max |diff| (CPU): see atol."""
    n, base, alpha, b1, b2, wd = 14, 0.02, 30.0, 0.9, 0.99, 0.1
    shapes = [TALL, WIDE]
    grads = draw_grads(shapes, n)
    lrs = lr_schedule(n, base)
    assert len(set(lrs)) > 5                       # the lr really moves
    ref = reference_transcription(init_params(shapes), grads, lrs, base_lr=base,
                                  alpha=alpha, beta1=b1, beta2=b2, wd=wd,
                                  period=period)
    ours, flags, _ = run_ours(init_params(shapes), grads, lrs, wd=wd, alpha=alpha,
                              beta1=b1, beta2=b2, period=period)
    assert flags == [t % period == 0 for t in range(n)]
    assert True in flags and False in flags        # both step types covered
    for t in range(n):
        for o, r in zip(ours[t], ref[t]):
            assert torch.allclose(o, r, atol=1e-6, rtol=0), (t, (o - r).abs().max())


def test_sign_step_rate_follows_the_schedule():
    """The sign-step rate is derived from the group's CURRENT lr: with wd = 0
    every entry moves by exactly lr_t / alpha on a sign step."""
    n, base, alpha = 9, 0.02, 10.0
    grads = draw_grads([WIDE], n)
    lrs = lr_schedule(n, base)
    params = init_params([WIDE])
    start = params[0].detach().clone()
    snaps, flags, opt = run_ours(params, grads, lrs, wd=0.0, alpha=alpha,
                                 beta1=0.9, beta2=0.99, period=5)
    assert opt.param_groups[0]["sign_lr_ratio"] == pytest.approx(1 / alpha)
    prev = start
    for t in range(n):
        delta = (snaps[t][0] - prev).abs()
        if not flags[t]:
            assert torch.allclose(delta, torch.full_like(delta, lrs[t] / alpha),
                                  rtol=1e-4, atol=1e-7), t
        prev = snaps[t][0]


# ------------------------------------------- oracle 2: the real reference

class _CpuTorch:
    """``torch`` as the reference module sees it: everything forwarded, except
    ``zeros(device="cuda")`` lands on the CPU."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def zeros(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return torch.zeros(*args, **kwargs)


@pytest.fixture(scope="module")
def ref_cls():
    """The vendored reference LionMuon class (files untouched)."""
    assert (REF_DIR / "lion_muon.py").is_file(), "third_party/lion_muon missing"
    mp = pytest.MonkeyPatch()
    mp.delenv("WORLD_SIZE", raising=False)      # single process
    mp.delenv("RANK", raising=False)
    pkg_name = "_lionmuon_reference"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(REF_DIR)]
    sys.modules[pkg_name] = pkg
    try:
        # @torch.compile on the reference Newton-Schulz: a no-op, and only
        # while the reference modules are being imported
        with pytest.MonkeyPatch.context() as imp:
            imp.setattr(torch, "compile",
                        lambda f=None, **kw: f if f is not None else (lambda g: g))
            mod = importlib.import_module(f"{pkg_name}.lion_muon")
        mod.torch = _CpuTorch()
        yield mod.LionMuon
    finally:
        mp.undo()
        for name in [n for n in sys.modules if n.startswith(pkg_name)]:
            del sys.modules[name]


def run_reference(ref_cls, params, grads, lrs, *, base_lr, alpha, beta1, beta2,
                  wd, period, sync_to=None):
    """Step the real reference. ``sync_to`` (per-step snapshots of another
    run): copy that run's parameters in BEFORE each step, so each step is
    compared in isolation (the EMA buffers need no sync: they depend on the
    gradients only)."""
    opt = ref_cls(params, lr=base_lr, lion_lr=base_lr / alpha, beta1=beta1,
                  beta2=beta2, ns_steps=5, muon_every_k=period, nesterov=False,
                  weight_decay=wd)
    assert opt.world_size == 1
    snaps = []
    for t, (g_t, lr) in enumerate(zip(grads, lrs)):
        for group in opt.param_groups:
            group["lr"] = lr
        if sync_to is not None and t > 0:
            with torch.no_grad():
                for p, src in zip(params, sync_to[t - 1]):
                    p.copy_(src)
        for p, g in zip(params, g_t):
            p.grad = g.clone()
        opt.step()
        snaps.append([p.detach().clone() for p in params])
    return snaps


#: Muon-step tolerances against the real (bf16) reference. bf16 keeps 8
#: significand bits; the reference runs all 5 Newton-Schulz iterations in it
#: (the quintic amplifies round-off in small singular directions) and then
#: rounds the update through a bf16 buffer. Measured on this machine (CPU,
#: this file's 14-step case, parameters synced before every step):
#:   worst max-entry error  = 0.029 x (lr_t * 0.2 * sqrt(max(m, n)))
#:   worst Frobenius error  = 0.041 x ||update||_F
#: The same convention as tests/test_optim.py's stock-torch-Muon oracle
#: (atol = 0.1 x adjusted lr): about 3x the measured worst case, and an order
#: of magnitude below the update itself. Sign steps are NOT covered by these:
#: they are compared at fp32 round-off.
BF16_ENTRY_TOL = 0.1
BF16_FRO_TOL = 0.12


def test_real_reference_step_by_step(ref_cls):
    """Each step in isolation against the real reference (parameters synced
    before every step). Sign steps: identical direction and matching size to
    fp32 round-off. Muon steps: equal up to the reference's bf16."""
    n, base, alpha, b1, b2, wd, period = 14, 0.02, 30.0, 0.9, 0.99, 0.1, 5
    shapes = [TALL, WIDE]
    grads = draw_grads(shapes, n)
    lrs = lr_schedule(n, base)
    start = [p.detach().clone() for p in init_params(shapes)]
    ours, flags, _ = run_ours(init_params(shapes), grads, lrs, wd=wd, alpha=alpha,
                              beta1=b1, beta2=b2, period=period)
    ref = run_reference(ref_cls, init_params(shapes), grads, lrs, base_lr=base,
                        alpha=alpha, beta1=b1, beta2=b2, wd=wd, period=period,
                        sync_to=ours)
    worst_muon = 0.0
    for t in range(n):
        before = start if t == 0 else ours[t - 1]
        for o, r, b, shape in zip(ours[t], ref[t], before, shapes):
            if flags[t]:
                step_mag = lrs[t] * scale(torch.Size(shape), "moonlight")
                err = float((o - r).abs().max()) / step_mag
                worst_muon = max(worst_muon, err)
                assert err < BF16_ENTRY_TOL, (t, err)
                update = o - b * (1 - lrs[t] * wd)
                assert float((o - r).norm() / update.norm()) < BF16_FRO_TOL, t
            else:
                # the decayed point both sides start the sign move from
                sign_lr = lrs[t] / alpha
                d_o = o - b * (1 - sign_lr * wd)
                d_r = r - b * (1 - sign_lr * wd)
                assert torch.equal(torch.sign(d_o), torch.sign(d_r)), t
                assert torch.allclose(o, r, atol=1e-6, rtol=0), t
    assert worst_muon > 0.0     # bf16 really differs: the oracle is not ours


def test_real_reference_free_running_trajectory(ref_cls):
    """No syncing: both run the same 14-step schedule from the same init.
    The only divergence is the bf16 error of each Muon step (3 of them),
    which accumulates additively because the gradients are fixed."""
    n, base, alpha, b1, b2, wd, period = 14, 0.02, 30.0, 0.9, 0.99, 0.1, 5
    shapes = [TALL, WIDE]
    grads = draw_grads(shapes, n)
    lrs = lr_schedule(n, base)
    ours, flags, _ = run_ours(init_params(shapes), grads, lrs, wd=wd, alpha=alpha,
                              beta1=b1, beta2=b2, period=period)
    ref = run_reference(ref_cls, init_params(shapes), grads, lrs, base_lr=base,
                        alpha=alpha, beta1=b1, beta2=b2, wd=wd, period=period)
    for i, shape in enumerate(shapes):
        budget = 0.0
        for t in range(n):
            if flags[t]:
                budget += BF16_ENTRY_TOL * lrs[t] * scale(torch.Size(shape), "moonlight")
            assert float((ours[t][i] - ref[t][i]).abs().max()) <= budget + 1e-6, t


# ---------------------------------------------------------------- limits

def test_period_one_is_pure_moonlight_scaled_ns_of_the_ema():
    """P = 1: never a sign step; every update is ns(u) * moonlight scale."""
    n, lr, wd, b1, b2 = 6, 0.01, 0.05, 0.9, 0.99
    shapes = [TALL, WIDE]
    grads = draw_grads(shapes, n)
    params = init_params(shapes)
    prev = [p.detach().clone() for p in params]
    m = [torch.zeros(s) for s in shapes]
    opt = LionMuon(params, lr=lr, wd=wd, alpha=30.0, beta1=b1, beta2=b2, period=1)
    for t in range(n):
        for p, g in zip(params, grads[t]):
            p.grad = g.clone()
        opt.step()
        assert opt.last_step_muon is True
        for i, p in enumerate(params):
            u = b1 * m[i] + (1 - b1) * grads[t][i]
            want = prev[i] * (1 - lr * wd) - lr * ns(u) * scale(p.shape, "moonlight")
            assert torch.allclose(p, want, atol=1e-6, rtol=0), (t, i)
            m[i] = b2 * m[i] + (1 - b2) * grads[t][i]
            prev[i] = p.detach().clone()


def test_huge_period_takes_one_muon_step_then_only_sign_steps():
    n, lr, alpha = 8, 0.01, 10.0
    grads = draw_grads([TALL], n)
    params = init_params([TALL])
    snaps, flags, _ = run_ours(params, grads, [lr] * n, wd=0.0, alpha=alpha,
                               beta1=0.9, beta2=0.99, period=10 ** 9)
    assert flags == [True] + [False] * (n - 1)
    for t in range(1, n):
        delta = (snaps[t][0] - snaps[t - 1][0]).abs()
        assert torch.allclose(delta, torch.full_like(delta, lr / alpha),
                              rtol=1e-4, atol=1e-7), t


def test_stacked_expert_tensor_equals_independent_matrices():
    """A (k, m, n) stack is k independent matrices with the trailing-two-dims
    Moonlight scale — like the rest of the Muon family."""
    k, shape, n = 3, (6, 10), 7
    gen = torch.Generator().manual_seed(3)
    init = torch.randn(k, *shape, generator=gen)
    grads = [torch.randn(k, *shape, generator=gen) for _ in range(n)]
    kw = dict(wd=0.1, alpha=30.0, beta1=0.9, beta2=0.99, period=3)
    stacked = nn.Parameter(init.clone())
    singles = [nn.Parameter(init[j].clone()) for j in range(k)]
    o_stack = LionMuon([stacked], lr=0.02, **kw)
    o_single = LionMuon(singles, lr=0.02, **kw)
    for g in grads:
        stacked.grad = g.clone()
        for j, p in enumerate(singles):
            p.grad = g[j].clone()
        o_stack.step()
        o_single.step()
    for j, p in enumerate(singles):
        assert torch.allclose(stacked[j], p, atol=1e-6, rtol=0)


def test_rejects_vectors_and_bad_hypers():
    with pytest.raises(ValueError):
        opt = LionMuon([nn.Parameter(torch.ones(4))], lr=0.1)
        opt.param_groups[0]["params"][0].grad = torch.ones(4)
        opt.step()
    w = nn.Parameter(torch.ones(3, 4))
    for bad in (dict(alpha=0.0), dict(period=0), dict(beta1=1.0), dict(beta2=1.0),
                dict(wd=-0.1)):
        with pytest.raises(ValueError):
            LionMuon([w], lr=0.1, **bad)
    with pytest.raises(ValueError):
        LionMuon([w], lr=-0.1)


def test_step_counter_is_global_across_parameters():
    """A parameter without a gradient on some step still advances its
    counter, so all matrices take their Muon steps on the same global steps
    (the reference's single step counter) — it matters for MoE experts that
    receive no tokens on a step."""
    a, b = init_params([TALL, WIDE])
    opt = LionMuon([a, b], lr=0.01, wd=0.0, alpha=10.0, period=3)
    grads = draw_grads([TALL, WIDE], 7)
    b_moves = []
    for t in range(7):
        a.grad = grads[t][0].clone()
        b.grad = None if t in (1, 2, 3) else grads[t][1].clone()
        before = b.detach().clone()
        opt.step()
        b_moves.append((b.detach() - before).abs())
        assert opt.last_step_muon == (t % 3 == 0)
    assert opt.state[a]["step"] == opt.state[b]["step"] == 7
    for t in (1, 2, 3):
        assert float(b_moves[t].max()) == 0.0       # untouched without a grad
    # t = 6 is a global Muon step for b too (its own 4th update): not a sign step
    assert not torch.allclose(b_moves[6], torch.full_like(b_moves[6], 0.001),
                              rtol=1e-3, atol=0)
    for t in (4, 5):
        assert torch.allclose(b_moves[t], torch.full_like(b_moves[t], 0.001),
                              rtol=1e-3, atol=1e-7)


# ---------------------------------------------------------------- resume

def test_checkpoint_resume_is_bit_exact():
    """N steps straight == k steps, state_dict round trip (through torch.save)
    into a FRESH optimizer on fresh parameters, N - k steps. k = 4 is not a
    multiple of the period, so a lost or reset step counter would move the
    Muon steps and break equality."""
    n, k, period = 13, 4, 5
    shapes = [TALL, WIDE]
    grads = draw_grads(shapes, n)
    lrs = lr_schedule(n, 0.02)
    kw = dict(wd=0.1, alpha=30.0, beta1=0.9, beta2=0.99, period=period)

    straight, flags_straight, _ = run_ours(init_params(shapes), grads, lrs, **kw)

    first, _, opt1 = run_ours(init_params(shapes), grads[:k], lrs[:k], **kw)
    buf = io.BytesIO()
    torch.save({"opt": opt1.state_dict(), "params": first[-1]}, buf)
    buf.seek(0)
    ck = torch.load(buf, weights_only=False)

    params2 = [nn.Parameter(p.clone()) for p in ck["params"]]
    opt2 = LionMuon(params2, lr=lrs[0], **kw)
    opt2.load_state_dict(ck["opt"])
    assert [opt2.state[p]["step"] for p in params2] == [k, k]
    flags2 = []
    for t in range(k, n):
        for group in opt2.param_groups:
            group["lr"] = lrs[t]
        for p, g in zip(params2, grads[t]):
            p.grad = g.clone()
        opt2.step()
        flags2.append(opt2.last_step_muon)
        for p, want in zip(params2, straight[t]):
            assert torch.equal(p, want), t
    assert flags2 == flags_straight[k:]
    assert flags2 == [t % period == 0 for t in range(k, n)]


# ---------------------------------------------------------------- param split

@pytest.mark.parametrize("arch", ["mlp", "dense", "moe"])
def test_param_split_equals_muons(arch):
    torch.manual_seed(0)
    model = {"mlp": lambda: build_mlp(128), "dense": build_dense,
             "moe": build_moe}[arch]()
    lion = split_params(model, "lionmuon", arch)
    muon = split_params(model, "muon", arch)
    assert [n for n, _ in lion.muon] == [n for n, _ in muon.muon]
    assert [n for n, _ in lion.adamw] == [n for n, _ in muon.adamw]
    assert lion.muon and lion.adamw
    assert all(p.ndim >= 2 for _, p in lion.muon)


def _lion_build_cfg(arch):
    return {
        "optimizer": "lionmuon", "arch": arch, "lr": 1e-3, "wd": 0.1,
        "momentum": None, "adamw_betas": [0.9, 0.95], "adamw_eps": 1e-8,
        "adamw_group_lr": 3e-4, "adamw_group_wd": 0.05,
        "ns_steps": 5, "ns_dtype": "fp32",
        "lionmuon": {"period": 5, "alpha": 30.0, "beta1": 0.9, "beta2": 0.99},
    }


def test_build_optimizers_groups_and_hypers():
    """Matrix group: LionMuon at the run's own (lr, wd) and the hashed
    lionmuon dict; AdamW group: adamw_group_lr/wd with betas (0.9, 0.95)."""
    torch.manual_seed(0)
    model = build_mlp(128)
    opts, groups = build_optimizers(_lion_build_cfg("mlp"), model)
    assert isinstance(opts[0], LionMuon) and isinstance(opts[1], torch.optim.AdamW)
    g = opts[0].param_groups[0]
    assert (g["lr"], g["wd"], g["beta1"], g["beta2"], g["period"]) == (
        1e-3, 0.1, 0.9, 0.99, 5)
    assert g["sign_lr_ratio"] == pytest.approx(1 / 30)
    assert (g["ns_steps"], g["ns_dtype"]) == (5, torch.float32)
    assert [id(p) for p in g["params"]] == [id(p) for _, p in groups.muon]
    a = opts[1].param_groups[0]
    assert (a["lr"], a["weight_decay"], a["betas"]) == (3e-4, 0.05, (0.9, 0.95))
    assert [id(p) for p in a["params"]] == [id(p) for _, p in groups.adamw]
    cfg = _lion_build_cfg("mlp")
    del cfg["lionmuon"]["alpha"]
    with pytest.raises(KeyError):        # no silent default for a hashed value
        build_optimizers(cfg, model)


def test_one_real_step_on_moe():
    """One real step on the MoE (some experts get no tokens, so some
    matrices have no gradient): finite, and something moved."""
    torch.manual_seed(0)
    model = build_moe()
    ids = torch.randint(0, 4096, (2, 32))
    loss = model(input_ids=ids, labels=ids).loss
    opts, groups = build_optimizers(_lion_build_cfg("moe"), model)
    model.zero_grad(set_to_none=True)
    loss.backward()
    before = [p.detach().clone() for p in model.parameters()]
    for opt in opts:
        opt.step()
    assert opts[0].last_step_muon is True
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert all(torch.isfinite(p).all() for p in after)
    assert {opts[0].state[p]["step"] for _, p in groups.muon} == {1}


# ---------------------------------------------------------------- end to end

def lion_cfg(data_path, **over):
    cfg = mlp_cfg(
        data_path, optimizer="lionmuon", rule="moonlight", lr=0.01, wd=0.1,
        momentum=None, adamw_group_lr=0.01, adamw_group_wd=0.0,
        lionmuon={"period": 5, "alpha": 30.0, "beta1": 0.9, "beta2": 0.99})
    cfg.update(over)
    return cfg


def test_train_run_end_to_end_idempotent_and_exact_flops(tmp_path):
    """A tiny lionmuon run through train.run: finite loss, idempotent on its
    hash, and recorded optimizer FLOPs = ceil(N / 5) Muon-step costs + the
    per-parameter overhead on every step — the Muon steps actually taken."""
    data_root = write_cifar_world(tmp_path / "cifar5m")
    cfg = lion_cfg(data_root)
    n = cfg["budget_base_steps"]
    assert n == 16
    res = train.run(cfg, tmp_path)
    assert res["status"] == "completed"
    assert math.isfinite(res["final_val_loss"])
    assert res["hash"] == runner.config_hash(cfg)
    assert res["config"]["lionmuon"] == cfg["lionmuon"]
    assert runner.result_path(tmp_path, cfg).is_file()

    torch.manual_seed(0)
    model = build_mlp(16)
    groups = split_params(model, "lionmuon", "mlp")
    mats = [tuple(p.shape) for _, p in groups.muon]
    vecs = [p.numel() for _, p in groups.adamw]
    n_params = sum(p.numel() for p in model.parameters())
    ns_cost = sum(flops_mod.ns_matrix_flops(s, 5) for s in mats)
    n_muon = math.ceil(n / 5)
    assert n_muon == 4                                  # t = 0, 5, 10, 15
    want = n_muon * ns_cost + n * 10 * n_params
    assert res["flops"]["optimizer"] == want
    assert want == flops_mod.lionmuon_optimizer_flops(mats, vecs, n_steps=n, period=5)
    # ... and it is what the planner estimates for the same run
    assert runner.estimate_flops(cfg) == (
        res["flops"]["model"] + res["flops"]["optimizer"])
    # the Moonlight run on the same model pays NS on every step
    moon = n * (ns_cost + 10 * n_params)
    assert moon - want == (n - n_muon) * ns_cost

    # idempotent: the second call returns the stored result, runs nothing
    stamp = runner.result_path(tmp_path, cfg).stat().st_mtime_ns
    res2 = train.run(cfg, tmp_path)
    assert res2 == json.loads(runner.result_path(tmp_path, cfg).read_text())
    assert res2["finished_at"] == res["finished_at"]
    assert runner.result_path(tmp_path, cfg).stat().st_mtime_ns == stamp


def test_hash_separates_every_lionmuon_value(tmp_path):
    cfg = lion_cfg(tmp_path)
    h = runner.config_hash(cfg)
    for key, value in (("alpha", 10.0), ("beta2", 0.95), ("period", 4),
                       ("beta1", 0.95)):
        other = dict(cfg, lionmuon=dict(cfg["lionmuon"], **{key: value}))
        assert runner.config_hash(other) != h, key
