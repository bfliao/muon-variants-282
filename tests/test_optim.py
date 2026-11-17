"""Tests for muonarch/optim.py (ns, ns_vec, scale, MuonFamily) plus the
ManifoldMuon ``msign_calls`` accounting added at Gate 1.

torch.optim.Muon convention mappings (torch 2.15, read from
.venv/.../torch/optim/_muon.py before writing these tests)
----------------------------------------------------------------------
PLAN.md's reference code is the specification; torch is the oracle. The
mappings needed to compare one step of MuonFamily against torch.optim.Muon:

1.  Momentum form. torch keeps an EMA buffer, ``buf.lerp_(grad, 1-mu)``
    (i.e. buf = mu*buf + (1-mu)*grad) and uses the Nesterov update
    ``grad.lerp(buf, mu)`` = (1-mu)*grad + mu*buf. PLAN keeps the sum form
    buf = mu*buf + grad, u = grad + mu*buf. By induction torch's buffer is
    exactly (1-mu) times PLAN's, so torch's pre-NS update is exactly
    (1-mu) times PLAN's u. Newton-Schulz normalizes its input by its
    Frobenius norm, so the scalar cancels and the NS outputs agree; no
    configuration is needed, the forms are equivalent through NS.
2.  Nesterov flag. torch default ``nesterov=True`` corresponds to PLAN's
    u = grad + mu*buf; we pass it explicitly.
3.  Weight decay point. Both apply decoupled decay ``p *= 1 - lr*wd`` with
    the UNADJUSTED lr, before subtracting the (shape-scaled) update; torch's
    default weight_decay is 0.1, so wd is always passed explicitly.
4.  Scale rules. ``adjust_lr_fn="original"`` is sqrt(max(1, A/B)) with
    (A, B) = param.shape[:2] = (fan_out, fan_in): PLAN's rule="keller".
    ``adjust_lr_fn="match_rms_adamw"`` is 0.2*sqrt(max(A, B)): PLAN's
    rule="moonlight". torch multiplies the lr used for the update add only,
    identical to PLAN multiplying the update by scale().
5.  NS coefficients / steps / eps. Identical defaults on both sides:
    (3.4445, -4.7750, 2.0315), 5 steps, eps 1e-7. torch guards the
    normalization with ``norm().clamp(min=eps)`` where PLAN adds eps
    (``norm() + 1e-7``); for these inputs (norm >> eps) the relative
    difference is ~1e-7 and far below every tolerance used here.
6.  NS dtype — the one unavoidable mismatch. torch 2.15 hard-codes
    ``grad.to(dtype=torch.bfloat16, copy=True)`` inside
    ``_zeropower_via_newtonschulz``; no constructor argument controls it.
    PLAN specifies fp32. An exact match is therefore impossible against the
    stock oracle. We assert two ways:
      (a) tight: the oracle is "configured to match PLAN's settings" by
          monkeypatching torch.optim._muon._zeropower_via_newtonschulz with
          a verbatim copy whose ONLY change is fp32 instead of bf16 (torch's
          own op order, addmm and clamp retained). Measured max |param diff|
          after 2 steps over all 12 shape/rule/wd cases (CPU, torch 2.15):
          2.4e-7; asserted at atol=1e-5.
      (b) stock oracle: tolerance justified by bf16 resolution (machine eps
          2^-8 ~ 3.9e-3 on NS iterates of order ~1, amplified over 5
          iterations, times the adjusted lr). Measured max |param diff|
          after 2 steps: 3.4e-4 (keller), 4.2e-4 (moonlight); asserted at
          atol = 0.1 * adjusted_lr per shape/rule (2e-3 to 3.9e-3 here),
          >= 5x the measured worst case yet ~10x below the update magnitude.
"""

import io
import math

import pytest
import torch

import torch.optim._muon as torch_muon_mod

from muonarch.optim import NS_COEFFS, MuonFamily, ns, ns_vec, scale
from muonarch.manifold import ManifoldMuon


# --------------------------------------------------------------- constants
def test_ns_coeffs():
    assert NS_COEFFS == (3.4445, -4.7750, 2.0315)
    # and they match torch's (shared-oracle sanity)
    assert NS_COEFFS == (
        torch_muon_mod.DEFAULT_A,
        torch_muon_mod.DEFAULT_B,
        torch_muon_mod.DEFAULT_C,
    )


# --------------------------------------------------------------- ns_vec == diag(ns(diag))
@pytest.mark.parametrize("d", [3, 17, 64, 128])
def test_ns_vec_matches_diag_ns(d):
    """ns_vec(v) == torch.diag(ns(torch.diag(v))).

    The two paths compute the same elementwise quintic (diag(v) stays
    diagonal under every NS iteration); only fp32 rounding order differs
    (B*x**3 + C*x**5 vs (B*x^2 + C*x^4)*x). Measured max |diff| over the
    four sizes (CPU, torch 2.15): 3.0e-6; asserted at atol=1e-5.
    """
    torch.manual_seed(d)
    v = torch.randn(d)
    out_vec = ns_vec(v)
    out_mat = torch.diag(ns(torch.diag(v)))
    assert out_vec.shape == v.shape
    assert torch.allclose(out_vec, out_mat, atol=1e-5, rtol=0), (
        f"max diff {(out_vec - out_mat).abs().max().item():.3e}"
    )


# --------------------------------------------------------------- batched ns
@pytest.mark.parametrize("shape", [(6, 24, 48), (6, 48, 24)], ids=["wide", "tall"])
def test_ns_batched_experts_matches_loop(shape):
    """ns over a stacked (k, m, n) tensor of same-shape 'expert' matrices
    equals a per-matrix loop, tall and wide.

    Batched vs single matmul kernels may accumulate differently; measured
    max |diff| (CPU, torch 2.15): 0.0, bit-identical on both orientations;
    asserted at atol=1e-5 to leave room for other backends' batched kernels.
    """
    torch.manual_seed(shape[-1])
    G = torch.randn(*shape)
    batched = ns(G)
    looped = torch.stack([ns(G[i]) for i in range(shape[0])])
    assert batched.shape == G.shape
    assert torch.allclose(batched, looped, atol=1e-5, rtol=0), (
        f"max diff {(batched - looped).abs().max().item():.3e}"
    )


# --------------------------------------------------------------- scale anchors
def test_scale_anchors():
    """Anchors from design.md section 3.1 / the task brief.

    nn.Linear weight shape is (fan_out, fan_in), so keller((4096, 128)) has
    fan_out/fan_in = 32. Note: the brief's "moonlight((128, 3072)) ~ 11.0905"
    is a typo for 0.2*sqrt(3072) = 11.0851 (design.md's ~11.09); we assert
    the formula value.
    """
    assert scale(torch.Size([128, 3072]), "moonlight") == pytest.approx(
        0.2 * math.sqrt(3072)
    )
    assert scale(torch.Size([128, 3072]), "moonlight") == pytest.approx(11.09, abs=5e-3)
    assert scale(torch.Size([128, 128]), "moonlight") == pytest.approx(2.2627, abs=1e-4)
    assert scale(torch.Size([128, 128]), "keller") == 1.0
    # wide matrices clamp to 1 under keller (fan_out < fan_in)
    assert scale(torch.Size([128, 3072]), "keller") == 1.0
    assert scale(torch.Size([4096, 128]), "keller") == pytest.approx(
        math.sqrt(32)
    )  # ~5.6569
    assert scale(torch.Size([4096, 128]), "keller") == pytest.approx(5.6569, abs=1e-4)
    # batched (expert-stacked) shapes use the trailing two dims
    assert scale(torch.Size([16, 96, 128]), "moonlight") == pytest.approx(
        0.2 * math.sqrt(128)
    )
    with pytest.raises(ValueError):
        scale(torch.Size([4, 4]), "nope")


# --------------------------------------------------------------- fp32 defaults
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_ns_outputs_fp32_by_default(in_dtype):
    torch.manual_seed(0)
    G = torch.randn(12, 8).to(in_dtype)
    v = torch.randn(16).to(in_dtype)
    assert ns(G).dtype == torch.float32
    assert ns_vec(v).dtype == torch.float32


def test_ns_fp32_path_has_no_hidden_bf16():
    """The default (fp32) path must be genuinely fp32: against a float64
    reference, fp32 ns is orders of magnitude closer than bf16 ns."""
    torch.manual_seed(1)
    G = torch.randn(48, 32)
    ref64 = ns(G.double(), dtype=torch.float64).double()
    err32 = (ns(G).double() - ref64).abs().max().item()
    err16 = (ns(G, dtype=torch.bfloat16).double() - ref64).abs().max().item()
    assert err32 < 1e-4
    assert err16 > 1e-3
    assert err32 < err16 / 100


def test_ns_does_not_mutate_input():
    torch.manual_seed(2)
    G = torch.randn(10, 6)
    G0 = G.clone()
    ns(G)
    assert torch.equal(G, G0)
    v = torch.randn(7)
    v0 = v.clone()
    ns_vec(v)
    assert torch.equal(v, v0)


# --------------------------------------------------------------- the oracle tests
ORACLE_SHAPES = [(48, 48), (96, 32), (32, 96)]  # square, tall, wide
ORACLE_RULES = [("keller", "original"), ("moonlight", "match_rms_adamw")]
LR = 0.02
MOMENTUM = 0.95


def _two_steps(shape, rule, wd, patch_fp32, monkeypatch):
    """Run two identical steps of MuonFamily and torch.optim.Muon on the same
    weights and grads; return (ours, theirs) parameter trajectories."""
    torch.manual_seed(hash((shape, rule)) % (2**31))
    W0 = torch.randn(*shape)
    grads = [torch.randn(*shape), torch.randn(*shape)]

    if patch_fp32:
        monkeypatch.setattr(
            torch_muon_mod, "_zeropower_via_newtonschulz", _fp32_zeropower
        )

    p = torch.nn.Parameter(W0.clone())
    ours = MuonFamily([p], lr=LR, wd=wd, momentum=MOMENTUM, rule=rule)
    q = torch.nn.Parameter(W0.clone())
    adjust = dict(ORACLE_RULES)[rule]
    theirs = torch.optim.Muon(
        [q],
        lr=LR,
        weight_decay=wd,
        momentum=MOMENTUM,
        nesterov=True,
        ns_steps=5,
        eps=1e-7,
        adjust_lr_fn=adjust,
    )
    ours_traj, theirs_traj = [], []
    for g in grads:
        p.grad = g.clone()
        q.grad = g.clone()
        ours.step()
        theirs.step()
        ours_traj.append(p.detach().clone())
        theirs_traj.append(q.detach().clone())
    return ours_traj, theirs_traj


def _fp32_zeropower(grad, ns_coefficients, ns_steps, eps):
    """Verbatim torch.optim._muon._zeropower_via_newtonschulz (torch 2.15)
    with its single hard-coded bfloat16 cast changed to float32 — the only
    way to 'configure' the oracle's NS dtype to PLAN's fp32 setting, since
    torch exposes no dtype argument. Op order (addmm, clamp) is torch's own.
    """
    a, b, c = ns_coefficients
    ortho_grad = grad.to(dtype=torch.float32, copy=True)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


@pytest.mark.parametrize("wd", [0.0, 0.1], ids=["wd0", "wd0.1"])
@pytest.mark.parametrize("rule,adjust", ORACLE_RULES, ids=["keller", "moonlight"])
@pytest.mark.parametrize("shape", ORACLE_SHAPES, ids=str)
def test_muonfamily_matches_torch_muon_fp32_oracle(
    shape, rule, adjust, wd, monkeypatch
):
    """THE oracle test, tight version: two steps of MuonFamily vs
    torch.optim.Muon with its NS run in fp32 (dtype-only monkeypatch, see
    module docstring mapping 6a). Remaining differences are mappings 1 and 5
    (momentum scalar through the normalization; clamp-vs-add eps) plus op
    order, all O(1e-7) relative. Measured max |param diff| over all 12
    cases, both steps (CPU, torch 2.15): 2.4e-7. Asserted at atol=1e-5.
    """
    ours, theirs = _two_steps(shape, rule, wd, patch_fp32=True, monkeypatch=monkeypatch)
    for step, (a, b) in enumerate(zip(ours, theirs), 1):
        assert torch.allclose(a, b, atol=1e-5, rtol=0), (
            f"step {step}: max diff {(a - b).abs().max().item():.3e}"
        )


@pytest.mark.parametrize("wd", [0.0, 0.1], ids=["wd0", "wd0.1"])
@pytest.mark.parametrize("rule,adjust", ORACLE_RULES, ids=["keller", "moonlight"])
@pytest.mark.parametrize("shape", ORACLE_SHAPES, ids=str)
def test_muonfamily_matches_torch_muon_stock_bf16_oracle(
    shape, rule, adjust, wd, monkeypatch
):
    """THE oracle test against the UNPATCHED torch.optim.Muon.

    Exact match is impossible for a documented reason: torch 2.15 hard-codes
    bf16 inside its Newton-Schulz (module docstring, mapping 6). The
    justifiable tolerance is bf16 resolution propagated through the update:
    |delta_p| <= adjusted_lr * |ns error|, with |ns error| measured at
    <= 0.07 absolute on NS outputs of order ~1 (bf16 eps 2^-8 amplified over
    5 iterations). Measured max |param diff| over all 12 cases, both steps
    (CPU, torch 2.15): 4.2e-4; asserted at atol = 0.1 * adjusted_lr (2e-3
    keller square, ~3.9e-3 moonlight tall), >= 5x the measured worst case
    and ~10x below the update magnitude.
    """
    ours, theirs = _two_steps(
        shape, rule, wd, patch_fp32=False, monkeypatch=monkeypatch
    )
    adjusted_lr = LR * scale(torch.Size(shape), rule)
    for step, (a, b) in enumerate(zip(ours, theirs), 1):
        assert torch.allclose(a, b, atol=0.1 * adjusted_lr, rtol=0), (
            f"step {step}: max diff {(a - b).abs().max().item():.3e} "
            f"(bound {0.1 * adjusted_lr:.3e})"
        )


# --------------------------------------------------------------- MuonFamily contracts
def test_muonfamily_1d_params_use_ns_vec():
    """A 1-D parameter gets the elementwise sign-like update (no scale)."""
    torch.manual_seed(13)
    v0 = torch.randn(32)
    g = torch.randn(32)
    p = torch.nn.Parameter(v0.clone())
    opt = MuonFamily([p], lr=0.1, wd=0.0, momentum=0.0)
    p.grad = g.clone()
    opt.step()
    # momentum 0, zero buffer => u = ns_vec(g)
    expected = v0 - 0.1 * ns_vec(g)
    assert torch.allclose(p.detach(), expected, atol=1e-6, rtol=0)


def test_muonfamily_ns_steps_and_dtype_are_config():
    """ns_steps/ns_dtype flow through to the update (faithful reruns are
    config-only): 6-step bf16 differs from the 5-step fp32 default, and
    matches a manual 6-step bf16 ns of the same Nesterov input."""
    torch.manual_seed(17)
    W0 = torch.randn(24, 16)
    g = torch.randn(24, 16)

    def one_step(**kw):
        p = torch.nn.Parameter(W0.clone())
        opt = MuonFamily([p], lr=0.1, momentum=0.0, **kw)
        p.grad = g.clone()
        opt.step()
        return p.detach()

    default = one_step()
    faithful = one_step(ns_steps=6, ns_dtype=torch.bfloat16)
    assert not torch.allclose(default, faithful, atol=1e-5, rtol=0)
    # mirror the reference op order: u = ns(u)*scale in ns_dtype, then .to(p.dtype)
    manual = W0 - 0.1 * (
        ns(g, steps=6, dtype=torch.bfloat16) * scale(W0.shape, "keller")
    ).to(W0.dtype)
    assert torch.allclose(faithful, manual, atol=1e-6, rtol=0)


def test_state_dict_round_trip_resumes_bit_identical():
    """Budget-curve branch runs load optimizer state (design.md section 3.8
    step 4), so a save/load mid-momentum must resume exactly.

    3 steps with non-default hypers (ns_steps=6, ns_dtype=bf16, wd=0.1),
    torch.save/torch.load (weights_only=True) of params + optimizer
    state_dict into a fresh instance constructed with deliberately DIFFERENT
    hypers, then 2 more steps: bit-identical to 5 straight steps, and
    load_state_dict must restore the saved group hypers (ns_steps, ns_dtype,
    rule, wd, lr, momentum), not keep the constructor's."""
    torch.manual_seed(5)
    W0 = torch.randn(24, 16)
    v0 = torch.randn(32)
    grads = [(torch.randn(24, 16), torch.randn(32)) for _ in range(5)]
    hypers = dict(lr=0.1, wd=0.1, momentum=0.95, rule="keller",
                  ns_steps=6, ns_dtype=torch.bfloat16)

    def fresh(**kw):
        pm = torch.nn.Parameter(W0.clone())
        pv = torch.nn.Parameter(v0.clone())
        return pm, pv, MuonFamily([pm, pv], **kw)

    def run(opt, pm, pv, gs):
        for gm, gv in gs:
            pm.grad = gm.clone()
            pv.grad = gv.clone()
            opt.step()

    # reference: 5 straight steps
    pm, pv, opt = fresh(**hypers)
    run(opt, pm, pv, grads)

    # resumed: 3 steps, save, load into a differently-configured instance
    pm2, pv2, opt2 = fresh(**hypers)
    run(opt2, pm2, pv2, grads[:3])
    buf = io.BytesIO()
    torch.save(
        {"pm": pm2.detach(), "pv": pv2.detach(), "opt": opt2.state_dict()}, buf
    )
    buf.seek(0)
    ckpt = torch.load(buf, weights_only=True)
    pm3 = torch.nn.Parameter(ckpt["pm"].clone())
    pv3 = torch.nn.Parameter(ckpt["pv"].clone())
    opt3 = MuonFamily(
        [pm3, pv3], lr=0.5, wd=0.0, momentum=0.5, rule="moonlight",
        ns_steps=5, ns_dtype=torch.float32,  # deliberately different
    )
    opt3.load_state_dict(ckpt["opt"])
    g = opt3.param_groups[0]
    assert g["ns_steps"] == 6
    assert g["ns_dtype"] == torch.bfloat16
    assert g["rule"] == "keller"
    assert g["wd"] == 0.1
    assert g["lr"] == 0.1
    assert g["momentum"] == 0.95
    run(opt3, pm3, pv3, grads[3:])
    assert torch.equal(pm3.detach(), pm.detach())
    assert torch.equal(pv3.detach(), pv.detach())


def test_nesterov_update_hook():
    """nesterov_update (the ManifoldMuon integration hook): returns the same
    u as the manual buf/u computation, mutates the buffer exactly once per
    call, and raises for a parameter the optimizer does not manage."""
    torch.manual_seed(29)
    p = torch.nn.Parameter(torch.randn(12, 8))
    opt = MuonFamily([p], lr=0.1, momentum=0.9)

    g1 = torch.randn(12, 8)
    p.grad = g1.clone()
    u1 = opt.nesterov_update(p)
    # zero buffer: buf <- 0.9*0 + g1 = g1 (exactly), u = g1 + 0.9*buf
    assert torch.equal(opt.state[p]["momentum_buffer"], g1)
    assert torch.equal(u1, g1 + 0.9 * g1)

    g2 = torch.randn(12, 8)
    p.grad = g2.clone()
    u2 = opt.nesterov_update(p)
    buf2 = 0.9 * g1 + g2  # mutated exactly once more, not twice
    assert torch.equal(opt.state[p]["momentum_buffer"], buf2)
    assert torch.equal(u2, g2 + 0.9 * buf2)

    unmanaged = torch.nn.Parameter(torch.randn(12, 8))
    unmanaged.grad = torch.randn(12, 8)
    with pytest.raises(KeyError):
        opt.nesterov_update(unmanaged)


def test_muonfamily_skips_gradless_params():
    p1 = torch.nn.Parameter(torch.randn(4, 4))
    p2 = torch.nn.Parameter(torch.randn(4, 4))
    before = p2.detach().clone()
    opt = MuonFamily([p1, p2], lr=0.1)
    p1.grad = torch.randn(4, 4)
    opt.step()
    assert torch.equal(p2.detach(), before)


# --------------------------------------------------------------- msign_calls
def _manifold_step(k=3, dual_cap=5, tol=0.0, shape=(24, 12)):
    torch.manual_seed(23)
    params = []
    for _ in range(k):
        p = torch.nn.Parameter(torch.randn(*shape))
        p.grad = 0.1 * torch.randn(*shape)
        params.append(p)
    opt = ManifoldMuon([p for p in params], lr=0.1, dual_cap=dual_cap, tol=tol,
                       momentum=0.0)
    return opt.step()


def test_msign_calls_at_cap():
    """cap=5, tol=0 on k matrices in ONE shape group: the ascent loop runs
    all 5 iterations (one batched msign call each, the residual check never
    passes at tol=0) and the retraction adds one more, so every matrix in
    the group reports cap + 1 = 6 msign calls. Counted at the call sites
    (Gate-1 decision), and here it happens to equal dual_steps + 1."""
    stats = _manifold_step(k=3, dual_cap=5, tol=0.0)
    assert stats["dual_steps"] == [5, 5, 5]
    assert stats["msign_calls"] == [6, 6, 6]
    stats = _manifold_step(k=2, dual_cap=3, tol=0.0)
    assert stats["msign_calls"] == [4, 4]


def test_msign_calls_counts_the_early_stop_call():
    """tol=inf: the FIRST ascent msign call already reveals convergence
    (dual_steps = 0, no Lambda update applied), but that call was made and
    must be counted — plus the retraction: 2 calls, NOT the 1 that deriving
    from dual_steps (0 + retraction) would give."""
    stats = _manifold_step(k=3, dual_cap=5, tol=float("inf"))
    assert stats["dual_steps"] == [0, 0, 0]
    assert stats["msign_calls"] == [2, 2, 2]


def test_msign_calls_cap_zero_counts_fallback():
    """dual_cap=0: no ascent loop; the raw-direction msign(G) fallback and
    the retraction are both real calls."""
    stats = _manifold_step(k=1, dual_cap=0, tol=0.0)
    assert stats["dual_steps"] == [0]
    assert stats["msign_calls"] == [2]


def test_msign_calls_keeps_existing_keys_and_alignment():
    stats = _manifold_step(k=2, dual_cap=5, tol=1e-6)
    assert set(stats) >= {"dual_steps", "dual_residual", "msign_calls"}
    assert len(stats["msign_calls"]) == len(stats["dual_steps"]) == 2
    assert len(stats["dual_residual"]) == 2
    # within one shape group every matrix participates in every batched call
    assert stats["msign_calls"][0] == stats["msign_calls"][1]
    assert stats["msign_calls"][0] >= max(stats["dual_steps"]) + 1
