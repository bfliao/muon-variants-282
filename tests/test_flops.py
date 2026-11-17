"""Tests for muonarch/flops.py (design.md section 3.7; PLAN.md Protocol
FLOP-speedup formula; DEVIATIONS.md #7).

Conventions pinned here:
- ns_matrix_flops(shape, k) = k * (4*m^2*n + 2*m^3), m = min(dims),
  n = max(dims)  (PLAN Protocol formula).
- MLP fwd+bwd = 6 * batch * sum(in * out) over the Linear stack (2*m*n fwd
  matmul per sample, backward 2x forward); bias adds and ReLU are dropped --
  they are elementwise, negligible, and FlopCounterMode does not count them
  either, so the conventions align.
- LM: F = 6 * N_active * D + F_attn (PLAN formula), with the tied embedding
  counted ONCE in N_active, and F_attn = 12 * n_layers * n_q_heads * head_dim
  * seq_len * tokens (eager-attention matmul convention: QK^T and A@V are each
  2*T^2*d per head fwd, backward 2x forward; GQA KV heads are repeated to the
  n_q_heads query heads before the matmuls, so n_q_heads is the multiplier;
  the causal mask is not exploited -- full T^2, matching FlopCounterMode's
  2*m*n*k matmul counting).
- Manifold optimizer FLOPs use msign_calls -- the per-matrix mean number of
  msign invocations the optimizer itself reported (DEVIATIONS.md #7), a
  REQUIRED argument, never derived from dual steps.
"""

import pytest
import torch
import torch.nn as nn

from muonarch.flops import (
    attn_flops,
    break_even_ratio,
    lm_flops_per_step,
    mlp_flops_per_step,
    ns_matrix_flops,
    optimizer_flops_per_step,
    verify_against_counter,
)

# ---------------------------------------------------------------- NS formula


def test_ns_flops_formula():
    """k * (4*m^2*n + 2*m^3), m = min(dims), n = max(dims)."""
    assert ns_matrix_flops((128, 3072), 5) == 5 * (4 * 128**2 * 3072 + 2 * 128**3)
    assert ns_matrix_flops((128, 3072), 5) == 1_027_604_480
    # symmetric in orientation
    assert ns_matrix_flops((3072, 128), 5) == ns_matrix_flops((128, 3072), 5)
    # square: 4*m^3 + 2*m^3 = 6*m^3 per iteration
    assert ns_matrix_flops((256, 256), 1) == 6 * 256**3
    # linear in iteration count
    assert ns_matrix_flops((64, 32), 10) == 10 * ns_matrix_flops((64, 32), 1)
    # torch.Size accepted
    assert ns_matrix_flops(torch.Size([128, 128]), 5) == 5 * 6 * 128**3


# ----------------------------------------------------------------- model FLOPs


def test_mlp_flops_per_step_value():
    dims = [3072, 128, 128, 128, 10]
    expected = 6 * 256 * sum(a * b for a, b in zip(dims, dims[1:]))
    assert mlp_flops_per_step((128, 128, 128), 256) == expected
    assert expected == 656_277_504


def test_lm_flops_per_step_value():
    """F = 6*N_active*D + F_attn with the documented Qwen3 geometry."""
    n_active, tokens = 1_700_000, 8192
    fa = attn_flops(tokens, seq_len=256, n_layers=6, n_q_heads=2, head_dim=64)
    assert fa == 12 * 6 * 2 * 64 * 256 * tokens
    assert lm_flops_per_step(n_active, tokens) == 6 * n_active * tokens + fa
    # explicit F_attn overrides the analytic term
    assert lm_flops_per_step(n_active, tokens, f_attn=123) == 6 * n_active * tokens + 123


# ------------------------------------------------------------- optimizer FLOPs

MLP_MUON_SHAPES = {
    128: [(128, 3072), (128, 128), (128, 128)],
    256: [(256, 3072), (256, 256), (256, 256)],
    512: [(512, 3072), (512, 512), (512, 512)],
}
# AdamW-group sizes for the MLP (biases + classifier) at width w
MLP_VEC_SIZES = lambda w: [w, w, w, 10, 10 * w]  # noqa: E731


def test_optimizer_flops_ns_family():
    shapes = MLP_MUON_SHAPES[128]
    vecs = MLP_VEC_SIZES(128)
    ns_sum = sum(ns_matrix_flops(s, 5) for s in shapes)
    got = optimizer_flops_per_step(shapes, vecs, "muon", ns_steps=5)
    assert got >= ns_sum  # NS term plus the ~10 flops/param overhead
    overhead = got - ns_sum
    n_params = sum(m * n for m, n in shapes) + sum(vecs)
    assert overhead == 10 * n_params  # documented, negligible
    assert overhead < 0.01 * ns_sum
    # the three NS-family variants share the count
    for name in ("moonlight", "muonall"):
        assert optimizer_flops_per_step(shapes, vecs, name, ns_steps=5) == got


def test_manifold_requires_and_uses_msign_calls():
    """Manifold FLOPs = msign_calls * sum ns_matrix_flops(shape, msign_steps);
    msign_calls is the optimizer's own reported count and is REQUIRED."""
    shapes = MLP_MUON_SHAPES[128]
    with pytest.raises((TypeError, ValueError)):
        optimizer_flops_per_step(shapes, [], "manifold")
    per_call = sum(ns_matrix_flops(s, 10) for s in shapes)
    f7 = optimizer_flops_per_step(shapes, [], "manifold", msign_calls=7)
    f8 = optimizer_flops_per_step(shapes, [], "manifold", msign_calls=8)
    assert f8 - f7 == per_call  # linear in the reported call count
    assert f7 == 7 * per_call + 10 * sum(m * n for m, n in shapes)
    # fractional means (logged msign_calls_mean) are accepted
    f_half = optimizer_flops_per_step(shapes, [], "manifold", msign_calls=7.5)
    assert f7 < f_half < f8


def test_unknown_optimizer_rejected():
    with pytest.raises(ValueError):
        optimizer_flops_per_step([(8, 8)], [], "sgd")


# --------------------------------------------------------------- break-even


def test_break_even_ratio_values():
    """PLAN: NS costs ~1.8x fwd+bwd at width 128 and ~7.9x at 512 (batch 256);
    break-even token speedup = (model + optimizer) / model, >= ~4.7 for
    width >= 256 (H2a). Reproduced to loose tolerance (+-30%)."""
    ratios = {}
    for w in (128, 256, 512):
        model = mlp_flops_per_step((w, w, w), 256)
        opt = optimizer_flops_per_step(
            MLP_MUON_SHAPES[w], MLP_VEC_SIZES(w), "muon", ns_steps=5
        )
        ratios[w] = opt / model
        assert break_even_ratio(model, opt) == pytest.approx(1 + opt / model)
    assert ratios[128] == pytest.approx(1.8, rel=0.30)
    assert ratios[512] == pytest.approx(7.9, rel=0.30)
    # H2a: break-even >= ~4.7 for width >= 256
    assert 1 + ratios[256] == pytest.approx(4.7, rel=0.30)
    assert 1 + ratios[256] > 4.0 and 1 + ratios[512] > 4.0
    # monotone in width
    assert ratios[128] < ratios[256] < ratios[512]


def test_break_even_trivial_for_free_optimizer():
    assert break_even_ratio(1000, 0) == 1.0


# ------------------------------------------------- FlopCounterMode verification


def _real_mlp():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(3072, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )


def test_analytic_vs_flopcountermode_within_5pct_mlp():
    """verify_against_counter gives the MLP input requires_grad so the counter
    runs the first layer's grad-input matmul too: the uniform 6*m*n*batch
    convention (which PLAN's 1.8-7.9x NS ratios assume) counts every layer's
    backward as 2x forward, first layer included."""
    model = _real_mlp()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = torch.randn(256, 3072)
    analytic, counted = verify_against_counter(model, opt, batch)
    assert counted > 0
    assert abs(analytic / counted - 1) <= 0.05, (analytic, counted)
    # and the analytic path is the documented mlp formula
    assert analytic == mlp_flops_per_step((128, 128, 128), 256)


def test_analytic_vs_flopcountermode_within_5pct_lm():
    """Dense Qwen3 from PLAN.md's exact config snippet; batch 4x256 to keep
    the test fast (tokens scaled accordingly)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    common = dict(
        vocab_size=4096, hidden_size=128, num_hidden_layers=6,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        max_position_embeddings=256, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(
        Qwen3Config(intermediate_size=384, attn_implementation="eager", **common)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = torch.randint(0, 4096, (4, 256))
    analytic, counted = verify_against_counter(model, opt, batch)
    assert counted > 0
    assert abs(analytic / counted - 1) <= 0.05, (analytic, counted)
    # analytic path is the documented 6*N*D + F_attn formula with the tied
    # embedding counted once
    n_active = sum(p.numel() for p in model.parameters())
    assert analytic == lm_flops_per_step(
        n_active, 4 * 256, seq_len=256, n_layers=6, n_q_heads=2, head_dim=64
    )
