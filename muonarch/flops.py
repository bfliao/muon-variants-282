"""Analytical FLOP counts (model + optimizer) and FlopCounterMode verification.

Implements design.md section 3.7 and PLAN.md's Protocol FLOP-speedup formula

    F = 6 * N_active * D + F_attn + S * sum_{W in Muon} k * (4 m^2 n + 2 m^3),
    m = min(rows, cols), n = max(rows, cols),

decoupled from the not-yet-written model/optimizer modules: every function
takes plain shape lists or explicit numbers, never ParamGroups.

Conventions (aligned with ``torch.utils.flop_counter.FlopCounterMode``, which
counts a matmul of (a, b) @ (b, c) as 2*a*b*c FLOPs and counts NO elementwise
ops):

- MLP (3072 -> w -> w -> w -> 10, biases, ReLU): forward matmul is 2*m*n per
  sample per Linear, backward is 2x forward (grad-input and grad-weight
  matmuls), i.e. the 6*m*n*batch convention -- for EVERY layer, including the
  first. In a real training step autograd skips the first layer's grad-input
  matmul (the data needs no grad), but PLAN's quoted 1.8-7.9x NS ratios are
  computed under the uniform 6*m*n convention, so this module keeps it;
  ``verify_against_counter`` enables grad on the input so FlopCounterMode
  measures the same set of matmuls. Bias adds and ReLU are DROPPED: they are
  O(batch * width) elementwise terms, negligible, and FlopCounterMode does not
  count them either.
- LM: ``6 * N_active * D`` with the tied embedding counted ONCE (the lookup is
  0 FLOPs; the tied lm_head matmul is exactly the tensor's 6*numel*D), plus an
  explicit attention term

      F_attn = 12 * n_layers * n_q_heads * head_dim * seq_len * tokens.

  Derivation (eager-attention matmul convention, Qwen3 geometry: 6 layers,
  2 query heads / 1 KV head repeated to 2 via repeat_kv, head_dim 64, seq
  256): per layer and sequence each query head does QK^T (2*T^2*d) and
  A @ V (2*T^2*d) forward = 4*T^2*d; backward is 2x forward; with
  tokens = n_seqs * T this sums to 12 * L * h_q * d * T * tokens. The causal
  mask is not exploited (full T^2), matching FlopCounterMode. Norm/RoPE
  parameters inside N_active contribute no matmuls (~0.1% overcount).
- Manifold Muon optimizer FLOPs use ``msign_calls`` -- the per-matrix mean
  number of msign invocations per step that the OPTIMIZER ITSELF reported
  (dual-ascent calls including the early-stop check call, plus the
  retraction; logged as ``msign_calls_mean``). It is a REQUIRED argument for
  "manifold" and is never derived from the logged dual steps (Gate 1,
  DEVIATIONS.md #7).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

__all__ = [
    "ns_matrix_flops",
    "mlp_flops_per_step",
    "attn_flops",
    "lm_flops_per_step",
    "optimizer_flops_per_step",
    "lionmuon_optimizer_flops",
    "break_even_ratio",
    "verify_against_counter",
]

MLP_IN, MLP_OUT = 3072, 10

#: Documented, negligible per-parameter overhead (AdamW moments, momentum,
#: ns_vec's elementwise polynomial): ~10 FLOPs per parameter per step.
_FLOPS_PER_PARAM = 10

_NS_FAMILY = ("muon", "moonlight", "muonall")


def ns_matrix_flops(shape: Sequence[int], iters: int) -> int:
    """k * (4*m^2*n + 2*m^3), m = min(dims), n = max(dims) (PLAN Protocol)."""
    m, n = min(shape), max(shape)
    return iters * (4 * m * m * n + 2 * m**3)


def mlp_flops_per_step(widths: Sequence[int], batch: int) -> int:
    """fwd + bwd FLOPs of one step of the Linear stack 3072 -> w1 -> w2 -> w3
    -> 10 (the 6*m*n*batch convention; bias/ReLU terms dropped, see module
    docstring)."""
    dims = [MLP_IN, *widths, MLP_OUT]
    return 6 * batch * sum(a * b for a, b in zip(dims, dims[1:]))


def attn_flops(
    tokens: int,
    seq_len: int = 256,
    n_layers: int = 6,
    n_q_heads: int = 2,
    head_dim: int = 64,
) -> int:
    """F_attn = 12 * L * h_q * d * T * tokens (fwd + bwd, eager matmul
    convention, full T^2; see module docstring for the derivation)."""
    return 12 * n_layers * n_q_heads * head_dim * seq_len * tokens


def lm_flops_per_step(
    n_active_params: int,
    tokens_per_step: int,
    seq_len: int = 256,
    n_layers: int = 6,
    n_q_heads: int = 2,
    head_dim: int = 64,
    f_attn: int | None = None,
) -> int:
    """F = 6 * N_active * D + F_attn (PLAN formula). ``n_active_params``
    counts the tied embedding once; MoE uses active (top-4) parameters.
    ``f_attn`` overrides the analytic attention term when given explicitly."""
    if f_attn is None:
        f_attn = attn_flops(tokens_per_step, seq_len, n_layers, n_q_heads, head_dim)
    return 6 * n_active_params * tokens_per_step + f_attn


def optimizer_flops_per_step(
    muon_shapes: Iterable[Sequence[int]],
    vec_sizes: Iterable[int],
    optimizer: str,
    ns_steps: int = 5,
    msign_steps: int = 10,
    msign_calls: float | None = None,
    muon_step: bool | None = None,
) -> int:
    """Optimizer FLOPs for one step.

    - "muon"/"moonlight"/"muonall": sum of ns_matrix_flops(shape, ns_steps)
      over the Muon-group matrices, plus ~10 FLOPs/param (AdamW moments,
      momentum, ns_vec's O(d) elementwise polynomial) -- included, negligible.
    - "adamw": the ~10 FLOPs/param term only.
    - "manifold": msign_calls * sum of ns_matrix_flops(shape, msign_steps) --
      the same 4m^2n + 2m^3 per iteration, msign_steps iterations per call --
      where ``msign_calls`` is the per-matrix mean count of msign invocations
      the optimizer reported (REQUIRED; DEVIATIONS.md #7), plus the
      ~10 FLOPs/param term.
    - "lionmuon": ``muon_step`` is REQUIRED (which kind of step this is).
      A Muon step (True) costs exactly what Moonlight's step costs on the
      same shapes — NS on every matrix plus the ~10 FLOPs/param term; a sign
      step (False) costs the ~10 FLOPs/param term only (EMA, sign and the
      update are elementwise). Never averaged over the period here; a run
      total is ``lionmuon_optimizer_flops``.
    """
    muon_shapes = list(muon_shapes)
    n_params = sum(int(m) * int(n) for m, n in muon_shapes) + sum(vec_sizes)
    overhead = _FLOPS_PER_PARAM * n_params
    if optimizer in _NS_FAMILY:
        return sum(ns_matrix_flops(s, ns_steps) for s in muon_shapes) + overhead
    if optimizer == "manifold":
        if msign_calls is None:
            raise ValueError(
                "manifold FLOPs require msign_calls, the optimizer's own "
                "reported per-matrix mean msign-invocation count "
                "(DEVIATIONS.md #7); it is never derived from dual steps"
            )
        per_call = sum(ns_matrix_flops(s, msign_steps) for s in muon_shapes)
        return round(msign_calls * per_call) + overhead
    if optimizer == "adamw":
        return overhead
    if optimizer == "lionmuon":
        if muon_step is None:
            raise ValueError(
                "lionmuon FLOPs require muon_step (True: Newton-Schulz step, "
                "False: sign step); the two step types are never averaged"
            )
        if muon_step:
            return sum(ns_matrix_flops(s, ns_steps) for s in muon_shapes) + overhead
        return overhead
    raise ValueError(f"unknown optimizer {optimizer!r}")


def lionmuon_optimizer_flops(
    muon_shapes: Iterable[Sequence[int]],
    vec_sizes: Iterable[int],
    n_steps: int,
    period: int,
    ns_steps: int = 5,
) -> int:
    """Exact Lion-Muon optimizer FLOPs of a run of ``n_steps`` steps started
    at step 0: the Muon steps are t = 0, period, 2*period, ... < n_steps,
    i.e. ceil(n_steps / period) of them, each at the Muon-step cost; every
    other step at the sign-step cost."""
    muon_shapes, vec_sizes = list(muon_shapes), list(vec_sizes)
    n_muon = -(-n_steps // period)
    muon_fl = optimizer_flops_per_step(
        muon_shapes, vec_sizes, "lionmuon", ns_steps=ns_steps, muon_step=True)
    sign_fl = optimizer_flops_per_step(
        muon_shapes, vec_sizes, "lionmuon", muon_step=False)
    return n_muon * muon_fl + (n_steps - n_muon) * sign_fl


def break_even_ratio(model_flops_per_step: float, optimizer_flops_per_step: float) -> float:
    """Token speedup needed for FLOP break-even against a free-optimizer
    baseline (H2a): total/model FLOPs = (model + optimizer) / model."""
    return (model_flops_per_step + optimizer_flops_per_step) / model_flops_per_step


def _analytic_model_flops(model: torch.nn.Module, batch: torch.Tensor) -> int:
    """Analytic fwd+bwd FLOPs for the two project architectures."""
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        cfg = model.config
        # parameters() deduplicates shared tensors: tied embedding counted once
        n_active = sum(p.numel() for p in model.parameters())
        return lm_flops_per_step(
            n_active,
            tokens_per_step=batch.numel(),
            seq_len=batch.shape[-1],
            n_layers=cfg.num_hidden_layers,
            n_q_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim,
        )
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    if not linears:
        raise ValueError("model is neither a causal LM nor a Linear stack")
    dims = [linears[0].in_features] + [l.out_features for l in linears]
    for prev, l in zip(linears, linears[1:]):
        if l.in_features != prev.out_features:
            raise ValueError("Linear stack dims do not chain")
    return 6 * batch.shape[0] * sum(a * b for a, b in zip(dims, dims[1:]))


def verify_against_counter(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    batch: torch.Tensor,
) -> tuple[int, int]:
    """(analytic, counted): analytic fwd+bwd FLOPs vs FlopCounterMode measured
    around forward + backward (+ optimizer.step() where countable -- AdamW's
    elementwise update registers 0 counted FLOPs, and its analytic ~10/param
    term is <0.1% here, so the analytic side reports the model term).

    FlopCounterMode counts matmuls as 2*m*n*k and no elementwise ops; the
    analytic conventions in this module are aligned with that (see module
    docstring), so the test tolerance of 5% is slack, not calibration. For a
    float-input model the input is given requires_grad so the counter also
    runs the first layer's grad-input matmul, which the uniform 6*m*n
    convention counts (see module docstring).
    """
    from torch.utils.flop_counter import FlopCounterMode

    model.train()
    model.zero_grad(set_to_none=True)
    counter = FlopCounterMode(display=False)
    with counter:
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            loss = model(input_ids=batch, labels=batch).loss
        else:
            x = batch.detach().clone().requires_grad_(True)
            loss = model(x).float().pow(2).mean()
        loss.backward()
        if optimizer is not None:
            optimizer.step()
    counted = counter.get_total_flops()
    analytic = _analytic_model_flops(model, batch)
    return analytic, counted
