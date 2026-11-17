"""Manifold Muon (Stiefel) optimizer.

Algorithm per the vendored third_party/manifolds/src/manifold_muon.py
(Bernstein 2025, "Modular manifolds"), with the changes fixed in PLAN.md
(Decisions log: "Manifold Muon settings"):

- the raw gradient is replaced by the same Nesterov momentum buffer as
  MuonFamily (buf = momentum*buf + grad; u = grad + momentum*buf); with
  momentum=0 and a zero buffer this reduces to u = grad, the reference
  behavior;
- dual ascent is capped at ``dual_cap`` with early stop at ``tol``; the dual
  step is dual_lr * (1 - step/dual_cap), i.e. the reference's linear decay
  with the cap substituted for its ``steps`` argument;
- same-shape parameters within a param group are stacked and processed in one
  batched msign call (dual ascent included), with per-matrix early-stop
  tracking: once a matrix's residual ||H||/sqrt(numel(H)) < tol its Lambda is
  frozen; the loop stops when all matrices are converged or the cap is hit;
- step() returns per-matrix dual-ascent statistics for logging, including
  "msign_calls": the msign invocations actually made, counted at the call
  sites (Gate-1 decision) — ascent calls incl. the early-stop check call,
  plus the retraction — for FLOP accounting;
- no weight decay anywhere.

Self-contained: no imports from third_party/ (the shared Nesterov code path
with MuonFamily arrives in Phase 2).
"""

import math
from typing import Iterable

import torch

from muonarch.msign import msign

__all__ = ["ManifoldMuon", "grouped_by_shape"]


def grouped_by_shape(params) -> dict[torch.Size, list]:
    """Group parameters by shape, preserving encounter order.

    Same-shape parameters (e.g. the MoE's 16 experts per projection) are later
    stacked into one (k, m, n) tensor so dual ascent and retraction run as
    single batched msign calls.
    """
    groups: dict[torch.Size, list] = {}
    for p in params:
        groups.setdefault(p.shape, []).append(p)
    return groups


@torch.no_grad()
def _manifold_update(
    W: torch.Tensor,
    G: torch.Tensor,
    lr: float,
    dual_cap: int,
    tol: float,
    dual_lr: float,
    msign_steps: int,
    msign_dtype: torch.dtype,
) -> tuple[torch.Tensor, list[int], list[float], int]:
    """One batched Manifold Muon update for stacked matrices W, G of shape
    (k, m, n). Returns (new_W, dual_steps_per_matrix,
    final_residual_per_matrix, msign_calls).

    ``msign_calls`` counts the msign invocations actually made at the call
    sites (Gate-1 decision; never derived from dual_steps): one per
    dual-ascent loop iteration entered — including the final call when early
    stop fires — plus the cap==0 fallback call if taken, plus the retraction
    call. Each batched call covers every matrix in the stack (converged
    matrices only have their Lambda frozen; they are still in the msign
    input), so the count applies to each matrix in the group.
    """
    # Ensure the matrices are tall, as in the reference.
    should_transpose = W.size(-2) < W.size(-1)
    if should_transpose:
        W = W.mT
        G = G.mT
    ascent = _dual_ascent_vectorized if _VECTORIZED_DUAL_ASCENT else _dual_ascent_loop
    A, _Lambda, steps_taken, residuals, msign_calls = ascent(
        W, G, dual_cap, tol, dual_lr, msign_steps, msign_dtype)
    # Descend on the primal problem, then retract to the manifold.
    new_W = W - lr * A
    new_W = msign(new_W, steps=msign_steps, dtype=msign_dtype)
    msign_calls += 1
    if should_transpose:
        new_W = new_W.mT
    return new_W, steps_taken, residuals, msign_calls


# Internal switch (NOT a config field, not hashed): True = the vectorized dual
# ascent (speed-only change, Gate 3); False = the original per-matrix Python
# loop, kept as the test oracle. Both give the same A, Lambda, per-matrix
# dual-step counts, residuals and msign call counts.
_VECTORIZED_DUAL_ASCENT = True


@torch.no_grad()
def _dual_ascent_loop(W, G, dual_cap, tol, dual_lr, msign_steps, msign_dtype):
    """Original dual ascent on tall stacks W, G (k, m, n): per-matrix Python
    bookkeeping with one ``.item()`` sync per matrix per iteration.
    Returns (A, Lambda, steps_taken, residuals, msign_calls) — msign_calls
    excludes the retraction."""
    k = W.size(0)
    # Initialize the dual variable.
    Lambda = -0.25 * (W.mT @ G + G.mT @ W)
    denom = math.sqrt(Lambda.size(-1) * Lambda.size(-2))  # sqrt(H.numel()) per matrix
    converged = [False] * k
    steps_taken = [0] * k
    residuals = [float("inf")] * k
    A = None
    msign_calls = 0  # counted at the call sites, never derived from dual_steps
    # Ascend on the dual problem to find the update direction A.
    for step in range(dual_cap):
        A = msign(G + 2 * W @ Lambda, steps=msign_steps, dtype=msign_dtype)
        msign_calls += 1
        # Deviation of A from the tangent space, per matrix.
        H = W.mT @ A + A.mT @ W
        res = H.norm(dim=(-2, -1)) / denom  # (k,)
        for i in range(k):
            if not converged[i]:
                residuals[i] = res[i].item()
                if residuals[i] < tol:
                    converged[i] = True
        if all(converged):
            break
        # Update the dual variable, freezing converged matrices.
        scale = dual_lr * (1 - step / dual_cap)
        for i in range(k):
            if not converged[i]:
                Lambda[i] -= scale * H[i]
                steps_taken[i] += 1
    if A is None:  # dual_cap == 0: no ascent, use the raw direction
        A = msign(G, steps=msign_steps, dtype=msign_dtype)
        msign_calls += 1
    return A, Lambda, steps_taken, residuals, msign_calls


@torch.no_grad()
def _dual_ascent_vectorized(W, G, dual_cap, tol, dual_lr, msign_steps, msign_dtype):
    """Same algorithm as _dual_ascent_loop with the per-matrix bookkeeping as
    masked tensor ops: an ``active`` mask per matrix freezes a converged
    matrix's residual, Lambda and dual-step count exactly where the loop
    would. One host sync per iteration (the all-converged check) instead of
    one per matrix. The msign calls themselves are unchanged: every
    iteration's batched call covers the whole stack, as before."""
    k = W.size(0)
    Lambda = -0.25 * (W.mT @ G + G.mT @ W)
    denom = math.sqrt(Lambda.size(-1) * Lambda.size(-2))
    active = torch.ones(k, dtype=torch.bool, device=W.device)
    steps_t = torch.zeros(k, dtype=torch.int64, device=W.device)
    residuals_t = torch.full((k,), float("inf"), dtype=Lambda.dtype, device=W.device)
    A = None
    msign_calls = 0  # counted at the call sites, never derived from dual_steps
    for step in range(dual_cap):
        A = msign(G + 2 * W @ Lambda, steps=msign_steps, dtype=msign_dtype)
        msign_calls += 1
        H = W.mT @ A + A.mT @ W
        res = H.norm(dim=(-2, -1)) / denom  # (k,)
        # active matrices record this residual; those below tol converge now
        residuals_t = torch.where(active, res, residuals_t)
        active = active & ~(res < tol)
        if not bool(active.any()):  # the one host sync per iteration
            break
        scale = dual_lr * (1 - step / dual_cap)
        # torch.where (not a 0/1 multiply) so frozen Lambdas are untouched
        # bit-for-bit even if H holds non-finite values.
        Lambda = torch.where(active[:, None, None], Lambda - scale * H, Lambda)
        steps_t += active.to(torch.int64)
    if A is None:  # dual_cap == 0: no ascent, use the raw direction
        A = msign(G, steps=msign_steps, dtype=msign_dtype)
        msign_calls += 1
    return A, Lambda, steps_t.tolist(), residuals_t.tolist(), msign_calls


class ManifoldMuon(torch.optim.Optimizer):
    """Muon constrained to the Stiefel manifold; 2-D parameters only.

    No weight decay anywhere (constrained matrices have none by design).
    """

    def __init__(
        self,
        params,
        lr: float,
        dual_cap: int = 5,
        tol: float = 1e-6,
        dual_lr: float = 0.01,
        momentum: float = 0.95,
        msign_steps: int = 10,
        msign_dtype: torch.dtype = torch.float32,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if dual_cap < 0:
            raise ValueError(f"Invalid dual_cap: {dual_cap}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(
            lr=lr,
            dual_cap=dual_cap,
            tol=tol,
            dual_lr=dual_lr,
            momentum=momentum,
            msign_steps=msign_steps,
            msign_dtype=msign_dtype,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> dict:
        """One optimization step.

        Returns {"dual_steps": [...], "dual_residual": [...],
        "msign_calls": [...]}: per matrix, the number of dual-ascent updates
        actually applied to its Lambda, the final residual
        ||H||/sqrt(numel(H)), and the number of msign invocations actually
        made for its shape group this step — counted at the call sites
        (Gate-1 decision), never derived from dual_steps: every ascent-loop
        iteration's call (including the final call when early stop fires,
        and the cap==0 fallback) plus the retraction call. Batched calls
        cover every matrix in the group, so matrices in one group share the
        same count. Order: shape groups in first-seen order, matrices within
        a group in encounter order (grouped_by_shape may reorder when shapes
        interleave; no index->parameter map is exposed).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dual_steps: list[int] = []
        dual_residual: list[float] = []
        msign_calls: list[int] = []
        for group in self.param_groups:
            momentum = group["momentum"]
            with_grad = [p for p in group["params"] if p.grad is not None]
            for shape, plist in grouped_by_shape(with_grad).items():
                if len(shape) != 2:
                    raise ValueError(
                        f"ManifoldMuon only handles 2-D parameters, got shape {tuple(shape)}"
                    )
                # Nesterov momentum buffer, exactly as MuonFamily (PLAN.md).
                updates = []
                for p in plist:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(p.grad)
                    updates.append(p.grad + momentum * buf)
                W = torch.stack([p.detach().float() for p in plist])
                G = torch.stack([u.float() for u in updates])
                new_W, steps_taken, residuals, group_calls = _manifold_update(
                    W,
                    G,
                    lr=group["lr"],
                    dual_cap=group["dual_cap"],
                    tol=group["tol"],
                    dual_lr=group["dual_lr"],
                    msign_steps=group["msign_steps"],
                    msign_dtype=group["msign_dtype"],
                )
                for p, w in zip(plist, new_W):
                    p.copy_(w)
                dual_steps.extend(steps_taken)
                dual_residual.extend(residuals)
                msign_calls.extend([group_calls] * len(plist))

        result = {
            "dual_steps": dual_steps,
            "dual_residual": dual_residual,
            "msign_calls": msign_calls,
        }
        if loss is not None:
            result["loss"] = loss
        return result

    @staticmethod
    @torch.no_grad()
    def project_(params: Iterable[torch.nn.Parameter]) -> None:
        """Stiefel projection at init: W <- msign(W), in fp32.

        msign handles wide matrices by its own transpose-to-wide rule, so the
        result has all singular values ~1 for any 2-D orientation.
        """
        for p in params:
            if p.ndim != 2:
                raise ValueError(
                    f"project_ only handles 2-D parameters, got shape {tuple(p.shape)}"
                )
            p.copy_(msign(p.detach().float(), dtype=torch.float32))
