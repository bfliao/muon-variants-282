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
- step() returns per-matrix dual-ascent statistics for logging;
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
) -> tuple[torch.Tensor, list[int], list[float]]:
    """One batched Manifold Muon update for stacked matrices W, G of shape
    (k, m, n). Returns (new_W, dual_steps_per_matrix, final_residual_per_matrix).
    """
    # Ensure the matrices are tall, as in the reference.
    should_transpose = W.size(-2) < W.size(-1)
    if should_transpose:
        W = W.mT
        G = G.mT
    k = W.size(0)
    # Initialize the dual variable.
    Lambda = -0.25 * (W.mT @ G + G.mT @ W)
    denom = math.sqrt(Lambda.size(-1) * Lambda.size(-2))  # sqrt(H.numel()) per matrix
    converged = [False] * k
    steps_taken = [0] * k
    residuals = [float("inf")] * k
    A = None
    # Ascend on the dual problem to find the update direction A.
    for step in range(dual_cap):
        A = msign(G + 2 * W @ Lambda, steps=msign_steps, dtype=msign_dtype)
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
    # Descend on the primal problem, then retract to the manifold.
    new_W = W - lr * A
    new_W = msign(new_W, steps=msign_steps, dtype=msign_dtype)
    if should_transpose:
        new_W = new_W.mT
    return new_W, steps_taken, residuals


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

        Returns {"dual_steps": [...], "dual_residual": [...]}: per matrix, the
        number of dual-ascent updates actually applied to its Lambda and the
        final residual ||H||/sqrt(numel(H)). Order: shape groups in first-seen
        order, matrices within a group in encounter order (grouped_by_shape may
        reorder when shapes interleave; no index->parameter map is exposed).
        Ascent msign calls per group = max(steps)+1 when early stop fired,
        = cap otherwise (relevant to FLOP accounting).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dual_steps: list[int] = []
        dual_residual: list[float] = []
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
                new_W, steps_taken, residuals = _manifold_update(
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

        result = {"dual_steps": dual_steps, "dual_residual": dual_residual}
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
