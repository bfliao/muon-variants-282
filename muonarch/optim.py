"""The Muon family: one shared Newton-Schulz code path.

Muon (rule="keller"), Moonlight-Muon (rule="moonlight") and MuonAll (same
optimizer, fed every parameter incl. 1-D) share the single ``ns`` routine, so
the variants differ exactly in their stated idea (PLAN.md TL;DR 1). The
implementation is PLAN.md's reference ``MuonFamily`` verbatim in semantics,
parameterized by ``ns_steps``/``ns_dtype`` only so the faithful-settings
reruns (MuonAll: 6 steps, bf16) are a config change, not a code change
(design.md section 3.1).

Verified one-step/two-step equivalent to ``torch.optim.Muon``
(adjust_lr_fn="original"/"match_rms_adamw") in tests/test_optim.py; torch is
the oracle, never a dependency of this module.

split_params / build_optimizers / ParamGroups are appended in a later phase.
"""

import math

import torch

__all__ = ["NS_COEFFS", "ns", "ns_vec", "scale", "MuonFamily", "LionMuon"]

NS_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)


def ns(G: torch.Tensor, steps: int = 5, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Newton-Schulz orthogonalization of ``G`` (..., m, n).

    Leading batch dims are independent matrices (stacked same-shape experts).
    Tall matrices (m > n) are transposed for the iteration and transposed
    back; the input is normalized by its Frobenius norm (+1e-7) per matrix.
    Runs ``steps`` iterations in ``dtype`` (fp32 default — PLAN.md Decisions
    log "Numerical precision"; bf16 only in faithful reruns). Never mutates
    its input.
    """
    A, B, C = NS_COEFFS
    X = G.to(dtype)
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        M = X @ X.mT
        X = A * X + (B * M + C * M @ M) @ X
    return X.mT if tall else X


def ns_vec(g: torch.Tensor, steps: int = 5, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Elementwise Newton-Schulz for 1-D parameters, O(d).

    Equals ``torch.diag(ns(torch.diag(g)))`` (unit-tested): NS of a diagonal
    matrix stays diagonal, so the quintic acts elementwise. MuonAll's 1-D
    handling (PLAN.md TL;DR 3) without the d x d allocation.
    """
    A, B, C = NS_COEFFS
    x = g.to(dtype)
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        x = A * x + B * x**3 + C * x**5
    return x


def scale(shape: torch.Size, rule: str) -> float:
    """Shape-dependent update scale; nn.Linear weight = (fan_out, fan_in).

    rule "keller":    sqrt(max(1, fan_out / fan_in))
    rule "moonlight": 0.2 * sqrt(max(fan_out, fan_in))
    Uses the trailing two dims, so expert-stacked (k, m, n) shapes get the
    per-matrix scale.
    """
    m, n = shape[-2], shape[-1]
    if rule == "keller":
        return math.sqrt(max(1, m / n))
    if rule == "moonlight":
        return 0.2 * math.sqrt(max(m, n))
    raise ValueError(f"Unknown scale rule: {rule!r}")


class MuonFamily(torch.optim.Optimizer):
    """Muon / Moonlight-Muon / MuonAll — PLAN.md's reference step.

    buf = momentum*buf + grad;  u = grad + momentum*buf   (Nesterov)
    1-D params: u = ns_vec(u);  >=2-D: u = ns(u) * scale(shape, rule)
    p *= 1 - lr*wd (decoupled);  p -= lr*u
    """

    def __init__(
        self,
        params,
        lr: float,
        wd: float = 0.0,
        momentum: float = 0.95,
        rule: str = "keller",
        ns_steps: int = 5,
        ns_dtype: torch.dtype = torch.float32,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if wd < 0.0:
            raise ValueError(f"Invalid weight decay: {wd}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        scale(torch.Size([1, 1]), rule)  # validate the rule name early
        defaults = dict(
            lr=lr, wd=wd, momentum=momentum, rule=rule,
            ns_steps=ns_steps, ns_dtype=ns_dtype,
        )
        super().__init__(params, defaults)

    def _nesterov(self, p: torch.nn.Parameter, momentum: float) -> torch.Tensor:
        """Update p's momentum buffer in place and return the Nesterov update
        u = grad + momentum * buf (the raw gradient at momentum=0)."""
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
        buf = state["momentum_buffer"]
        buf.mul_(momentum).add_(p.grad)
        return p.grad + momentum * buf

    @torch.no_grad()
    def nesterov_update(self, p: torch.nn.Parameter) -> torch.Tensor:
        """Public buffer-update + Nesterov u for a managed parameter; shared
        with ManifoldMuon so "same Nesterov buffer as Muon" is one code path
        (design.md section 3.1)."""
        for group in self.param_groups:
            if any(q is p for q in group["params"]):
                return self._nesterov(p, group["momentum"])
        raise KeyError("Parameter is not managed by this optimizer")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                u = self._nesterov(p, g["momentum"])
                if p.ndim == 1:
                    u = ns_vec(u, steps=g["ns_steps"], dtype=g["ns_dtype"])
                else:
                    u = ns(u, steps=g["ns_steps"], dtype=g["ns_dtype"]) * scale(
                        p.shape, g["rule"]
                    )
                p.mul_(1 - g["lr"] * g["wd"])
                p.add_(u.to(p.dtype), alpha=-g["lr"])
        return loss


class LionMuon(torch.optim.Optimizer):
    """Lion-Muon: a Muon step every ``period`` steps, a Lion (sign) step
    otherwise, on ONE shared EMA buffer (reference: brain-lab-research/
    lion-muon @ e9077a8, ``src/optim/lion_muon.py``, Algorithm 1 — single
    process, non-adaptive, ``nesterov=False``; vendored unmodified under
    third_party/lion_muon/ as the test oracle, never a dependency).

    Per matrix, with EMA buffer m (zeros at start) and step counter t = 0, 1, ...

        u = beta1 * m + (1 - beta1) * g
        t % period == 0:  u = ns(u) * scale(shape, "moonlight"),  eta = lr
        otherwise:        u = sign(u),                            eta = lr / alpha
        p *= 1 - eta * wd (decoupled, with THIS step's eta);  p -= eta * u
        m = beta2 * m + (1 - beta2) * g

    so step 0 is a Muon step. Newton-Schulz and the Moonlight scale are the
    Muon family's shared ``ns`` / ``scale`` (one NS code path); stacked
    (k, m, n) expert tensors are k independent matrices, as in MuonFamily.

    * The sign-step rate follows the lr schedule: the group stores the RATIO
      ``sign_lr_ratio`` = 1 / alpha and the sign rate is derived from the
      group's CURRENT ``lr`` inside ``step()`` (train.py's scheduler rewrites
      only ``group["lr"]``).
    * The step counter lives in the per-parameter ``state`` ("step"), so
      ``state_dict()`` carries it across checkpoint save/restore. It is
      advanced for EVERY managed parameter on every ``step()`` call — also
      for one without a gradient this step — so it is the reference's single
      global counter: all parameters take their Muon steps together. A
      parameter without a gradient is otherwise left untouched (no update, no
      decay, buffer unchanged), exactly like MuonFamily.
    * ``last_step_muon`` reports whether the latest ``step()`` call was a
      Muon step (None before the first call); train.py's FLOP accounting
      reads it, so the recorded optimizer FLOPs count the Muon steps actually
      taken.

    Matrices only (ndim >= 2): 1-D parameters belong to the AdamW group.
    """

    def __init__(
        self,
        params,
        lr: float,
        wd: float = 0.0,
        alpha: float = 30.0,
        beta1: float = 0.9,
        beta2: float = 0.99,
        period: int = 5,
        ns_steps: int = 5,
        ns_dtype: torch.dtype = torch.float32,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if wd < 0.0:
            raise ValueError(f"Invalid weight decay: {wd}")
        if not alpha > 0.0:
            raise ValueError(f"Invalid alpha (lr / sign-step lr): {alpha}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if int(period) != period or period < 1:
            raise ValueError(f"Invalid period: {period}")
        defaults = dict(
            lr=lr, wd=wd, sign_lr_ratio=1.0 / alpha, beta1=beta1, beta2=beta2,
            period=int(period), ns_steps=ns_steps, ns_dtype=ns_dtype,
        )
        super().__init__(params, defaults)
        self.last_step_muon: bool | None = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        took_muon = False
        for g in self.param_groups:
            beta1, beta2 = g["beta1"], g["beta2"]
            for p in g["params"]:
                state = self.state[p]
                t = state.get("step", 0)
                state["step"] = t + 1          # global counter, see docstring
                is_muon = t % g["period"] == 0
                took_muon = took_muon or is_muon
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    raise ValueError(
                        "LionMuon handles matrices only (ndim >= 2); got a "
                        f"parameter of shape {tuple(p.shape)}")
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                grad = p.grad
                u = m * beta1 + grad * (1 - beta1)
                if is_muon:
                    u = ns(u, steps=g["ns_steps"], dtype=g["ns_dtype"]) * scale(
                        p.shape, "moonlight"
                    )
                    step_lr = g["lr"]
                else:
                    u = torch.sign(u)
                    step_lr = g["lr"] * g["sign_lr_ratio"]
                p.mul_(1 - step_lr * g["wd"])
                p.add_(u.to(p.dtype), alpha=-step_lr)
                m.mul_(beta2).add_(grad, alpha=1 - beta2)
        self.last_step_muon = took_muon
        return loss


# ---------------------------------------------------------------------------
# Parameter groups + optimizer construction (design.md section 3.1, Protocol
# "Optimizers and parameter groups" table; appended at Gate 1).
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402


@dataclass
class ParamGroups:
    """The parameter -> group map, carried as (name, param) pairs.

    muon:  NS-updated params (hidden matrices; EVERY param under muonall).
    adamw: AdamW-updated params (empty under muonall).
    The map goes into the run config (``as_config``), is printed at startup
    (``table``) and reports the Muon share (``fractions``): about 0.69 on the
    dense transformer, > 0.99 on the MLP (Protocol).
    """

    muon: list = field(default_factory=list)
    adamw: list = field(default_factory=list)

    def fractions(self) -> dict:
        n_muon = sum(p.numel() for _, p in self.muon)
        n_adamw = sum(p.numel() for _, p in self.adamw)
        total = n_muon + n_adamw
        if total == 0:
            raise ValueError("ParamGroups is empty")
        return {"muon": n_muon / total, "adamw": n_adamw / total}

    def table(self) -> str:
        rows = [(n, "muon", p) for n, p in self.muon]
        rows += [(n, "adamw", p) for n, p in self.adamw]
        width = max(len(n) for n, _, _ in rows)
        lines = [
            f"{n:<{width}}  {g:<5}  {tuple(p.shape)!s:<16} {p.numel():>9,}"
            for n, g, p in rows
        ]
        f = self.fractions()
        lines.append(
            f"fractions: muon {f['muon']:.4f}  adamw {f['adamw']:.4f}"
        )
        return "\n".join(lines)

    def as_config(self) -> dict:
        out = {n: "muon" for n, _ in self.muon}
        out.update({n: "adamw" for n, _ in self.adamw})
        return out


# Explicit name patterns for the transformer split (verified against the real
# Qwen3/Qwen3Moe modules on transformers 4.57.6 in tests/test_param_groups.py).
# Muon group: attention projections, the dense SwiGLU, and each MoE expert's
# three separate matrices.
_TRANSFORMER_MUON_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",   # dense SwiGLU gate — NOT the MoE router
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)
# AdamW group: embeddings/tied head (yielded once under tying), every norm
# gain (input/post-attention layernorms, final norm, QK-norm q_norm/k_norm),
# every bias, and the MoE router ``mlp.gate.weight`` (a Qwen3MoeSparseMoeBlock
# attribute — distinct from the dense FFN's ``gate_proj``).
_TRANSFORMER_ADAMW_NAMES = ("model.embed_tokens.weight", "lm_head.weight")


def _classify_transformer(name: str, param) -> str:
    if ".mlp.experts." in name and name.endswith(
        (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")
    ):
        return "muon"
    if name.endswith(_TRANSFORMER_MUON_SUFFIXES):
        return "muon"
    if name in _TRANSFORMER_ADAMW_NAMES:
        return "adamw"
    if name.endswith(".bias"):
        return "adamw"
    if name.endswith("norm.weight") and param.ndim == 1:
        return "adamw"  # input/post_attention layernorms, model.norm, q/k_norm
    if name.endswith(".mlp.gate.weight"):
        return "adamw"  # MoE router
    raise RuntimeError(
        f"split_params: unassigned transformer parameter {name!r} "
        f"(shape {tuple(param.shape)}) — extend the explicit pattern list"
    )


def _classify_mlp(name: str, param, classifier_weight: str) -> str:
    if name.endswith(".bias"):
        return "adamw"
    if name == classifier_weight:
        return "adamw"  # the w->10 classifier
    if name.endswith(".weight") and param.ndim == 2:
        return "muon"   # input 3072->w and hidden w->w matrices
    raise RuntimeError(
        f"split_params: unassigned MLP parameter {name!r} "
        f"(shape {tuple(param.shape)})"
    )


def split_params(model, optimizer: str, arch: str) -> "ParamGroups":
    """Protocol parameter-group table, verbatim.

    adamw                   -> everything in the adamw group.
    muon/moonlight/manifold/lionmuon -> muon: attention q/k/v/o, dense SwiGLU
        gate/up/down, each MoE expert's gate/up/down separately, MLP input
        and hidden matrices; adamw: embeddings/tied head (once), CIFAR
        classifier, all norms incl. QK-norm gains, all biases, MoE router.
    muonall                 -> EVERY parameter in the muon group (matrices by
        ns — the tied embedding once, Keller scale sqrt(4096/128) ~ 5.66;
        1-D params by ns_vec; MuonFamily dispatches on ndim).

    Raises on unknown names (nothing is silently dropped) and verifies the
    split is complete and disjoint.
    """
    if optimizer not in ("adamw", "muon", "moonlight", "manifold", "muonall",
                         "lionmuon"):
        raise ValueError(f"Unknown optimizer: {optimizer!r}")
    if arch not in ("mlp", "dense", "moe"):
        raise ValueError(f"Unknown arch: {arch!r}")

    named = list(model.named_parameters())  # tied embedding yielded ONCE
    groups = ParamGroups()

    if optimizer == "adamw":
        groups.adamw = list(named)
    elif optimizer == "muonall":
        groups.muon = list(named)
    else:
        if arch == "mlp":
            import torch.nn as nn
            linear_names = [
                n for n, m in model.named_modules() if isinstance(m, nn.Linear)
            ]
            if not linear_names:
                raise RuntimeError("split_params: MLP has no nn.Linear layers")
            classifier_weight = f"{linear_names[-1]}.weight"
            classify = lambda n, p: _classify_mlp(n, p, classifier_weight)
        else:
            classify = _classify_transformer
        for name, param in named:
            getattr(groups, classify(name, param)).append((name, param))

    seen = [n for n, _ in groups.muon] + [n for n, _ in groups.adamw]
    if len(seen) != len(set(seen)):
        dupes = sorted({n for n in seen if seen.count(n) > 1})
        raise RuntimeError(f"split_params: doubly-assigned parameters {dupes}")
    missing = {n for n, _ in named} - set(seen)
    if missing:
        raise RuntimeError(f"split_params: unassigned parameters {sorted(missing)}")
    return groups


_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _as_dtype(value) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    return _DTYPES[value]


def build_optimizers(cfg: dict, model) -> tuple:
    """(list of optimizers, ParamGroups) per design.md 3.1 / Gate-1 semantics.

    Gate-0 answer 12: in Muon AND Manifold runs the AdamW group uses
    cfg["adamw_group_lr"] AND cfg["adamw_group_wd"] (= AdamW's tuned values
    for that study); the optimizer's own tuned wd applies to its Muon group
    only. Moonlight shares one lr and one wd across both groups. Lion-Muon
    follows Muon/Manifold: its AdamW group uses adamw_group_lr/wd, its own
    (lr, wd) apply to the matrix group; alpha/beta1/beta2/period come from
    the hashed cfg["lionmuon"] dict (all four REQUIRED — no silent defaults).
    """
    from muonarch.manifold import ManifoldMuon  # local: keeps optim <- msign-free

    name = cfg["optimizer"]
    groups = split_params(model, name, cfg["arch"])
    betas = cfg.get("adamw_betas")
    betas = (0.9, 0.95) if betas is None else tuple(betas)
    eps = cfg.get("adamw_eps")
    eps = 1e-8 if eps is None else eps
    momentum = cfg.get("momentum", 0.95)
    ns_steps = cfg.get("ns_steps", 5)
    ns_dtype = _as_dtype(cfg.get("ns_dtype", "fp32"))
    muon_params = [p for _, p in groups.muon]
    adamw_params = [p for _, p in groups.adamw]

    def adamw_group(lr, wd):
        return torch.optim.AdamW(
            adamw_params, lr=lr, weight_decay=wd, betas=betas, eps=eps
        )

    if name == "adamw":
        opts = [adamw_group(cfg["lr"], cfg["wd"])]
    elif name == "muon":
        opts = [
            MuonFamily(muon_params, lr=cfg["lr"], wd=cfg["wd"],
                       momentum=momentum, rule="keller",
                       ns_steps=ns_steps, ns_dtype=ns_dtype),
            adamw_group(cfg["adamw_group_lr"], cfg["adamw_group_wd"]),
        ]
    elif name == "moonlight":
        opts = [
            MuonFamily(muon_params, lr=cfg["lr"], wd=cfg["wd"],
                       momentum=momentum, rule="moonlight",
                       ns_steps=ns_steps, ns_dtype=ns_dtype),
            adamw_group(cfg["lr"], cfg["wd"]),  # one lr+wd shared by design
        ]
    elif name == "manifold":
        m = cfg["manifold"]
        opts = [
            ManifoldMuon(
                muon_params, lr=cfg["lr"],
                dual_cap=m.get("dual_cap", 5), tol=m.get("tol", 1e-6),
                dual_lr=m.get("dual_lr", 0.01), momentum=momentum,
                msign_steps=m.get("msign_steps", 10),
                msign_dtype=_as_dtype(m.get("msign_dtype", "fp32")),
            ),  # no wd on constrained matrices (Protocol)
            adamw_group(cfg["adamw_group_lr"], cfg["adamw_group_wd"]),
        ]
    elif name == "muonall":
        opts = [
            MuonFamily(muon_params, lr=cfg["lr"], wd=cfg["wd"],
                       momentum=momentum, rule="keller",
                       ns_steps=ns_steps, ns_dtype=ns_dtype),
        ]
    elif name == "lionmuon":
        lm = cfg["lionmuon"]
        opts = [
            LionMuon(muon_params, lr=cfg["lr"], wd=cfg["wd"],
                     alpha=lm["alpha"], beta1=lm["beta1"], beta2=lm["beta2"],
                     period=lm["period"],
                     ns_steps=ns_steps, ns_dtype=ns_dtype),
            adamw_group(cfg["adamw_group_lr"], cfg["adamw_group_wd"]),
        ]
    else:  # unreachable: split_params validated the name
        raise ValueError(f"Unknown optimizer: {name!r}")
    return opts, groups


__all__ += ["ParamGroups", "split_params", "build_optimizers"]
