"""Single-run training loop: run(), evaluate(), diagnostics(), checkpoints,
JSONL/result schemas (design.md §3.8).

Loss conventions
----------------
LM: Hugging Face causal-LM ``labels=`` shifts labels internally, which drops
one prediction per window. Our batches already provide aligned ``(x, y)``
pairs (data.py: a 257-token window gives x = first 256 tokens, y = the same
window shifted by one — 256 next-token targets), so the cross-entropy is
computed MANUALLY over every position::

    CE = F.cross_entropy(logits.view(-1, V), y.view(-1))

with NO internal shifting — ``logits[:, :-1]`` is never used and every token
is predicted exactly once. ``labels=`` is never passed to the model. Verified
against transformers' own shifted loss in tests/test_train.py.

MoE: in transformers 4.57.6 ``Qwen3MoeForCausalLM.forward`` returns
``outputs.aux_loss`` = the UNcoefficiented Switch-Transformer load-balancing
loss whenever ``output_router_logits=True`` (modeling_qwen3_moe.py lines
671-678 compute it via ``load_balancing_loss_func``); the model multiplies it
by ``router_aux_loss_coef`` only inside ``if labels is not None`` (lines
679-680), which we never trigger. train.py therefore adds
``cfg["router_aux_loss_coef"] * aux_loss`` to the training loss itself and
logs the raw aux loss separately.

MLP: ``F.cross_entropy(logits, labels)``.

Checkpoint convention (budget-constant runs)
--------------------------------------------
The checkpoint labelled ``step`` = k (k = ``checkpoint_step(base, m)``) is
saved at the BOUNDARY ENTERING step k: after step k-1 has fully completed
(parameter update applied, optimizer state advanced, k training batches
consumed) and before step k executes. It captures the model state_dict, every
optimizer state_dict, the global step k, the data position (k batches
consumed of the seed's fixed permutation — data.py's prefix contract makes
the batch sequence a pure function of (seed, step index)), and the torch RNG
state. A branch run loads it and executes global steps k .. branch_total-1,
which replays this run's own continuation bit-exactly on CPU
(tests/test_train.py::test_checkpoint_roundtrip_bit_exact): both runs' lr
histories agree below each one's own decay, and both consume the same
permutation prefix.

Divergence (design.md §9, round-2 answer 10)
--------------------------------------------
NaN/Inf in the training loss at ANY step, or the train-batch loss or the val
loss above the THEORETICAL initial loss at any logged step — ln(4096) for the
LM (vocab size from the model config), ln(10) for CIFAR (MLP_OUT classes) —
stops the run immediately. The result is then written with
``status="diverged"`` and ``diverged_at_step``; a diverged run is a COMPLETED
run (it must not re-run). The "+0.5 nats above best" criterion is
analysis-time only and never stops a run.

Measured fact (independent review): at seed-0 init the dense LM's val loss is
~8.3404, i.e. ~0.023 ABOVE the theoretical threshold ln 4096 = 8.3178 — the
comparison is a strict ``>`` with no grace margin, so a run whose loss has
not moved below the threshold by its first logged eval is killed as diverged
BY DESIGN. No planned grid configuration gets near the false-kill regime (it
would take lr <~ 1e-5 to still be at the initial loss after 100 steps).

Manifold statistics (coordinator note, Gate-1 / DEVIATIONS.md #7)
-----------------------------------------------------------------
``ManifoldMuon.step()`` returns per-matrix lists whose length VARIES by step:
a parameter with ``grad is None`` (an MoE expert that received no tokens) is
skipped and contributes no entries. Window summaries (``dual_steps_mean``
etc.) therefore average over whatever entries each step returned, and the
FLOP accumulation reconstructs each step's per-shape-group msign counts from
the documented step() ordering (shape groups in first-seen order over the
params that had grads), so skipped matrices contribute zero FLOPs that step.
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from muonarch import flops as flops_mod
from muonarch import runner
from muonarch.data import (
    Cifar5m,
    FineWeb,
    cifar_batches,
    lm_batches,
    lm_val_batches,
    normalize,
)
from muonarch.manifold import ManifoldMuon, grouped_by_shape
from muonarch.models import MLP_OUT, build_model
from muonarch.optim import build_optimizers
from muonarch.schedule import (
    CHECKPOINT_FRACTIONS,
    branch_steps,
    checkpoint_step,
    make_schedule,
)

__all__ = [
    "run",
    "evaluate",
    "diagnostics",
    "setup",
    "lm_cross_entropy",
    "lm_forward_loss",
    "clip_gradients",
    "divergence_threshold",
    "save_checkpoint",
    "load_checkpoint",
]


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def lm_cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean CE over every position: logits (B, L, V) vs targets y (B, L).

    NO internal shifting — y already holds the next-token target for every
    position (data.py window convention), so ``logits[:, :-1]`` is never
    used and each token is predicted exactly once.
    """
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def lm_forward_loss(model, x: torch.Tensor, y: torch.Tensor,
                    aux_coef: float | None = None):
    """(loss, ce, aux) for one LM batch. ``labels=`` is NEVER passed.

    aux is the model's uncoefficiented load-balancing loss (MoE with
    output_router_logits=True; None on the dense model); the training loss
    adds ``aux_coef * aux`` explicitly (see module docstring).
    """
    out = model(input_ids=x, use_cache=False)
    ce = lm_cross_entropy(out.logits.float(), y)
    aux = getattr(out, "aux_loss", None)
    loss = ce if aux is None or aux_coef is None else ce + aux_coef * aux
    return loss, ce, aux


def mlp_forward_loss(model, x: torch.Tensor, y: torch.Tensor):
    """(loss, loss, None) for one MLP batch — plain cross-entropy."""
    loss = F.cross_entropy(model(x), y)
    return loss, loss, None


def divergence_threshold(cfg: dict, model) -> float:
    """The theoretical initial loss: ln(10) for the MLP (MLP_OUT classes),
    ln(vocab) for the LMs (from the model config)."""
    if cfg["arch"] == "mlp":
        return math.log(MLP_OUT)
    return math.log(model.config.vocab_size)


# ---------------------------------------------------------------------------
# gradient clipping
# ---------------------------------------------------------------------------

def clip_gradients(params, clip: float | None) -> float:
    """Jointly clip the global gradient norm over ALL given parameters at
    ``clip`` (None = off — the Phase-I sweep includes off/0.5/1.0). Returns
    the PRE-clip global norm either way (logged as ``grad_norm``)."""
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    if clip is not None:
        return float(torch.nn.utils.clip_grad_norm_(params, clip))
    return float(
        torch.linalg.vector_norm(
            torch.stack([p.grad.detach().norm() for p in params])
        )
    )


# ---------------------------------------------------------------------------
# setup (paired-seed contract)
# ---------------------------------------------------------------------------

def setup(cfg: dict, device: torch.device):
    """(model, optimizers, groups). ``torch.manual_seed(cfg["seed"])`` runs
    IMMEDIATELY before build_model, so seed s gives identical init across
    optimizers (paired comparisons). ``ManifoldMuon.project_`` is applied to
    the muon group iff cfg["init_projection"] — the only sanctioned init
    difference (manifold runs and the Muon init ablation)."""
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg)
    model = model.to(device=device, dtype=torch.float32)
    model.train()
    opts, groups = build_optimizers(cfg, model)
    if cfg.get("init_projection"):
        ManifoldMuon.project_([p for _, p in groups.muon])
    return model, opts, groups


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

_DATA_DIRS = {"cifar5m": "cifar5m", "fineweb_edu": "fineweb"}


def open_data(cfg: dict, root: Path, device: torch.device):
    """Open the dataset named by cfg["data"]. The location comes from the
    logged-only cfg["data_path"] (machine-dependent) or defaults to
    <root>/data/<dataset dir>; the hashed cfg["data"]["version"] pins the
    CONTENT and is asserted against the on-disk meta. Optional size overrides
    inside cfg["data"] (test-only, hashed — a shrunken world is a different
    run identity): n_train/n_val/n_test for CIFAR; min_train_tokens/
    val_tokens for FineWeb."""
    d = cfg["data"]
    path = Path(cfg.get("data_path") or Path(root) / "data" / _DATA_DIRS[d["dataset"]])
    if d["dataset"] == "cifar5m":
        kwargs = {k: d[k] for k in ("n_train", "n_val", "n_test") if k in d}
        ds = Cifar5m(path, device, **kwargs)
    elif d["dataset"] == "fineweb_edu":
        kwargs = {k: d[k] for k in ("min_train_tokens", "val_tokens") if k in d}
        ds = FineWeb(path, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {d['dataset']!r}")
    if ds.version() != d["version"]:
        raise RuntimeError(
            f"data version mismatch: config pins {d['version']!r}, "
            f"on-disk data is {ds.version()!r}"
        )
    return ds


def train_batches(cfg: dict, data, total_steps: int):
    """The run's training iterator — a pure function of (seed, steps, batch
    geometry): same seed, same order, every optimizer; every budget takes a
    prefix. The LM pool is the FIXED default (data.LM_WINDOW_POOL) — never
    derived from the file size; cfg["data"]["pool_windows"] exists only as an
    explicit, hashed test-world override."""
    if cfg["arch"] == "mlp":
        return cifar_batches(data, seed=cfg["seed"], steps=total_steps,
                             batch=cfg["batch"])
    kwargs = {}
    if cfg["data"].get("pool_windows") is not None:
        kwargs["pool_windows"] = cfg["data"]["pool_windows"]
    return lm_batches(data, seed=cfg["seed"], steps=total_steps,
                      batch_seqs=cfg["batch"], seq_len=cfg["seq_len"], **kwargs)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, cfg: dict, data, device: torch.device,
             final: bool = False) -> dict:
    """Loss (and accuracy) on the fixed evaluation sets.

    LM: manual CE (same no-shift convention as training; no aux loss) over
    the first cfg["eval"]["val_small_tokens"] tokens of val.bin, or the whole
    file when final. MLP: full val split every eval; the test split only when
    final.
    """
    model.eval()
    try:
        if cfg["arch"] == "mlp":
            out = {}
            splits = [("val", data.x_val, data.y_val)]
            if final:
                splits.append(("test", data.x_test, data.y_test))
            for name, xs, ys in splits:
                loss_sum, correct = 0.0, 0
                for i in range(0, xs.shape[0], 1024):
                    x = normalize(xs[i:i + 1024])
                    y = ys[i:i + 1024]
                    logits = model(x)
                    loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
                    correct += int((logits.argmax(-1) == y).sum())
                out[f"{name}_loss"] = loss_sum / xs.shape[0]
                out[f"{name}_acc"] = correct / xs.shape[0]
            return out

        n_tokens = len(data.val) if final else cfg["eval"]["val_small_tokens"]
        loss_sum, n = 0.0, 0
        for x, y in lm_val_batches(data, n_tokens, batch_seqs=cfg["batch"],
                                   seq_len=cfg["seq_len"]):
            x, y = x.to(device), y.to(device)
            fwd = {"use_cache": False}
            if cfg["arch"] == "moe":
                fwd["output_router_logits"] = False  # aux not needed for eval loss
            logits = model(input_ids=x, **fwd).logits.float()
            loss_sum += float(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                reduction="sum"))
            n += y.numel()
        return {"val_loss": loss_sum / n}
    finally:
        model.train()


# ---------------------------------------------------------------------------
# diagnostics (logged, never tuned on)
# ---------------------------------------------------------------------------

def _sigma_max(W: torch.Tensor) -> float:
    return float(torch.linalg.svdvals(W.detach().float())[0])


def _capture_attention_forward(model, x: torch.Tensor):
    """Run one probe forward with a pre-hook on every self_attn module,
    capturing (module, hidden_states, position_embeddings). Returns
    (model outputs, captures)."""
    captured, hooks = [], []

    def pre_hook(module, args, kwargs):
        hs = kwargs.get("hidden_states")
        if hs is None and args:
            hs = args[0]
        pe = kwargs.get("position_embeddings")
        if pe is None:
            for a in args[1:]:
                if isinstance(a, tuple) and len(a) == 2:
                    pe = a
                    break
        captured.append((module, hs, pe))

    for layer in model.model.layers:
        hooks.append(layer.self_attn.register_forward_pre_hook(
            pre_hook, with_kwargs=True))
    try:
        out = model(input_ids=x, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return out, captured


def _max_attention_logit(captured) -> float | None:
    """Exact pre-softmax attention logits, recomputed from the captured
    inputs with each module's OWN q_proj/k_proj + QK-norm + RoPE + GQA
    repeat + scaling (the eager_attention_forward arithmetic), causally
    masked. Works under any attn implementation — no eager switch needed. If
    the capture failed (an unexpected call convention), returns None and the
    PLAN-sanctioned sigma_max(W_q/W_k) route — always logged alongside —
    stands in."""
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    worst = None
    for module, hs, pe in captured:
        if hs is None or pe is None:
            return None
        hidden_shape = (*hs.shape[:-1], -1, module.head_dim)
        q = module.q_norm(module.q_proj(hs).view(hidden_shape)).transpose(1, 2)
        k = module.k_norm(module.k_proj(hs).view(hidden_shape)).transpose(1, 2)
        cos, sin = pe
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, module.num_key_value_groups)
        logits = torch.matmul(q, k.transpose(-2, -1)) * module.scaling
        T = logits.size(-1)
        causal = torch.ones(T, T, dtype=torch.bool, device=logits.device).tril()
        m = float(logits.masked_fill(~causal, float("-inf")).max())
        worst = m if worst is None else max(worst, m)
    return worst


def _expert_stats(router_logits, num_experts: int, top_k: int) -> dict:
    """Expert token-share entropy and min/max share from the router logits
    the model itself returns (outputs.router_logits — the recorded output of
    every ``mlp.gate``, exactly what a hook on each gate would capture),
    using the same softmax→top-k selection as the aux loss."""
    cat = torch.cat([g.float() for g in router_logits], dim=0)
    sel = torch.topk(F.softmax(cat, dim=-1), top_k, dim=-1).indices
    counts = torch.bincount(sel.reshape(-1), minlength=num_experts).float()
    share = counts / counts.sum()
    entropy = float(-(share * (share + 1e-12).log()).sum())
    return {
        "expert_entropy": entropy,
        "expert_share_min": float(share.min()),
        "expert_share_max": float(share.max()),
    }


@torch.no_grad()
def diagnostics(model, cfg: dict, probe: torch.Tensor | None,
                groups=None, final: bool = False) -> dict:
    """Per-eval diagnostics on the fixed probe batch (design.md §3.8).

    Transformers: per-layer sigma_max of the q_proj/k_proj weights, and the
    exact max attention logit (recomputed via forward pre-hooks — see
    _max_attention_logit); MoE adds expert token-share entropy and min/max
    share. Final eval additionally: spectral norms of every muon-group
    matrix (all architectures).
    """
    out: dict = {}
    if cfg["arch"] in ("dense", "moe") and probe is not None:
        fwd, captured = _capture_attention_forward(model, probe)
        out["max_attn_logit"] = _max_attention_logit(captured)
        out["sigma_max_q"] = [
            _sigma_max(layer.self_attn.q_proj.weight) for layer in model.model.layers
        ]
        out["sigma_max_k"] = [
            _sigma_max(layer.self_attn.k_proj.weight) for layer in model.model.layers
        ]
        if cfg["arch"] == "moe":
            out.update(_expert_stats(
                fwd.router_logits,
                model.config.num_experts,
                model.config.num_experts_per_tok,
            ))
    if final and groups is not None:
        out["spectral_norms"] = {
            name: _sigma_max(p) for name, p in groups.muon if p.ndim >= 2
        }
    return out


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model, optimizers, step: int,
                    data_state: dict, rng_state) -> None:
    """Atomic save of everything a branch needs to replay steps step..total-1
    identically (see module docstring for the boundary convention). The tmp
    file is UNIQUE per writer (mkstemp), so two processes racing on the same
    hash cannot interleave writes into one tmp file; whichever os.replace
    lands last wins whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizers": [opt.state_dict() for opt in optimizers],
                "step": step,
                "data_state": data_state,
                "torch_rng_state": rng_state,
            },
            tmp,
        )
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_checkpoint(path: Path) -> dict:
    """Load with ``map_location="cpu"`` so a checkpoint saved under one
    device (e.g. MPS) is loadable by any process (e.g. a CPU session on the
    same machine). The restore path in run() places tensors on the current
    run's device: model.load_state_dict copies into the already-on-device
    parameters, and optimizer float state is moved to each parameter's device
    after load (torch casts it there itself; we normalize defensively)."""
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------

def _active_params(model) -> int:
    """Parameter count for the 6ND term: tied embedding once (parameters()
    deduplicates); MoE counts only the active top-k fraction of the expert
    parameters."""
    total = sum(p.numel() for p in model.parameters())
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "num_experts"):
        expert = sum(p.numel() for n, p in model.named_parameters()
                     if ".mlp.experts." in n)
        total -= round(expert * (1 - config.num_experts_per_tok / config.num_experts))
    return total


def _model_flops_per_step(cfg: dict, model) -> int:
    if cfg["arch"] == "mlp":
        return flops_mod.mlp_flops_per_step([cfg["width"]] * 3, cfg["batch"])
    c = model.config
    return flops_mod.lm_flops_per_step(
        _active_params(model),
        tokens_per_step=cfg["batch"] * cfg["seq_len"],
        seq_len=cfg["seq_len"],
        n_layers=c.num_hidden_layers,
        n_q_heads=c.num_attention_heads,
        head_dim=c.head_dim,
    )


def _manifold_step_flops(opt, msign_calls: list[int], msign_steps: int) -> int:
    """Reconstruct this step's msign FLOPs from the per-matrix counts, using
    ManifoldMuon.step()'s documented entry order (shape groups in first-seen
    order over the params that HAD grads; one shared count per group).
    Matrices skipped this step (grad None — e.g. unrouted MoE experts)
    contribute zero, as no msign ran for them."""
    total, i = 0, 0
    for group in opt.param_groups:
        with_grad = [p for p in group["params"] if p.grad is not None]
        for shape, plist in grouped_by_shape(with_grad).items():
            k = len(plist)
            chunk = msign_calls[i:i + k]
            if len(chunk) != k or any(c != chunk[0] for c in chunk):
                raise RuntimeError(
                    "manifold msign_calls entries do not match the step's "
                    "shape groups — ManifoldMuon.step() ordering changed?")
            total += chunk[0] * k * flops_mod.ns_matrix_flops(tuple(shape), msign_steps)
            i += k
    if i != len(msign_calls):
        raise RuntimeError(
            f"manifold msign_calls has {len(msign_calls)} entries, "
            f"expected {i} from the shape groups")
    return total


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, obj: dict) -> None:
    """Atomic JSON write with a UNIQUE tmp file per writer (mkstemp): two
    processes racing on the same result path each write their own tmp and the
    last os.replace wins whole — a shared tmp name could interleave into a
    corrupted file (found by review)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(obj, indent=1))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run(cfg: dict, root: Path | str, device: str | torch.device | None = None,
        data=None) -> dict:
    """Execute one run to completion (or divergence) and return its result
    dict — the only entry point the runner calls (design.md §3.8).

    Idempotent: if the result JSON already exists (any status, divergence
    included), it is loaded and returned — nothing is rebuilt or re-run.
    ``device`` defaults to CPU (the deterministic test setting; the runner
    passes the study device).
    ``data`` may inject an already-open dataset (tests); otherwise it is
    opened from cfg (see open_data).
    """
    root = Path(root)
    if runner.is_done(root, cfg):
        rp = runner.result_path(root, cfg)
        try:
            return json.loads(rp.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A corrupted result (e.g. from the pre-fix shared-tmp race) must
            # not wedge the run forever: treat it as not done and overwrite.
            warnings.warn(
                f"corrupt result JSON at {rp}; treating the run as NOT done "
                "and re-running", RuntimeWarning)

    device = torch.device(device or "cpu")
    started_at = _now()
    t0 = time.time()
    h = runner.config_hash(cfg)

    model, opts, groups = setup(cfg, device)
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in opts]
    print(groups.table())

    if data is None:
        data = open_data(cfg, root, device)
    base = cfg["budget_base_steps"]
    total = branch_steps(base, cfg["budget_multiplier"])
    schedule = make_schedule(cfg)
    batches = train_batches(cfg, data, total)
    every = cfg["eval"]["every_steps"]
    aux_coef = cfg.get("router_aux_loss_coef")
    clip = cfg.get("grad_clip")
    threshold = divergence_threshold(cfg, model)
    is_manifold = cfg["optimizer"] == "manifold"

    # ---- branch resume -------------------------------------------------
    start_step = 0
    if cfg.get("branch"):
        br = cfg["branch"]
        ck = load_checkpoint(
            runner.ckpt_file(root, cfg["study"], br["parent_hash"],
                             br["checkpoint_step"]))
        if ck["step"] != br["checkpoint_step"]:
            raise RuntimeError(
                f"checkpoint step {ck['step']} != branch config "
                f"checkpoint_step {br['checkpoint_step']}")
        model.load_state_dict(ck["model"])
        for opt, sd in zip(opts, ck["optimizers"]):
            opt.load_state_dict(sd)
            # torch casts float state to each param's device on load; verify
            # and move explicitly in case a buffer was left behind (int state
            # such as AdamW's "step" counter stays where torch wants it)
            for p, st in opt.state.items():
                for k, v in st.items():
                    if torch.is_tensor(v) and v.is_floating_point() \
                            and v.device != p.device:
                        st[k] = v.to(p.device)
        torch.set_rng_state(ck["torch_rng_state"])
        start_step = ck["step"]
        for _ in range(ck["data_state"]["consumed"]):  # replay the prefix
            next(batches)

    ckpt_steps = {}
    if cfg["phase"] == "budget_constant":
        ckpt_steps = {checkpoint_step(base, m): m for m in CHECKPOINT_FRACTIONS}

    # ---- FLOP bookkeeping ----------------------------------------------
    model_fl_step = _model_flops_per_step(cfg, model)
    muon_mats = [tuple(p.shape) for _, p in groups.muon if p.ndim >= 2]
    vec_sizes = [p.numel() for _, p in groups.muon if p.ndim == 1]
    vec_sizes += [p.numel() for _, p in groups.adamw]
    if is_manifold:
        msign_steps = cfg["manifold"].get("msign_steps", 10)
        opt_overhead = flops_mod.optimizer_flops_per_step(
            [], [p.numel() for p in model.parameters()], "adamw")
        static_opt_fl = None
    else:
        static_opt_fl = flops_mod.optimizer_flops_per_step(
            muon_mats, vec_sizes, cfg["optimizer"], ns_steps=cfg["ns_steps"])
    flops_total = {"model": 0, "optimizer": 0}

    logged_only = {
        "device": str(device),
        "hostname": socket.gethostname(),
        "precision": cfg.get("precision"),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "git_commit": _git_commit(),
    }

    log_file = runner.log_path(root, cfg)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # "w": a rerun after a crash (no result file) restarts the log cleanly
    f = open(log_file, "w")

    def log(obj: dict) -> None:
        f.write(json.dumps(obj) + "\n")
        f.flush()

    units_per_step = cfg["batch"] * (cfg["seq_len"] or 1)
    probe = None  # lazily-built fixed diagnostics batch

    status, diverged_at = "completed", None
    final_payload: dict = {}
    # window accumulators (reset at each eval)
    w_losses: list[float] = []
    w_aux: list[float] = []
    w_dual: list[float] = []
    w_resid: list[float] = []
    w_calls: list[float] = []
    # run-level manifold means (per-step means of whatever entries arrived)
    run_dual: list[float] = []
    run_calls: list[float] = []
    last_train_mean: float | None = None

    try:
        log({"t": "start", "hash": h, "config": {**cfg, **logged_only},
             "param_map": groups.as_config(), "fractions": groups.fractions()})

        model.train()
        for step in range(start_step, total):
            # checkpoint at the boundary ENTERING step `step` (see docstring)
            if step in ckpt_steps:
                path = runner.ckpt_path(root, cfg, step)
                save_checkpoint(path, model, opts, step,
                                data_state={"consumed": step},
                                rng_state=torch.get_rng_state())
                log({"t": "ckpt", "step": step, "path": str(path)})

            mult = schedule(step)
            for opt, bases in zip(opts, base_lrs):
                for g, b in zip(opt.param_groups, bases):
                    g["lr"] = b * mult  # each group scales its OWN base lr

            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            if cfg["arch"] == "mlp":
                loss, _, aux = mlp_forward_loss(model, x, y)
            else:
                loss, _, aux = lm_forward_loss(model, x, y, aux_coef)
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_gradients(model.parameters(), clip)

            step_manifold_fl = 0
            for opt in opts:
                stats = opt.step()
                if isinstance(stats, dict) and "msign_calls" in stats:
                    # variable-length lists: average whatever entries arrived
                    w_dual.extend(stats["dual_steps"])
                    w_resid.extend(stats["dual_residual"])
                    w_calls.extend(stats["msign_calls"])
                    if stats["dual_steps"]:
                        run_dual.append(_mean(stats["dual_steps"]))
                        run_calls.append(_mean(stats["msign_calls"]))
                    step_manifold_fl += _manifold_step_flops(
                        opt, stats["msign_calls"], msign_steps)

            flops_total["model"] += model_fl_step
            flops_total["optimizer"] += (
                opt_overhead + step_manifold_fl if is_manifold else static_opt_fl)

            loss_item = float(loss.detach())
            w_losses.append(loss_item)
            if aux is not None:
                w_aux.append(float(aux.detach()))

            done = step + 1
            if not math.isfinite(loss_item):
                status, diverged_at = "diverged", done
                break

            if done % every == 0:
                ev = evaluate(model, cfg, data, device)
                if probe is None and cfg["arch"] in ("dense", "moe"):
                    probe = next(lm_val_batches(
                        data, cfg["eval"]["val_small_tokens"] or len(data.val),
                        batch_seqs=cfg["batch"], seq_len=cfg["seq_len"],
                    ))[0].to(device)
                diag = diagnostics(model, cfg, probe, groups)
                record = {
                    "t": "eval",
                    "step": done,
                    "images_or_tokens": done * units_per_step,
                    "lr": opts[0].param_groups[0]["lr"],
                    "train_loss": _mean(w_losses),
                    "train_loss_last": loss_item,
                    "val_loss": ev["val_loss"],
                    "val_acc": ev.get("val_acc"),
                    "grad_norm": grad_norm,
                    "aux_loss": _mean(w_aux),
                    "dual_steps_mean": _mean(w_dual),
                    "dual_residual_mean": _mean(w_resid),
                    "msign_calls_mean": _mean(w_calls),
                    "max_attn_logit": diag.get("max_attn_logit"),
                    "expert_entropy": diag.get("expert_entropy"),
                    "expert_share_min": diag.get("expert_share_min"),
                    "expert_share_max": diag.get("expert_share_max"),
                }
                for key in ("sigma_max_q", "sigma_max_k"):
                    if key in diag:
                        record[key] = diag[key]
                log(record)
                last_train_mean = record["train_loss"]
                w_losses, w_aux = [], []
                w_dual, w_resid, w_calls = [], [], []
                # divergence: train-batch loss or val loss above the
                # theoretical initial loss at a LOGGED step, or NaN
                checks = (loss_item, ev["val_loss"])
                if any(not math.isfinite(v) or v > threshold for v in checks):
                    status, diverged_at = "diverged", done
                    break

        if status == "completed":
            ev = evaluate(model, cfg, data, device, final=True)
            if not math.isfinite(ev["val_loss"]) or ev["val_loss"] > threshold:
                status, diverged_at = "diverged", total
                # keep the computed full-set numbers in the result anyway
                final_payload = {
                    "final_val_loss": ev["val_loss"],
                    "final_val_acc": ev.get("val_acc"),
                    "final_test_loss": ev.get("test_loss"),
                    "final_test_acc": ev.get("test_acc"),
                }
            else:
                diag = diagnostics(model, cfg, probe, groups, final=True)
                flops_result = {
                    "model": flops_total["model"],
                    "optimizer": flops_total["optimizer"],
                    "total": flops_total["model"] + flops_total["optimizer"],
                }
                final_payload = {
                    "final_val_loss": ev["val_loss"],
                    "final_val_acc": ev.get("val_acc"),
                    "final_test_loss": ev.get("test_loss"),
                    "final_test_acc": ev.get("test_acc"),
                    "spectral_norms": diag.get("spectral_norms", {}),
                }
                log({"t": "final",
                     "val_loss": ev["val_loss"],
                     "test_loss": ev.get("test_loss"),
                     "test_acc": ev.get("test_acc"),
                     "spectral_norms": final_payload["spectral_norms"],
                     "flops": flops_result})
    finally:
        f.close()

    result = {
        "hash": h,
        "schema_version": cfg["schema_version"],
        "config": cfg,
        "status": status,
        "diverged_at_step": diverged_at,
        "final_val_loss": final_payload.get("final_val_loss"),
        "final_val_acc": final_payload.get("final_val_acc"),
        "final_test_loss": final_payload.get("final_test_loss"),
        "final_test_acc": final_payload.get("final_test_acc"),
        "train_val_gap": (
            last_train_mean - final_payload["final_val_loss"]
            if last_train_mean is not None
            and final_payload.get("final_val_loss") is not None else None),
        "flops": {
            "model": flops_total["model"],
            "optimizer": flops_total["optimizer"],
            "total": flops_total["model"] + flops_total["optimizer"],
        },
        "dual_steps_mean": _mean(run_dual),
        "msign_calls_mean": _mean(run_calls),
        "edge_rule_fired": None,  # set by the runner (later change)
        "wall_clock_s": time.time() - t0,
        "started_at": started_at,
        "finished_at": _now(),
        **logged_only,
    }
    # written atomically, ONLY on completion/divergence (a crash above never
    # reaches this line, so no partial result can exist)
    _atomic_write_json(runner.result_path(root, cfg), result)
    return result
