# Implementation design — Muon variants × architectures

Companion to `PLAN.md` (the source of truth; its **Decisions log** overrides everything,
including this document). This file specifies module APIs, data formats, the run-config
schema and hash, the runner's gating DAG, and how the WSD schedule supports the AdamW
budget curve. Nothing here changes a decision; every ambiguity found so far has been
answered by the project owner ("Clarifications" below and §9), and PLAN.md wording that
still lags those answers is listed in §9 as lines to be fixed, not resolved here.

## Clarifications (approved by Paul)

Six ambiguities resolved by the project owner; these are clarifications, not deviations,
and are applied throughout this design. They have since been folded into PLAN.md itself
as Decisions-log rows ("Phase 0 clarifications" through "Dense vs MoE controls",
PLAN.md:75–81). A second round of thirteen answers is recorded in §9.

1. **MLP activation and bias**: ReLU, biases on (PyTorch default `nn.Linear`, default
   init). Biases go in the AdamW group for every optimizer except MuonAll (so MuonAll's
   1-D handling is exercised in Study A). The bias-free MLP is only for the Thinking
   Machines oracle in Phase 1.
2. **Branch points vs eval grid**: checkpoint at the exact branch-start step,
   independent of the eval grid. A budget of b× has `round(b × S)` steps, where S is the
   1× step count (2,441 transformers, 3,520 MLP). The branch leaves the constant run at
   `round(0.8 × that)`, decays over the remaining steps, and gets a final full
   evaluation at its own end. Warmup is fixed in absolute steps at `round(0.05 × S)`:
   122 for transformers, 176 for MLP. Example: the 0.125× transformer branch has 305
   steps, branches at step 244, decays over 61. Inside branches, evaluate on the regular
   grid plus at the end.
3. **Manifold Muon Stage B**: best Stage-A LR × 2^k for k ∈ {−0.75, −0.5, −0.25, +0.25,
   +0.5, +0.75} (quarter-octave steps filling both factor-2 intervals around the
   winner). Apply the edge rule before Stage B, as for the other optimizers.
4. **MuonAll and the tied embedding**: one 4,096×128 tensor, counted once, updated by
   Newton–Schulz with Keller scale √max(1, 4096/128) ≈ 5.66 (as MuonAll's reference code
   does). Weight decay applies to it like any other ≥2-D tensor; its Newton–Schulz FLOPs
   are counted.
5. **Half-LR 2× run**: wd unchanged. Because PyTorch applies lr·wd, the effective decay
   per step also halves.
6. **Dense vs MoE**: identical tokenizer, data and data order, 20M tokens, batch 32×256,
   schedule, attention, tied embeddings, eval sets, seeds, clipping. Differs only in:
   the feed-forward block (16 experts of width 96, top-4 routing, active width 384 =
   dense SwiGLU), the router, and the auxiliary loss (coefficient 0.01,
   `norm_topk_prob=True`). The MoE has more total parameters (~3.9M vs ~1.7M), so Muon's
   optimizer FLOPs are higher on it — expected and counted.

## 1. Principles

- **One shared NS code path** (`muonarch/optim.py::ns`) for Muon, Moonlight-Muon and
  MuonAll; Manifold Muon uses the vendored Polar Express `msign` only. Variants differ
  exactly in their stated idea (PLAN TL;DR 1).
- **Runs are identified by a config hash.** A run is done iff
  `results/<study>/<run_hash>.json` exists; the runner skips done runs, so everything is
  resumable and idempotent. Divergence is a *completed* run with `status: "diverged"`
  (it must not re-run, and it feeds the divergence-rate metric).
- **No module imports `runner.py` except scripts.** Dependency direction:
  `msign ← manifold`, `optim/manifold/models/data/schedule/flops ← train`,
  `train + studies ← runner`, `(results/logs on disk) ← analysis`.
- **fp32 everywhere on Macs**; `precision` is a study-wide config field. All tensors
  entering `ns`/`ns_vec`/`msign` are cast to fp32 regardless of model dtype (except the
  MuonAll faithful-settings rerun, which sets `ns_dtype: bf16`).
- **Environment**: `uv` with committed `uv.lock`; `torch>=2.9` (`torch.optim.Muon` is the
  test oracle, never a dependency of `muonarch/optim.py`); `transformers==4.57.6`
  (Qwen3-MoE experts are separate `nn.Linear` in an `nn.ModuleList`; asserted at model
  build).

## 2. Repo layout (fixed)

```
muonarch/
  optim.py      # ns, ns_vec, scale, MuonFamily, split_params, build_optimizers
  msign.py      # vendored Polar Express msign (verbatim function from TML manifolds)
  manifold.py   # ManifoldMuon (adapted: cap exposed, Nesterov buffer, batched experts)
  models.py     # build_mlp / build_dense / build_moe / build_model, assertions
  data.py       # FineWeb memmap + CIFAR-5m tensors, deterministic prefix-consistent order
  schedule.py   # WSD schedule, warmup/checkpoint arithmetic, branch schedules
  flops.py      # analytical FLOPs (model + optimizer), FlopCounterMode verification
  train.py      # single-run training loop, eval, diagnostics, checkpoints, result JSON
  runner.py     # config hashing, done/skip logic, laptop assignment, DAG execution
  studies.py    # study definitions, grids, phase/DAG construction (pure, no I/O side effects)
  analysis.py   # DataFrame loading, curve fit, speedups, seed rule, figures 1-8, table
third_party/manifolds/   # Thinking Machines repo files UNMODIFIED + their MIT LICENSE
scripts/
  prepare_fineweb.py  prepare_cifar5m.py  oracle_tml.py  smoke.py  timing.py
tests/
configs/        # mlp128.yaml mlp256.yaml mlp512.yaml dense.yaml moe.yaml
results/<study>/<run_hash>.json      # written atomically, only on completion
logs/<study>/<run_hash>.jsonl        # streamed eval/diagnostic records
checkpoints/<study>/<run_hash>@<step>.pt   # only for budget-curve constant runs
figures/  tables/  PROGRESS.md  DEVIATIONS.md  REPORT_DRAFT.md
```

`muonarch/msign.py` is the single vendored function (`msign(G, steps)`) copied verbatim
from `third_party/manifolds/src/msign.py` so the package has no import path into
`third_party/`; a unit test asserts the two files' function bodies are identical, which
keeps "vendored, unmodified" checkable.

## 3. Module APIs

### 3.1 `muonarch/optim.py`

```python
NS_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)

def ns(G: Tensor, steps: int = 5, dtype: torch.dtype = torch.float32) -> Tensor
    # (..., m, n); leading batch dims = stacked experts. Transposes tall inputs,
    # normalizes by Frobenius norm (+1e-7), runs `steps` NS iterations in `dtype`.

def ns_vec(g: Tensor, steps: int = 5, dtype: torch.dtype = torch.float32) -> Tensor
    # Elementwise polynomial, O(d); == torch.diag(ns(torch.diag(g))) (unit-tested).

def scale(shape: torch.Size, rule: str) -> float
    # rule "keller":    sqrt(max(1, fan_out / fan_in))   (nn.Linear weight = (out, in))
    # rule "moonlight": 0.2 * sqrt(max(fan_out, fan_in))
    # Test anchors: moonlight((128, 3072)) ≈ 11.09; moonlight((128, 128)) ≈ 2.26;
    # keller((128, 128)) == 1.0.

@dataclass
class ParamGroups:
    muon:  list[tuple[str, nn.Parameter]]   # NS-updated (matrices; +1-D for MuonAll)
    adamw: list[tuple[str, nn.Parameter]]   # AdamW-updated (possibly empty)
    def fractions(self) -> dict[str, float]     # param-count share per group
    def table(self) -> str                      # printable name -> group map
    def as_config(self) -> dict[str, str]       # name -> group, stored in run log

def split_params(model: nn.Module, optimizer: str, arch: str) -> ParamGroups
    # Implements the Protocol parameter-group table verbatim:
    #   adamw:     everything -> adamw
    #   muon/moonlight/manifold: hidden matrices (q,k,v,o; MLP/SwiGLU; each expert's
    #       gate/up/down separately; MLP input 3072->w and hidden w->w) -> muon;
    #       embeddings/tied head, CIFAR classifier, norms, QK-norm gains, biases,
    #       MoE router -> adamw
    #   muonall:   every parameter -> muon (matrices via ns, 1-D params — incl. the MLP
    #       biases, Clarification 1 — via ns_vec, embeddings/tied head via ns); adamw
    #       group empty. The tied embedding/head is ONE 4096x128 tensor, counted once,
    #       NS-updated with Keller scale sqrt(max(1, 4096/128)) ≈ 5.66, wd applied like
    #       any other >=2-D tensor (Clarification 4).
    # Raises if any named parameter is unassigned or doubly assigned.
    # Expected fractions (tested): dense muon share ≈ 0.69; MLP > 0.99; MoE map
    # contains 16×3 separate expert matrices.

class MuonFamily(torch.optim.Optimizer):
    def __init__(self, params, lr: float, wd: float = 0.0, momentum: float = 0.95,
                 rule: str = "keller", ns_steps: int = 5,
                 ns_dtype: torch.dtype = torch.float32)
    # Exactly the PLAN.md reference implementation (Nesterov buffer, ns/ns_vec by
    # p.ndim, decoupled lr*wd decay), parameterized by ns_steps/ns_dtype so the
    # MuonAll faithful rerun (6 steps, bf16) is a config change, not a code change.
    def step(self) -> None
    def nesterov_update(self, p: nn.Parameter) -> Tensor   # buf update + u; shared
        # with ManifoldMuon so "same Nesterov buffer as Muon" is literally one code path.

def build_optimizers(cfg: dict, model: nn.Module) -> tuple[list[Optimizer], ParamGroups]
    # adamw     -> [torch.optim.AdamW(all, lr, wd, betas=cfg["adamw_betas"], eps=1e-8)]
    # muon      -> [MuonFamily(muon, lr, wd, rule="keller"),      # tuned wd: Muon group ONLY
    #               torch.optim.AdamW(adamw, lr=cfg["adamw_group_lr"],
    #                                 wd=cfg["adamw_group_wd"])]  # = AdamW's tuned lr AND wd
    # moonlight -> [MuonFamily(muon, lr, wd, rule="moonlight"),
    #               torch.optim.AdamW(adamw, lr, wd)]             # shared lr/wd by design
    # manifold  -> [ManifoldMuon(muon, lr, cap, tol, dual_lr),    # no wd on constrained
    #               torch.optim.AdamW(adamw, lr=cfg["adamw_group_lr"],
    #                                 wd=cfg["adamw_group_wd"])]  # = AdamW's tuned lr AND wd
    # muonall   -> [MuonFamily(all params, lr, wd, rule="keller")]
    # (Gate-0 answer 12: in Muon and Manifold runs the AdamW group reuses AdamW's tuned
    # lr and wd for that study; the optimizer's own tuned wd never touches that group.)
```

### 3.2 `muonarch/msign.py`

```python
def msign(G: Tensor, steps: int = 10) -> Tensor   # Polar Express; verbatim from TML
POLAR_EXPRESS_STEPS: int = 10
```

Supports leading batch dims `(k, m, n)` exactly as the reference does (the TML code is
already batched); used for both dual ascent and retraction.

### 3.3 `muonarch/manifold.py`

```python
class ManifoldMuon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, dual_cap: int = 5, tol: float = 1e-6,
                 dual_lr: float = 0.01, momentum: float = 0.95,
                 msign_steps: int = 10)
    # Update per Protocol: Nesterov buffer u (same code path as MuonFamily, via
    # optim.MuonFamily.nesterov_update semantics) replaces the raw gradient; dual
    # ascent on Λ capped at dual_cap with early stop at tol; dual step dual_lr
    # decaying linearly over the ascent; update -lr*msign(u + 2W(Λ+Λᵀ)); retraction
    # W <- msign(W). Wide matrices handled transposed-to-tall as in the reference.
    def step(self) -> dict        # {"dual_steps": list[int], "dual_residual": list[float]}
                                  # train.py logs these every step (summarized per eval).
    @staticmethod
    def project_(params: Iterable[nn.Parameter]) -> None   # W <- msign(W) at step 0;
        # also used by the Muon init ablation (same projected init, Muon update).

def grouped_by_shape(params) -> dict[torch.Size, Tensor]
    # Stacks same-shape parameters (MoE experts: 16 per projection shape) into one
    # (k, m, n) tensor so dual ascent + retraction run as single batched msign calls.
```

The faithful-settings rerun for Manifold Muon is `dual_cap=100` via config only.

### 3.4 `muonarch/models.py`

```python
MLP_IN, MLP_OUT = 3072, 10
def build_mlp(width: int) -> nn.Sequential
    # 4 nn.Linear: 3072->w, w->w, w->w, w->10, ReLU between them; biases ON (PyTorch
    # default nn.Linear, default init) — Clarification 1. The bias-free MLP exists only
    # inside third_party/ for the oracle check and is never built here.

COMMON_LM_KWARGS: dict   # exactly the PLAN.md snippet (vocab 4096, hidden 128, 6 layers,
                         # 2 heads / 1 KV head, head_dim 64, max_pos 256, tied embeddings)
def build_dense() -> Qwen3ForCausalLM        # intermediate_size=384; 1.70M params (tested)
def build_moe() -> Qwen3MoeForCausalLM       # 16 experts, top-4, moe_intermediate 96,
                                             # norm_topk_prob=True, aux coef 0.01 (fixed)
# Dense and MoE differ ONLY in the feed-forward block (16 experts of width 96, top-4,
# active width 384 = dense SwiGLU), the router, and the aux loss; tokenizer, data and
# data order, 20M tokens, batch 32x256, schedule, attention, tied embeddings, eval sets,
# seeds and clipping are identical (Clarification 6). MoE total params ~3.9M vs ~1.7M
# dense, so Muon-group optimizer FLOPs are higher on the MoE — expected and counted.
def build_model(cfg: dict) -> nn.Module      # dispatch on cfg["arch"]; calls the asserts
def assert_moe_unfused(model) -> None        # experts is nn.ModuleList of modules whose
                                             # gate/up/down are separate nn.Linear (4.57.6)
def count_params(model) -> dict[str, int]
```

Seeding contract: `train.py` calls `torch.manual_seed(cfg["seed"])` immediately before
`build_model`, so seed s gives identical init across optimizers (paired comparisons).
Manifold Muon (and the init ablation) then applies `ManifoldMuon.project_` to the
muon-group weights — the only sanctioned init difference.

### 3.5 `muonarch/data.py`

```python
# ---- FineWeb-Edu LM data ----
class FineWeb:
    def __init__(self, root: Path)           # expects train.bin, val.bin, tokenizer.json,
                                             # meta.json (validated on open)
    train: np.memmap                         # uint16, little-endian, >= 45_000_000 tokens
    val:   np.memmap                         # uint16, exactly 2_000_000 tokens
    meta:  dict                              # {"version", "n_train", "n_val",
                                             #  "vocab_size": 4096, "tokenizer_sha256",
                                             #  "source": "fineweb-edu/sample-10BT",
                                             #  "tokenizer_trained_on": "train_docs_only"}
    def version(self) -> str                 # meta["version"]; this string is hashed

def lm_batches(fw: FineWeb, seed: int, steps: int, batch_seqs: int = 32,
               seq_len: int = 256) -> Iterator[tuple[Tensor, Tensor]]
    # Non-overlapping 257-token windows (256 predictions each): sequence i covers tokens
    # [i*seq_len, i*seq_len + seq_len + 1) (x = first 256, y = shifted) — Gate-0 answer 7.
    # The window order is a permutation of ALL windows needed for the 2x budget, drawn
    # once from seed; every run takes a prefix.
    # => prefix consistency: the 1x run's data order is exactly the first half of the
    # 2x constant run's order for the same seed (required by the budget curve).

def lm_val_batches(fw: FineWeb, n_tokens: int) -> Iterator[tuple[Tensor, Tensor]]
    # Deterministic, unshuffled; n_tokens = 500_000 (periodic eval = the FIRST 0.5M
    # tokens of val.bin, Gate-0 answer 8) or 2_000_000 (final eval, the whole file).
    # Same sets for every run.

# ---- CIFAR-5m ----
class Cifar5m:
    def __init__(self, root: Path, device: torch.device)
    # Let C = concat(part0, part1) in file order (Gate-0 answer 1):
    x_train: Tensor  # uint8 [1_802_240, 32, 32, 3] = C[:1_802_240] (all of part 0 plus
                     # the start of part 1); the 1x budget uses the first 901_120
    y_train: Tensor  # int64 [1_802_240]
    x_val, y_val:   Tensor  # fixed 10_000 = C[1_802_240:1_812_240] (never trained on)
    x_test, y_test: Tensor  # fixed 10_000 = C[1_812_240:1_822_240] (disjoint from both)
    def version(self) -> str                 # meta.json version string; hashed

def cifar_batches(ds: Cifar5m, seed: int, steps: int, batch: int = 256
                  ) -> Iterator[tuple[Tensor, Tensor]]
    # One seed-determined permutation of all 1_802_240 indices; runs take a prefix
    # (1x = 901_120 images = 3_520 steps; no index appears twice => single pass,
    # no repeats at every budget). Yields normalized fp32 batches via normalize().

def normalize(x_uint8: Tensor) -> Tensor
    # Per-batch on device: fp32, /255, channel mean/std constants fixed in this module;
    # flattened to (batch, 3072). No augmentation.
```

**Disk formats.**

- `data/fineweb/{train.bin, val.bin}`: raw uint16 token ids (vocab 4096 < 2^16),
  nanoGPT-style memmap with one `<|endoftext|>` token appended after each document; the
  EOS token is inside the 4,096 vocab (Gate-0 answer 7). `tokenizer.json`: byte-level
  BPE, vocab 4096 incl. the EOS special token, trained by `scripts/prepare_fineweb.py`
  on **train documents only** (val docs held out *before* tokenizer training);
  `meta.json` as above. Built once, shared by all machines.
- `data/cifar5m/{train.npy, train_labels.npy, val.npy, val_labels.npy, test.npy,
  test_labels.npy, meta.json}`: uint8 HWC images / int64 labels, sliced by
  `scripts/prepare_cifar5m.py` from the concatenation C = part0 ‖ part1 with fixed
  contiguous indices (train = C[:1_802_240]; val = C[1_802_240:1_812_240]; test =
  C[1_812_240:1_822_240]) — Gate-0 answer 1. Download once, share.

### 3.6 `muonarch/schedule.py`

```python
CHECKPOINT_FRACTIONS: tuple[float, ...] = (0.125, 0.25, 0.5)

def warmup_steps(base_steps_1x: int, frac: float = 0.05) -> int
    # round(frac * base_steps_1x), absolute steps, identical for every branch; frac is
    # cfg["schedule"]["warmup_frac_of_1x"], Phase-I tunable over {0, 0.02, 0.05, 0.1}.
    # At the 5% default: 122 (LM), 176 (MLP). See §9 note on the budget curve's warmup.
def branch_steps(base_steps_1x: int, mult: float) -> int   # round(mult * base_steps_1x)
def checkpoint_step(base_steps_1x: int, mult: float) -> int
    # round(0.8 * branch_steps(base, mult)) — the exact step in the CONSTANT run at
    # which the branch for budget `mult` leaves it (80% of the branch budget), taken
    # literally and independent of the eval grid (Clarification 2). `round` is Python's
    # built-in round() in all three functions.

def wsd_multiplier(step: int, total_steps: int, warmup: int,
                   decay_frac: float = 0.2) -> float
    # step < warmup:                 (step + 1) / warmup          (linear warmup)
    # warmup <= step < 0.8*total:    1.0                          (stable)
    # else:                          linear from 1.0 at 0.8*total to 0.0 at total

def make_schedule(cfg: dict) -> Callable[[int], float]
    # Returns step -> lr multiplier for THIS run:
    #   total = branch_steps(base_1x, cfg["budget_multiplier"])
    #   warmup = warmup_steps(base_1x)        # ALWAYS from the 1x budget, every branch
    # For a budget_branch run, `step` is the global step (resumed from the checkpoint),
    # so lr(step) for step < checkpoint equals the constant run's lr(step) identically.
```

See §7 for how this realizes the budget curve.

### 3.7 `muonarch/flops.py`

```python
def model_flops_per_step(cfg: dict) -> int
    # MLP: 2 * (sum of m*n over the 4 Linears) * batch * 3   (fwd + bwd ≈ 3x fwd mults)
    # LM:  6 * N_active * tokens_per_step + F_attn (explicit per-layer attention term);
    # MoE uses active params (top-4 experts) for the 6ND term.
def ns_matrix_flops(shape: torch.Size, iters: int) -> int
    # k * (4*m^2*n + 2*m^3), m = min(dims), n = max(dims)  (PLAN Protocol formula)
def optimizer_flops_per_step(param_groups: ParamGroups, optimizer: str,
                             ns_steps: int, msign_steps: int = 10,
                             dual_steps: float | None = None) -> int
    # muon/moonlight/muonall: sum of ns_matrix_flops over muon-group matrices (iters =
    #   ns_steps), incl. MuonAll's tied embedding counted once (Clarification 4);
    #   ns_vec and AdamW terms ~10 flops/param, included but negligible.
    # manifold: ns_matrix_flops with iters = msign_steps, multiplied by
    #   (dual_steps + 2)  — dual ascent per step taken, + final msign of the update
    #   direction + retraction. `dual_steps` MUST be the logged mean of steps actually
    #   taken (never the cap).
def run_flops(result: dict) -> dict        # {"model": int, "optimizer": int, "total": int}
    # computed from a result JSON, using its logged per-eval dual-step means.
def verify_against_counter(model, optimizer, batch) -> tuple[int, int]
    # (analytic, FlopCounterMode around fwd+bwd+step); test asserts |ratio-1| <= 0.05.
BREAK_EVEN = lambda cfg: ...   # token speedup needed to win per FLOP (H2a reporting)
```

### 3.8 `muonarch/train.py`

```python
def run(cfg: dict, root: Path) -> dict
    # The only entry point the runner calls. Steps:
    # 1. h = runner.config_hash(cfg); return existing result if is_done (idempotent).
    # 2. torch.manual_seed(cfg["seed"]); build model; project init if
    #    cfg["init_projection"]; build_optimizers; print + log ParamGroups.table().
    # 3. Data iterator from cfg (seed, steps, prefix-consistent order);
    #    schedule = make_schedule(cfg).
    # 4. If cfg["branch"]: load checkpoints/<study>/<parent_hash>@<step>.pt
    #    (model, optimizer state incl. momentum buffers, data-stream position, RNG
    #    states) and continue at the checkpoint's global step.
    # 5. Loop: fwd, loss (+ MoE aux loss), bwd, clip_grad_norm_(1.0), set lr from
    #    schedule, step all optimizers. If cfg is a budget-curve constant run, save a
    #    checkpoint at each checkpoint_step(base, m) for m in CHECKPOINT_FRACTIONS.
    # 6. Every eval_every steps: evaluate() + diagnostics(); append one JSONL record.
    #    Divergence check (Gate-0 answer 10): NaN anywhere, or TRAIN or VALIDATION loss
    #    above the THEORETICAL initial loss (ln 4096 = 8.32 LM / ln 10 = 2.30 CIFAR) at
    #    any logged step  =>  stop now, write result with status="diverged" and
    #    diverged_at_step. The "+0.5 nats above the optimizer's best" criterion is
    #    analysis-time only (analysis.divergence_rate) and never stops a run.
    # 7. Final eval (2M-token set / CIFAR test set). Write result JSON atomically
    #    (tmp + os.replace) to results/<study>/<h>.json. Only completion writes it.

def evaluate(model, val_iter, cfg) -> dict      # {"val_loss", "val_acc"? , "aux_loss"?}
def diagnostics(model, batch, cfg) -> dict
    # max attention logit (eager recompute of q·k/sqrt(d) per layer), per-matrix
    # spectral norms (muon group; final eval only), MoE expert-share entropy,
    # min/max expert share. Logged, never tuned on.
def save_checkpoint(path, model, optimizers, step, data_state, rng_state) -> None
def load_checkpoint(path) -> dict
```

**`logs/<study>/<run_hash>.jsonl`** — one JSON object per line:

```jsonc
{"t":"start", "hash":..., "config":{...full, incl. logged-only fields...},
 "param_map":{name:group}, "fractions":{...}}
{"t":"eval", "step":176, "images_or_tokens":45056, "lr":..., "train_loss":...,
 "val_loss":..., "val_acc":..., "grad_norm":..., "max_attn_logit":...,
 "expert_entropy":..., "expert_share_min":..., "expert_share_max":...,
 "aux_loss":..., "dual_steps_mean":..., "dual_residual_mean":...}
{"t":"ckpt", "step":1408, "path":"checkpoints/..."}
{"t":"final", "val_loss":..., "test_loss":..., "test_acc":...,
 "spectral_norms":{name:...}, "flops":{"model":...,"optimizer":...,"total":...}}
```

**`results/<study>/<run_hash>.json`** — written only on completion:

```jsonc
{"hash":..., "schema_version":1, "config":{...}, "status":"completed"|"diverged",
 "diverged_at_step":null|int, "final_val_loss":..., "final_val_acc":...,
 "final_test_loss":..., "final_test_acc":..., "train_val_gap":...,
 "flops":{"model":...,"optimizer":...,"total":...}, "dual_steps_mean":...,
 "edge_rule_fired":bool, "device":"mps|cpu|cuda", "precision":"fp32",
 "torch_version":..., "transformers_version":..., "wall_clock_s":...,
 "git_commit":..., "started_at":..., "finished_at":...}
```

### 3.9 `muonarch/runner.py`

```python
HASHED_FIELDS: frozenset[str]                     # §4 — the identity of a run
def canonical(cfg: dict) -> dict                  # HASHED_FIELDS only, recursively
                                                  # sorted keys, floats via repr()
def config_hash(cfg: dict) -> str                 # sha256(canonical JSON).hexdigest()[:12]
def result_path(root, cfg) -> Path                # results/<study>/<hash>.json
def log_path(root, cfg) -> Path
def ckpt_path(root, cfg, step) -> Path
def is_done(root, cfg) -> bool                    # result JSON exists (any status)

class Runner:
    def __init__(self, root: Path, laptop_index: int, n_laptops: int = 4)
    def run_study(self, study: str) -> None
        # Loop until no phase can advance:
        #   dag = studies.build_dag(study, results_view)      # re-built each pass,
        #   for each phase whose `needs` are all complete:    # since later phases'
        #       for cfg in phase.configs():                   # configs depend on
        #           if is_done(cfg): continue                 # earlier results
        #           if int(config_hash(cfg), 16) % n_laptops != laptop_index: continue
        #           train.run(cfg, root)
        # A phase is complete when every one of its configs is_done.
    def status(self, study) -> str                 # table for PROGRESS.md (manual paste)
```

### 3.10 `muonarch/studies.py`

```python
STUDIES = ("mlp128", "mlp256", "mlp512", "dense", "moe")
OPTIMIZERS = ("adamw", "muon", "moonlight", "manifold", "muonall")

def load_study(name: str) -> dict        # parsed configs/<name>.yaml (fixed fields +
                                         # per-optimizer default lr/wd grid centres)
@dataclass
class Phase:
    name: str                            # e.g. "adamw_stage_a"
    needs: tuple[str, ...]               # phase names that must be complete first
    configs: Callable[[], list[dict]]    # full run configs (lazy: may read results)

PHASE1_STUDIES = ("mlp128", "dense")             # coordinate descent runs here
PHASE2_DONOR = {"mlp256": "mlp128", "mlp512": "mlp128", "moe": "dense"}

def build_dag(study: str, results: ResultsView) -> list[Phase]      # §6

# ---- Phase I: coordinate descent (mlp128 and dense only; PLAN "Hyperparameter search")
CD_ORDER = ("lr", "wd", "warmup", "momentum", "beta2", "clip")
CD_GRIDS = {
    "lr":       "6-point factor-2 grid around the current value",
    "wd":       (0.0, "0.3*d", "d", "3*d"),            # d = plain default (answer 3);
                                                       # manifold: skipped (no wd on
                                                       # constrained matrices)
    "warmup":   (0.0, 0.02, 0.05, 0.10),               # fraction of 1x steps
    "momentum": (0.9, 0.95, 0.98),                     # Muon family; AdamW instead
                                                       # sweeps beta1 (0.8, 0.9, 0.95)
    "beta2":    (0.95, 0.99, 0.999),                   # AdamW only
    "clip":     (None, 0.5, 1.0),
}
def cd_sweep_configs(study, opt, hp: str, current: dict) -> list[dict]
    # Seed-0 configs for one coordinate sweep, all other hps at their current best.
    # Starting point: literature defaults. Cells equal to the current value hash to an
    # existing run and are skipped. The edge rule applies to the lr sweep only.
def cd_accept(candidate_loss, incumbent_loss, pilot_std) -> bool
    # Accept only if final val loss improves by MORE than 2 * pilot_std (the seed std
    # measured in that study's pilot — Wen et al.'s 3e-3 recalibrated to our scale).
def cd_next(study, opt, results) -> list[dict] | None
    # Drives the descent as a chain of Phases (one per sweep): at most 2 full passes
    # over CD_ORDER, stopping early if a pass changes nothing; returns None when done.
def phase1_values(study, opt, results) -> dict   # the accepted hp set after descent;
                                                 # for mlp128/dense this IS the best
                                                 # config (used directly, no Stage A/B)

# ---- Phase II: 12-trial lr x wd recipe (mlp256, mlp512, moe only)
def inherited_values(study, opt, results) -> dict
    # Non-lr/wd hps (warmup, momentum, betas, clip) copied from PHASE2_DONOR[study]'s
    # Phase-I result for this optimizer; copied into every Phase-II config (hashed).
def stage_a_grid(study, opt) -> list[dict]        # 6 lrs, factor-2 log grid spanning
                                                  # 32x, centred on the literature
                                                  # default, at default wd (plain
                                                  # numbers: adamw/moonlight 0.1,
                                                  # muon/muonall 0.01); same positions
                                                  # relative to default at every width
def stage_b_grid(study, opt, stage_a: ResultsView) -> list[dict]
                                                  # 2 best lrs x wd in {0, 0.3*default,
                                                  # 3*default} (default already run in
                                                  # Stage A — Gate-0 answer 5; see §9
                                                  # note on PLAN.md:249); manifold
                                                  # instead: best Stage-A lr (after the
                                                  # edge rule) x 2^k, k in {-0.75,-0.5,
                                                  # -0.25,+0.25,+0.5,+0.75}
def edge_trials(study, opt, results: ResultsView) -> list[dict]
                                                  # <=2 beyond a grid edge; LR grids
                                                  # ONLY (Gate-0 answer 5), in both
                                                  # phases
def best_config(study, opt, results: ResultsView) -> dict           # min seed-0 final
                                                                    # val loss, tuning
                                                                    # trials only
# ---- Shared phases (every study)
def fresh_seed_configs(study, opt, best) -> list[dict]              # seeds 1..3, or 1..5
                                                                    # if the pilot chose
                                                                    # 5 (one seed count
                                                                    # for ALL studies)
def budget_curve_configs(study, adamw_best, results) -> list[dict]
    # §7: constant 2x runs on the curve seeds (1-3 or 1-5), their decay branches, plus
    # the half-LR 2x run on SEED 1 FIRST; if its 2x loss beats seed 1's branched 2x
    # loss, generate half-LR runs for every curve seed (used in every fit), otherwise
    # the branched 2x stands everywhere (Gate-0 answer 4).
def init_ablation_configs(study, muon_best) -> list[dict]           # init_projection=
                                                                    # True, fresh seeds
def faithful_configs(study, results) -> list[dict]
    # THREE reruns per study, all seed 1 (Gate-0 answer 2): Muon with bf16 NS (covers
    # Moonlight: shared routine, and Moonlight's scaling is already faithful); MuonAll
    # at 6 NS steps in bf16; Manifold at 100 dual steps (MLP-512 only if time remains).
def pilot_configs(study) -> list[dict]           # AdamW default cfg, SEEDS 101-103
                                                 # (Gate-0 answer 6); only in mlp128,
                                                 # mlp512, dense. Gates Phase I (its
                                                 # seed std sets cd_accept's threshold)
                                                 # and sets the single 3-vs-5 final
                                                 # seed count for all five studies.
def trial_counts(results) -> dict[str, int]      # trials per optimizer per study, for
                                                 # the report (PLAN Phase II text)
```

### 3.11 `muonarch/analysis.py`

```python
def load_results(root) -> pd.DataFrame           # one row per result JSON
def load_logs(root) -> pd.DataFrame              # one row per JSONL eval record
@dataclass
class FitResult: alpha: float; beta: float; gamma: float
def fit_budget_curve(points: list[tuple[float, float]]) -> FitResult
    # L(D) = alpha * D**-beta + gamma, least squares on the 5 (D, L) points of one seed
def invert_curve(fit: FitResult, loss: float) -> float          # D_AdamW(loss); may be inf
def token_speedup_range(study, df) -> pd.DataFrame
    # per optimizer: range over {3 fitted curves} x {3 optimizer seeds} of
    # invert_curve(fit, L_opt) / D;  values < 1 mean slower than AdamW
def flop_speedup_range(study, df) -> pd.DataFrame     # same ratio on total FLOPs
def lr_basin_width(study, opt, df, tol) -> float
    # width in factors of 2 of the lr range (Stage A / Phase-I lr sweep) within tol of
    # the best (tol = 0.02 nats LM; MLP = 1% RELATIVE final val loss, Gate-0 answer 9)
def divergence_rate(study, opt, df) -> float
    # share of ALL trials with status diverged, or final loss > that optimizer's best
    # final loss in the same study + 0.5 nats (analysis-time only; never stops a run —
    # Gate-0 answer 10)
def seed_rule(paired_diffs: np.ndarray, stds: tuple[float, float]) -> bool
    # all three paired seed differences same sign AND |mean| > 2 * max(stds);
    # a heuristic, labeled as such in every table it feeds
def h3_interaction(df) -> pd.DataFrame
    # per variant: mean over seeds of (gap on MoE - gap on dense), gap = L_var - L_adamw
    # paired by seed; "real" iff |mean| > 2 * its seed std
def null_result_rows(df) -> pd.DataFrame         # "indistinguishable after equal tuning"
    # entries with the measured seed spread, for the table and REPORT_DRAFT.md
def figure_1(df) ... figure_8(df) -> Path        # exactly PLAN's eight figures ->
                                                 # figures/fig<N>_<slug>.png; one fixed
                                                 # colour per optimizer (module constant)
def main_table(df) -> Path                       # tables/main.md: loss mean±std,
                                                 # speedups, basin width, divergence
                                                 # rate; bold only where seed_rule passes
def edge_rule_report(df) -> str                  # how often the edge rule fired
```

### 3.12 `scripts/`

- `prepare_fineweb.py --out data/fineweb --train-tokens 45000000 --val-tokens 2000000`:
  streams `HuggingFaceFW/fineweb-edu` config `sample-10BT`; splits *documents* into
  train/val first; trains byte-level BPE vocab 4096 on train docs only; writes memmaps,
  tokenizer.json, meta.json (version string includes a content hash).
- `prepare_cifar5m.py --out data/cifar5m`: downloads `part0.npz`/`part1.npz`, slices the
  fixed indices of §3.5, writes uint8 arrays + meta.json.
- `oracle_tml.py`: runs the **unmodified** `third_party/manifolds` training script on
  real CIFAR-10, AdamW + Manifold Muon, 3 seeds each; asserts/reports Manifold Muon
  beats AdamW in train and test accuracy (Decisions log oracle check). Output to
  `logs/oracle/`.
- `smoke.py --study S`: 200-step run per optimizer for that study: prints the param map,
  asserts loss decreases, exercises JSONL logging, and records per-step wall time. The
  smoke runs ARE the timing runs (Gate-0 answer 13): steps 21–200 are timed (the first
  20 skipped for warm-up effects), and Manifold Muon is run at cap 10 wherever its cap
  is still undecided.
- `timing.py`: reads the smoke runs' logs on this machine and writes/updates
  `hardware-results.csv` (replaces the 1 TFLOP/s placeholder; gates the Manifold cap-10
  option and scheduling). No extra training runs.

## 4. Run-config schema and hashing

Canonical schema (YAML shown; the dict form is what is hashed). `*` = enters the hash.

```yaml
schema_version: 1          # * bumping deliberately invalidates every hash
study: mlp128              # * mlp128|mlp256|mlp512|dense|moe
phase: stage_a             # * pilot|cd (Phase-I coordinate descent)|stage_a|stage_b|
                           #   edge|fresh_seed|budget_constant|budget_branch|
                           #   budget_halflr|init_ablation|faithful|smoke
arch: mlp                  # * mlp|dense|moe
width: 128                 # * MLP width; null for transformers
optimizer: muon            # * adamw|muon|moonlight|manifold|muonall
rule: keller               # * keller|moonlight|null (scale rule; null for adamw/manifold)
lr: 0.02                   # *
wd: 0.01                   # * the optimizer's OWN group only; null for manifold (no wd
                           #   on constrained matrices). Defaults (plain numbers, Gate-0
                           #   answer 3): adamw 0.1, moonlight 0.1, muon 0.01,
                           #   muonall 0.01; decay applied as lr*wd everywhere
momentum: 0.95             # * Phase-I tunable for the Muon family over {0.9, 0.95, 0.98}
adamw_betas: [0.9, 0.95]   # * Phase-I tunable: beta1 over {0.8, 0.9, 0.95},
                           #   beta2 over {0.95, 0.99, 0.999}
adamw_eps: 1.0e-8          # *
adamw_group_lr: 0.0003     # * AdamW-group lr inside muon AND manifold runs (= AdamW's
                           #   tuned lr for this study); null for adamw/moonlight/muonall
adamw_group_wd: 0.1        # * AdamW-group wd inside muon AND manifold runs (= AdamW's
                           #   tuned wd for this study; Gate-0 answer 12); null otherwise
ns_steps: 5                # * 6 only in the MuonAll faithful rerun
ns_dtype: fp32             # * bf16 only in the MuonAll faithful rerun
manifold:                  # * null unless optimizer == manifold
  dual_cap: 5              #   (100 in the faithful rerun; 10 only if timing allows)
  tol: 1.0e-6
  dual_lr: 0.01
  msign_steps: 10
init_projection: false     # * true for manifold always and for the Muon init ablation
seed: 0                    # *
budget_multiplier: 1.0     # * 0.125|0.25|0.5|1.0|2.0
branch:                    # * null except budget_branch runs:
  parent_hash: "ab12..."   #   hash of the constant run this branch resumes from
  checkpoint_step: 1408    #   step at which it leaves the constant run
budget_base_steps: 3520    # * steps at 1x (3520 MLP / 2441 LM)
budget_unit: images        # * images|tokens
batch: 256                 # * images per step (MLP) or sequences per step (LM: 32)
seq_len: null              # * 256 for LM, null for MLP
grad_clip: 1.0             # * Phase-I tunable over {null (off), 0.5, 1.0}
schedule:                  # *
  kind: wsd
  warmup_frac_of_1x: 0.05  #   warmup steps always absolute from the 1x budget; Phase-I
                           #   tunable over {0, 0.02, 0.05, 0.1}
  decay_frac: 0.2          #   linear decay to 0 over the last 20% of THIS run's budget
precision: fp32            # * fp32|tf32; uniform within a study
router_aux_loss_coef: 0.01 # * MoE only; null otherwise
eval:                      # *
  every_steps: 176         #   176 MLP / 100 LM
  val_small_tokens: null   #   500000 for LM
  val_final: full          #   2M-token set (LM) / 10k test set (MLP)
data:                      # *
  dataset: cifar5m         #   cifar5m|fineweb_edu
  version: "cifar5m-p0p1-v1"   # meta.json version string (content-derived), NOT a path
# ---- logged in the result/log files, NEVER hashed ----
device: mps                # machine-dependent; jobs are round-robined across laptops,
hostname: ...              # so identity must not depend on where a run executes
data_path: /...            # location varies per machine; `data.version` pins content
torch_version: 2.9.x       # pinned by uv.lock; recorded as a cross-check — hashing it
transformers_version: 4.57.6   # would fork run identities on a patch bump
git_commit: ...            # provenance only
started_at / finished_at / wall_clock_s / notes
```

**Why this split.** Hashed fields are exactly those that change what the run *computes*:
two configs agreeing on them must produce statistically identical results, so they may
share a result file (idempotency), and two configs differing in any of them must not
collide (no silent reuse of a stale result after a decision change — `schema_version`
exists to force a global invalidation when semantics change without a field changing).
Every hyperparameter Phase-I coordinate descent can vary — lr, wd, warmup
(`schedule.warmup_frac_of_1x`), momentum, `adamw_betas`, `grad_clip` — is a hashed
field, so each coordinate-descent trial has its own identity and duplicate cells
(e.g. the current value re-appearing in a later sweep) deduplicate automatically.
Logged-only fields vary with *where and when* the run executes: "device" in the
Decisions log means device class + dtype (MPS fp32, CPU fp32, CUDA TF32 — Gate-0
answer 11), which is exactly the hashed `precision` field together with the device
class it implies; the machine itself is logged per run, a study may span several Macs
on the same lockfile, and jobs are round-robined so no optimizer is tied to one
machine. Hashing `device`/`hostname` would give the same experiment different
identities on different laptops, breaking both resume-anywhere and the `hash % 4`
assignment; hashing library versions would do the same on a lockfile refresh. They are still recorded in every result for the
Decisions-log precision/device audit (fp32, one setting per study) — `analysis.py`
asserts uniformity within a study and flags violations into DEVIATIONS.md material.

**Canonicalization.** `runner.canonical` keeps only `HASHED_FIELDS`, sorts keys
recursively, serializes with `json.dumps(..., sort_keys=True, separators=(",", ":"))`,
floats via Python `repr` (shortest round-trip); hash = first 12 hex chars of sha256.
Tests: field order and logged-only fields do not change the hash; every hashed field
does.

## 5. configs/ YAML

One file per study holding only study-level constants and grid centres (run configs are
generated by `studies.py`, not written by hand): arch/width, `budget_base_steps`, batch,
seq_len, eval cadence, data version, `router_aux_loss_coef`, per-optimizer literature
default lr and default wd (plain numbers, Gate-0 answer 3: AdamW 0.1, Moonlight 0.1,
Muon 0.01, MuonAll 0.01), the per-optimizer Stage-A grid centre, and for Phase-II
studies the donor study (`mlp128` for the widths, `dense` for the MoE). These values are
copied into each generated run config (and therefore hashed); the YAML file itself is
not hashed.

## 6. Runner phase/gating DAG (per study)

Project-wide gate (once, before any study): environment + unit tests green, data files
built, `oracle_tml.py` passed, smoke/timing runs done (`smoke.py` executes the 200-step
runs, which ARE the timing runs — Gate-0 answer 13).

**Phase-I studies (mlp128, dense) — coordinate descent:**

```
pilot (adamw default, seeds 101-103; mlp128, mlp512, dense)
  ├─> sets the SINGLE n_final_seeds ∈ {3, 5} for ALL five studies
  └─> sets this study's cd_accept threshold (2 × pilot seed std)   [gates Phase I]

adamw_cd:  lr(6+edge) ─> wd(4) ─> warmup(4) ─> β1(3) ─> β2(3) ─> clip(3) ─> pass 2?
                │  (each sweep is a Phase needing the previous; ≤2 passes, early stop)
                v
     [adamw Phase-I values = adamw best]                 ──> adamw_fresh (seeds 1-3|1-5)
                │                                                  │
                ├──> muon_cd:     lr ─> wd ─> warmup ─> momentum ─> clip ─> ...
                │        └─> muon_fresh ─> init_ablation, muon_faithful (bf16 NS)
                └──> manifold_cd: lr ─> warmup ─> momentum ─> clip ─> ...   (no wd)
                         └─> manifold_fresh ─> manifold_faithful (100 dual steps)
                v
     budget_constant (2x, curve seeds) ─> budget_branches ─> budget_halflr (seed 1
       first; promoted to every curve seed only if it beats seed 1's branched 2x)

moonlight_cd: lr ─> wd ─> warmup ─> momentum ─> clip ─> ... ─> moonlight_fresh
muonall_cd:   lr ─> wd ─> warmup ─> momentum ─> clip ─> ... ─> muonall_fresh
                                                                 └─> muonall_faithful
(moonlight/muonall have no AdamW group, so their descents run parallel to AdamW's;
 muon/manifold wait for AdamW's Phase-I values — AdamW goes first.)
```

**Phase-II studies (mlp256, mlp512, moe) — 12-trial lr × wd recipe:**

```
[donor Phase I complete: mlp128 for widths, dense for moe]  ──┐  (inherited warmup,
pilot (mlp512 only, seeds 101-103)                            │   momentum, betas, clip
                                                              v   per optimizer)
adamw_stage_a (6) ─> adamw_edge (≤2, lr only) ─> adamw_stage_b (6: 2 lrs × wd
     {0, 0.3d, 3d}) ─> adamw_fresh ─> budget_constant ─> branches ─> halflr
          └─> [adamw best lr/wd] ──> muon_stage_a ─> edge ─> muon_stage_b ─> muon_fresh
          │                             └─> init_ablation, muon_faithful
          └──────────────────────> manifold_stage_a ─> edge ─> manifold_stage_b
                                        (quarter-octave lrs) ─> manifold_fresh
                                                                 └─> manifold_faithful
moonlight_stage_a ─> edge ─> moonlight_stage_b ─> moonlight_fresh      (parallel)
muonall_stage_a ─> edge ─> muonall_stage_b ─> muonall_fresh ─> muonall_faithful

analysis gate (all studies): fresh-seed phases + budget curve complete ─> figures/tables.
```

Rules encoded in `studies.build_dag`:

- **Pilot gates Phase I**: `cd_accept` needs the pilot seed std, so no coordinate
  descent starts before the study's pilot is done (dense and mlp128 are both piloted).
  The pilot also fixes one final-seed count {3, 5} applied to all five studies; if 5,
  fresh seeds AND the budget curve use seeds 1–5 (Gate-0 answer 6).
- **Muon and Manifold wait on AdamW's Phase-I values** (Phase-I studies) or on
  `adamw_stage_b` + edge (Phase-II studies): their `adamw_group_lr`/`adamw_group_wd`
  are AdamW's tuned values. Moonlight and MuonAll never wait.
- **Phase-II inheritance**: every Phase-II config carries the donor study's Phase-I
  non-lr/wd values for its optimizer (hashed), so a Phase-II study cannot start before
  its donor's Phase I ends.
- **Edge rule**: LR grids only (Gate-0 answer 5), in both phases; its own lazy phase,
  ≤2 configs beyond the edge, empty list = complete. Every result records
  `edge_rule_fired`; `analysis.edge_rule_report` reports frequency, and
  `studies.trial_counts` reports trials used per optimizer.
- **Selection** (`best_config`): Phase-I studies use the coordinate-descent result
  directly; Phase-II studies take min seed-0 final val loss over stages A, B, edge.
  Fresh seeds are the only reported numbers.
- **Budget curve** needs AdamW's best config and runs per curve seed: one constant-LR
  run to 2×, three decay branches, the ordinary fresh-seed 1× run, plus the half-LR 2×
  run on seed 1 first (promoted to all curve seeds only if it beats seed 1's branched
  2× loss — Gate-0 answer 4).
- **Faithful reruns are three per study, all seed 1** (Gate-0 answer 2): Muon with bf16
  NS (covers Moonlight), MuonAll 6 steps bf16, Manifold 100 dual steps.
- The optional equal-compute and 4×-batch checks are not in the DAG; if run, they are
  added as explicit extra phases and noted in DEVIATIONS.md.
- Cut order under time pressure (PLAN): drop `manifold_*` from `moe` first, then from
  `mlp512`; the DAG supports removal of a whole optimizer chain without touching others.
- "Shared training settings … not tuned" now reads: fixed at the Phase-I-selected
  values within each study, identical across all of that study's budgets and seeds.

The runner executes any ready config whose `hash % n_laptops == laptop_index`; since the
hash ignores device fields, all four laptops compute identical DAGs and partition the
work deterministically. Crashes lose at most the in-flight run.

## 7. WSD schedule and budget-curve branching

Budgets (base 1×): MLP 3,520 steps × 256 images = 901,120 images; LM 2,441 steps × 8,192
tokens ≈ 20M tokens. Warmup is **always** an absolute step count from the 1× budget
(`round(w × base_1x_steps)`; 176 MLP / 122 LM at the 5% default), identical for every
branch and the 2× runs, so all budget points share warmup. Note: PLAN's budget-curve
rows fix w = 0.05, while Phase I can now tune AdamW's warmup fraction — see the §9 note;
the design keeps w as a hashed config field either way.

For a run with `budget_multiplier` m: `total = round(m × base)`; lr is constant (after
warmup) until `0.8 × total`, then decays linearly to 0 at `total`.

The constant run (`budget_constant`, m = 2) follows the same rule — its decay starts at
`round(0.8 × round(2 × base))` (MLP 5,632; LM 3,906), after every checkpoint — and saves
checkpoints at `round(0.8 × round(m' × base))` for m' ∈ {0.125, 0.25, 0.5}
(Clarification 2; Python `round()`):

| m' | MLP branch steps / branch point / decay | LM branch steps / branch point / decay |
| --- | --- | --- |
| 0.125 | 440 / 352 / 88 | 305 / 244 / 61 |
| 0.25  | 880 / 704 / 176 | 610 / 488 / 122 |
| 0.5   | 1760 / 1408 / 352 | 1220 / 976 / 244 |

Each `budget_branch` run loads the checkpoint (model, optimizer state, data position,
RNG) and runs the remaining steps of its budget under linear decay, evaluating on the
regular eval grid plus a final full evaluation at its own end (Clarification 2). Because warmup is fixed from 1× and both schedules are
constant between warmup and the branch point, the branch's lr history is identical to
the constant run's up to the checkpoint (property test:
`tests/test_schedule.py::test_branch_prefix_equivalence`), and because data order is a
seed-determined permutation of the 2× pool of which every budget takes a prefix (§3.5),
the branch also sees exactly the data the constant run saw — same data order, same seed,
as the Decisions log requires. The five fit points per seed are then: three branches,
the ordinary 1× fresh-seed run (same seed, same data prefix, decay over its last 20%),
and the constant run's own 2× decay; the separate half-LR 2× run guards the 2× point
(wd unchanged — because PyTorch applies lr·wd, its effective decay per step also halves;
Clarification 5). The half-LR run executes on seed 1 first: if it beats seed 1's
branched 2× loss, it is run on every curve seed and used in every fit; otherwise the
branched 2× point stands everywhere (Gate-0 answer 4). "Seeds 1–3" throughout this
section becomes "seeds 1–5" if the pilot chose 5 final seeds (Gate-0 answer 6).

Sanity inequality: warmup (0.05×base) < smallest branch point (0.8 × 0.125 × base =
0.1×base) — the checkpoint is never inside warmup.

## 8. Test suite (planned)

| File | Covers |
| --- | --- |
| `tests/test_ns.py` | `ns_vec` ≡ `diag(ns(diag(v)))`; `MuonFamily(rule="keller"/"moonlight")` one step ≡ `torch.optim.Muon` (`adjust_lr_fn="original"/"match_rms_adamw"`); batched `ns` over stacked experts ≡ per-matrix loop; fp32/5-step defaults |
| `tests/test_scale.py` | scale anchors 11.09 / 2.26 / keller square = 1 |
| `tests/test_param_groups.py` | map complete + disjoint; dense muon fraction ≈ 69%; MLP > 99%; MoE 16×3 separate expert matrices; router placement per optimizer |
| `tests/test_manifold.py` | retraction keeps WᵀW ≈ I (Stiefel stay); phase-1 equivalence: adapted `ManifoldMuon` with zero momentum and reference cap reproduces the unmodified `third_party` update; batched experts ≡ per-matrix; cap respected + steps-taken logged; vendored `msign.py` identical to `third_party` |
| `tests/test_models.py` | dense = 1.70M params; MoE unfused in 4.57.6; seed reproducibility of init |
| `tests/test_flops.py` | analytic vs `FlopCounterMode` within 5%; NS formula; Manifold FLOPs use logged dual steps |
| `tests/test_schedule.py` | branch prefix equivalence; warmup fixed to 1×; decay over last 20% to exactly 0; checkpoint-step arithmetic |
| `tests/test_data.py` | CIFAR train/val/test disjoint and sliced at C[1,802,240:1,822,240] of part0‖part1; single pass no repeats; seed sets order; 1× order is prefix of 2× order; FineWeb counts/dtype (≥45M, 2M, uint16, vocab 4096 incl. EOS); one EOS after each document; tokenizer trained on train docs only (meta flag) |
| `tests/test_runner.py` | hash stable under key order; logged-only fields don't change hash; every hashed field does; skip-if-done; `hash % 4` partition |
| `tests/test_train.py` | divergence detection (theoretical thresholds, train OR val loss, any logged step) stops and records; the +0.5-nat criterion never stops a run; grad clip applied (and off when null); result written only on completion; checkpoint round-trip resumes bit-exact data order |
| `tests/test_studies.py` | DAG gating (pilot gates Phase I; muon/manifold wait on AdamW; moonlight/muonall don't; Phase-II studies wait on their donor's Phase I); coordinate descent: sweep order, acceptance only above 2× pilot std, ≤2 passes with early stop, manifold skips wd; Phase-II inheritance of non-lr/wd values; Stage-B wd set {0, 0.3d, 3d}; edge rule lr-only; grids share positions across widths; pilot seeds 101–103; half-LR seed-1-first promotion; faithful = 3 configs (Muon bf16, MuonAll 6/bf16, Manifold 100); stage/edge/fresh counts |
| `tests/test_analysis.py` | curve fit inverts correctly on synthetic data; speedup reported as range over 3×3; seed rule (sign + 2·std); basin width; divergence rate; null-result rows carry seed spread |

## 9. Clarifications round 2 (Gate 0, approved by Paul)

All thirteen open questions from the first draft of this document were answered by the
project owner at Gate 0; none remain open. Recorded here and applied throughout
(§3.1, §3.5, §3.10, §4, §5, §6, §7, §8). The two-phase tuning decision these answers
accompany is now PLAN.md's "Hyperparameter search" Decisions-log row.

1. **Image counts confirmed**: training = the first 1,802,240 images of parts 0 and 1
   concatenated; the 1× budget = the first 901,120; validation = the next 10k; test =
   the 10k after that.
2. **Faithful reruns are three**: Muon with bf16 Newton–Schulz (as in Keller's repo;
   covers Moonlight too, since they share the routine and Moonlight's scaling is
   already faithful), MuonAll at 6 steps bf16, Manifold at 100 dual steps. All seed 1.
3. **Default wd per optimizer, plain numbers, not scaled**: AdamW 0.1, Moonlight 0.1,
   Muon 0.01, MuonAll 0.01. Decay applied as lr·wd everywhere.
4. **Half-LR 2× run**: seed 1 first. If it beats seed 1's branched 2× loss, run it on
   every curve seed and use it in every fit; otherwise keep the branched 2× everywhere.
5. **Edge rule**: learning-rate grids only. Stage B wds are {0, 0.3×default,
   3×default} (the default already ran in Stage A).
6. **Pilot**: seeds 101–103. One seed count applies to all studies, decided from the
   piloted ones; if 5, fresh seeds AND the budget curve use seeds 1–5.
7. **FineWeb**: one `<|endoftext|>` token after each document, inside the 4,096 vocab.
   Non-overlapping 257-token windows (256 predictions each), seed-shuffled order.
8. **0.5M periodic eval set** = the first 0.5M tokens of the 2M validation set.
9. **MLP LR-sensitivity 1%** = relative final validation loss.
10. **Divergence**: NaN, or train OR validation loss above ln 4096 / ln 10
    (theoretical) at any logged step → stops the run. The "+0.5 nats above best"
    criterion is analysis-time only (same optimizer's best in the same study) and
    never stops a run.
11. **"Device"** = device class + dtype (MPS fp32, CPU fp32, CUDA TF32). A study may
    span several Macs on the same lockfile. Log the machine per run; round-robin so no
    optimizer is tied to one machine.
12. **AdamW group in two-group runs**: in Muon AND Manifold runs the AdamW group uses
    AdamW's tuned lr AND wd for that study. Muon's tuned wd applies to the Muon group
    only. Moonlight shares one lr and one wd across both groups by design.
13. **Smoke runs ARE the timing runs**: the same 200-step runs; time steps 21–200
    (skip the first 20 for warm-up effects); include Manifold Muon at cap 10 wherever
    its cap is still undecided.

### Noted discrepancies — PLAN.md lines to be fixed (not resolved here)

- **Stage-B wd set** (PLAN.md:249): Phase II step 2 still says {0, default,
  3× default}; Paul's Gate-0 answer 5 sets {0, 0.3×default, 3×default}. The Gate-0
  message wins; this design implements {0, 0.3×default, 3×default}.
- **"Same wd on both groups" for Muon** (PLAN.md:219): superseded by Gate-0 answer 12
  (Muon's tuned wd on the Muon group only; AdamW group uses AdamW's tuned lr and wd).
- **Budget-curve warmup "fixed at 5% of the 1× budget"** (PLAN.md:77, 257, 266) vs the
  Phase-I warmup sweep {0, 2%, 5%, 10%} (PLAN.md:239): if AdamW's Phase-I warmup is
  not 5%, the curve's warmup should presumably be AdamW's tuned fraction (still
  absolute in 1×-budget steps); the design keeps the fraction a hashed config field so
  either reading is a config value, but PLAN.md should state which.
- **Val/test provenance** (PLAN.md:68 "Validation and test images come from the same
  part", PLAN.md:136 "Part 0 gives 900k training images plus fixed 10k validation and
  10k test sets"): Gate-0 answer 1 places val/test AFTER the full 2× train block of
  the part0‖part1 concatenation, i.e. inside part 1, not part 0.
- **"Not tuned" shared settings** (PLAN.md:257 schedule "not tuned"; PLAN.md:258
  "clipping at 1.0 for all"): superseded by the Phase-I sweeps of warmup and clipping;
  now "fixed at Phase-I-selected values within each study".
- **Protocol intro** (PLAN.md:212 "Every optimizer gets 12 tuning trials on seed 0")
  and the run-count table (PLAN.md:354 "5 optimizers × 12 trials"): true only for
  Phase-II studies; Phase-I studies (mlp128, dense) run coordinate descent instead
  (≈19–40 trials per optimizer depending on passes and dedup), and the compute rows
  were updated (+6–10 h) but the per-study run table was not.
- **Pilot seed wording** (PLAN.md:41 "3 seeds"): Gate-0 answer 6 fixes them as
  101–103, distinct from tuning seed 0 and fresh seeds 1–3/1–5; PLAN.md does not say
  this.
