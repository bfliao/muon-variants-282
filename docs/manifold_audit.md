# Manifold Muon audit (after the Gate-3 sweeps, 2026-11-17)

**Question.** In the frozen-protocol sweeps, Manifold Muon has the worst final validation
loss at every MLP width and on dense (MLP-128 0.914 vs AdamW 0.888 vs Muon 0.807).
Is that a bug, a tuning artefact, or a property of the method?

**Answer.** No bug found. The deficit comes from the Stiefel constraint (all singular values = 1)
in our regime. Part of it is a protocol interaction: Manifold Muon, unlike Muon, gains from a
larger learning rate on the unconstrained parameters (classifier and biases). Nothing in
`results/` or the frozen protocol was changed. All runs below are diagnostic only.

## 1. The method itself was already validated (Phase 1 oracle, PROGRESS.md)

The unmodified Thinking Machines script ran on real CIFAR-10. Settings: bias-free MLP
3072-128-64-10, batch 1024, 3 epochs, 100 dual steps, bf16 vendored msign, seeds 42–44.

| Optimizer | Train acc | Test acc |
|---|---|---|
| Manifold Muon, lr 0.1 | 61.64 ± 0.20 | 51.73 ± 0.53 |
| AdamW, lr 3e-3 | 52.00 ± 0.49 | 48.66 ± 0.30 |

The blog reports its numbers only in a figure (`svgs/cifar10.svg`: 3 epochs, 3 seeds, best
learning rate per optimizer). We read its peak values out of the figure's vector data and put
them next to ours:

| | Blog train | Ours train | Blog test | Ours test |
|---|---|---|---|---|
| Manifold Muon | 62.8 (lr 0.125) | 61.64 ± 0.20 (lr 0.1) | 52.2 | 51.73 ± 0.53 |
| AdamW | 51.8 (lr ≈2e-3) | 52.00 ± 0.49 (lr 3e-3) | 49.1 | 48.66 ± 0.30 |

Every number is within about 1 point of the blog's. The gaps are of the order of the seed std,
plus a different learning-rate grid point (0.1 vs 0.125) and different hardware (MPS vs CUDA).
So this is a quantitative replication, not just the same direction. It is not bit-exact, which
nobody could expect across GPUs. The weights stayed on the manifold for every seed. Our
`muonarch/manifold.py` matches the vendored update bit-for-bit in bf16 (tests in
`tests/test_manifold_equivalence.py`).

## 2. Re-check with our implementation (`scripts/diag_ref_regime.py`)

This uses the reference regime: their MLP, batch 1024, 5 epochs, a linear-decay learning rate,
all layers constrained, momentum 0 and 100 dual steps. It runs on a 50k-image CIFAR-5m subset
in place of CIFAR-10, with one seed (42).

| Optimizer | Test acc | Train acc |
|---|---|---|
| Manifold Muon, lr 0.1 | 59.66 | 72.33 |
| Manifold Muon, lr 0.1, cap 5 | 59.65 | 71.92 |
| Manifold Muon, lr 0.1, momentum 0.95 | 60.05 | 71.41 |
| AdamW, best (lr 3e-3, wd 0.01) | 58.02 | 64.11 |
| Muon (Keller), lr 0.03 | 60.50 | 80.40 |

In this regime Manifold Muon beats AdamW again, so our implementation behaves like the reference.

## 3. Why it loses in our protocol: MLP-128 ablations, seed 0 (`scripts/diag_manifold.py`)

Each row is one run of 3520 steps, starting from the MLP-128 winning configs. The seed std of
final validation loss is about 0.005.

| Run | Val loss |
|---|---|
| Manifold Muon, frozen-protocol config (cap 5) | 0.9156 |
| Cap 0 (Muon step, then project back to the manifold) | 0.9286 |
| Cap 100 (the faithful rerun, seed 1) | 0.9141 |
| Momentum 0 (the reference setting) | 1.0332 |
| Fixed forward scale ×0.35 / ×2.8 / ×8 on the constrained layers | 1.1911 / 0.9433 / diverged |
| Classifier and bias weight decay 0 | 0.9085 |
| Classifier and bias lr ×4 / ×16 / ×64 | 0.9004 / **0.8940** / 0.9049 |
| Classifier and bias lr ×16, weight decay 0 | 0.8942 |
| **Muon**, frozen-protocol config | 0.7981 |
| **Muon**, classifier and bias lr ×16 | 0.8058 |

Seeds 1–3 from the sweeps add two more reference points:
- Muon started from a Stiefel-projected init and then left unconstrained (init ablation): 0.827,
  against 0.807 for Muon.
- The best learning rate is inside the grid at every size, for example MLP-128 lr
  0.0016 / 0.0031 / 0.0063 gives 0.939 / 0.916 / 0.919.

## 4. Conclusions

1. **Not a bug.** The update matches the reference, the weights stay on the manifold (all final
   spectral norms are 1.000), the method beats AdamW in its own regime, and a replica of the
   sweep run reproduces 0.91558 exactly.
2. **Not under-tuned dual ascent.** Caps of 0, 5 and 100 land within 0.015 of each other.
   Momentum, our only addition, helps.
3. **The constraint is the cost.** Unconstrained Muon's trained matrices have spectral norms of
   2.6–2.9 (MLP-128) and 1.6–12.3 (dense). A unit-norm init costs Muon only 0.02, but holding the
   constraint costs about 0.11. No fixed rescale helps.
4. **The classifier-rate sensitivity is specific to Manifold Muon.** With the classifier and bias
   lr ×16, Manifold Muon improves by 0.022 (4× the seed std) while Muon gets slightly worse
   (+0.008). The likely reason is that the constrained matrices cannot grow, so the scale has to
   come from the unconstrained parameters. The frozen protocol pins that rate to AdamW's tuned lr
   for both optimizers. Even at its best, Manifold Muon would only roughly tie AdamW on MLP-128
   (0.894 vs 0.888) and stay far behind Muon.
5. **Dense:** the Manifold Muon vs AdamW gap (4.067 vs 4.037) is within AdamW's seed noise
   (std 0.027).

**Report wording.** State the result as measured under the frozen protocol, and give point 4 as
a limitation: the classifier lr is shared by design and is not re-tuned per optimizer.

## Reproducing

```
uv run python scripts/diag_manifold.py      # writes logs/diagnostics/manifold_audit/summary.jsonl
uv run python scripts/diag_ref_regime.py    # prints JSON rows; ~6 min on MPS
```

Neither script writes to `results/`. Both use MPS, and the first holds the 5.3 GB CIFAR-5m train
set in memory, so don't run it alongside two other MLP runners.
