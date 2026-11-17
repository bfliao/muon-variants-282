# Progress log

## 2026-11-16 — Phase 0 complete, Phase 1 underway
- Gate 0 passed: docs/design.md + docs/decisions_checklist.md written and reviewed;
  Paul answered 6 + 13 clarification questions (recorded in design.md); PLAN.md
  updated by Paul with two-phase tuning (Hyperparameter search row).
- Environment: uv venv, Python 3.12.9 (3.11 broke on vendored f-strings → pinned 3.12),
  torch 2.15.0, transformers 4.57.6, MPS available, torch.optim.Muon present.
- Vendored thinking-machines-lab/manifolds @ 89dcae50f01af59f1e0570289474da3a2ecaa60b
  byte-for-byte into third_party/manifolds/ (verified with cmp; sha256 in VENDORED_COMMIT.txt).
- Phase 1 oracle Stage 1 (seed 42: AdamW lr {3e-4,1e-3,3e-3,1e-2} wd 0; Manifold
  lr {0.05,0.1,0.2}; 3 epochs, batch 1024, 100 dual steps) running on MPS under caffeinate.
- muonarch/msign.py + manifold.py + equivalence tests being written (subagent).
- Oracle Stage 1 done (MPS, bf16 vendored msign): AdamW best lr 3e-3 (test 48.60%),
  Manifold best lr 0.1 (test 51.90%); Manifold first-layer singular values within
  ±0.003 of 1 at every LR. ~0.04 s/step AdamW, ~0.45 s/step Manifold (100 dual steps).
- muonarch/msign.py + manifold.py written, 24 tests green; independent review:
  APPROVE (bf16 path bit-identical to vendored; probes all bit-exact).
- Oracle Stage 2 done: over seeds 42-44, Manifold Muon (lr 0.1) train 61.64 ± 0.20 /
  test 51.73 ± 0.53 beats AdamW (lr 3e-3) train 52.00 ± 0.49 / test 48.66 ± 0.30;
  singular values on-manifold every seed. Phase 1 pass criteria all met.
- Gate 1 report sent; waiting for go-ahead to Phase 2.
