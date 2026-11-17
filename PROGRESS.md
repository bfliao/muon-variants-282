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
- Gate 1 passed (commit tagged gate-1). Phase 2 underway.

## Phase 2 module status
- APPROVED: msign/manifold (+msign_calls), optim core (torch-oracle-verified),
  schedule+flops (FlopCounterMode-exact), data (re-review after 2 fixes:
  fineweb resume reconciliation, fixed 156,224-window pool).
- Datasets built: fineweb (45.0M train / 2M val tokens, v fineweb-edu-10bt-v1-5d037df9),
  cifar5m (1,802,240/10k/10k, v cifar5m-v1-b7ab24de). Disk note: machine at ~98%,
  9.8 GiB free after deleting the re-downloadable .npz archives.
- APPROVED (cont.): models + param groups (dense 1,706,368 params, muon 69.13%;
  MLP >99%; MoE unfused, router=mlp.gate; membership verified by exhaustive
  enumeration, 15/15 pairs); analysis (after 7 review fixes: tuning-only
  divergence rate, Phase-I "cd" support, overflow-safe inversion, H6/H3 edges,
  contiguous basin). Note for studies.py: always emit the cd_sweep config field.
- APPROVED (cont.): train.py + run-identity layer (after 2 fixes: unique-tmp atomic
  writes with corrupt-result recovery, hashed schedule fields honored; plus
  map_location hardening); runner/studies/configs/smoke+timing (after 3 fixes:
  manifold cap+1 scaling, smoke per-unit failure isolation, exclude_optimizers
  cut lever).
- Gate-2 consolidated checklist audit: every line ticked (18 exact, 54 name-drift
  corrected in place, 1 deferred to Phase 6, 3 flags all closed: script tests,
  edge_trial observability, trial_counts test). All 14 known traps verified
  neutralized with file:line evidence.
- Gate 2 report sent; waiting for go-ahead to Phase 3 (smoke + timing runs).

## 2026-11-17 — Phase 3 launched
- Gate 2 go-ahead received. Team briefing PDF at docs/team_briefing.pdf; repo pushed to a private GitHub repo (tags gate-1, gate-2).
- Phase 3 smoke+timing (scripts/phase3.sh) running under caffeinate: all 5 studies, every optimizer, Manifold caps 5 and 10; then tables/compute_estimate.md.
- Gate-3 prep: profiled Manifold Muon (per-matrix dual-ascent bookkeeping was 81-84% of the MoE optimizer step); vectorized it (DEVIATIONS #9, bit-identical, 76 new tests, review APPROVE; suite 393 passed); re-timed transformers per step (DEVIATIONS #10). MoE timings contaminated by a concurrent Zoom call (fwd+bwd varied 160-1130 ms for identical work): re-time on an idle machine before Gate 3.
- MLP protocol change (DEVIATIONS #12): full Phase-I CD at MLP-256/512 (headline) + inherited Phase-II track (seed 0, H4); MLP-256 pilot added. studies.py + analysis.py track separation; review FIX NEEDED (analysis mixed tracks) -> fixed -> APPROVE. Suite 407 passed. MLP-128/dense/MoE job hashes unchanged vs pre-change snapshot. Pilots: mlp128 std 0.0053, dense std 0.0291. MLP-256/512 NOT started pending Paul confirmation.
