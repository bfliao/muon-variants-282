# Decisions checklist — PLAN.md → implementation → test

One line per Decisions-log row and per Protocol row/requirement. File/function names
follow `docs/design.md`; tests are the planned suite (they may not exist yet). Paul's
Phase-0 clarifications (PLAN.md rows 75–81) and Gate-0 answers (design.md §9) are folded
into the relevant lines.

## Decisions log

- [ ] Fixed budget per study, schedule ends at budget, only divergence (NaN, or train/val loss above the theoretical initial loss at any logged step) stops early → impl: muonarch/train.py:run + muonarch/schedule.py:make_schedule → test: tests/test_train.py::test_divergence_detection_stops_and_records
- [ ] Tokens-to-target from fitted AdamW curve: branches 0.125/0.25/0.5× off one constant 2× run, 1× fresh run, half-LR 2× guard, per-seed fit, speedup as range → impl: muonarch/studies.py:budget_curve_configs + muonarch/analysis.py:fit_budget_curve → test: tests/test_analysis.py::test_fit_inversion_recovers_speedup
- [ ] Equal-compute check optional, 3 seeds, reported separately → impl: muonarch/studies.py:build_dag (extra phase, off by default) → test: tests/test_studies.py::test_optional_phases_not_scheduled
- [ ] Scope: claims only about ≈5M-parameter models → impl: REPORT_DRAFT.md (scope paragraph) → test: tests/test_report.py::test_scope_claim_present
- [ ] Study A and B reported separately, no cross-study ranking → impl: muonarch/analysis.py:main_table (per-study sections) → test: tests/test_analysis.py::test_no_cross_study_ranking
- [ ] Null results valid, reported with measured seed spread; AdamW pilot first (seeds 101–103, mlp128/mlp512/dense) sets ONE 3-vs-5 final-seed count for all studies → impl: muonarch/analysis.py:null_result_rows + muonarch/studies.py:pilot_configs → test: tests/test_analysis.py::test_null_result_rows_carry_seed_spread
- [ ] H2a reported as break-even prediction confirmed, not a discovery → impl: muonarch/flops.py:BREAK_EVEN + REPORT_DRAFT.md → test: tests/test_flops.py::test_break_even_ratio_values
- [ ] One shared NS routine: 5 iterations, Keller coefficients (3.4445, −4.775, 2.0315), fp32; MuonAll at 5 steps not 6 → impl: muonarch/optim.py:ns → test: tests/test_ns.py::test_ns_fp32_and_5_steps_default
- [ ] msign for Manifold Muon: Polar Express, 10 iterations, from TML msign.py, for dual ascent and retraction → impl: muonarch/msign.py:msign → test: tests/test_manifold_equivalence.py::test_abc_coefficients_match_vendored + test_msign_bf16_matches_vendored_bitwise
- [ ] Manifold settings: cap 5 (10 only if timed run fits), tol 1e-6, dual step 0.01 linearly decaying, shared Nesterov buffer, batched same-shape experts, steps taken logged → impl: muonarch/manifold.py:ManifoldMuon → test: tests/test_manifold_equivalence.py (cap/steps-taken tests, class TestCapAndLogging)
- [ ] Faithful-settings reruns: THREE per study, all seed 1 — Muon with bf16 NS (covers Moonlight), MuonAll 6 steps bf16, Manifold 100 dual steps (MLP-512 only if time) → impl: muonarch/studies.py:faithful_configs → test: tests/test_studies.py::test_faithful_configs_official_settings
- [ ] fp32 on laptops; one precision setting per study ("device" = class + dtype: MPS fp32 / CPU fp32 / CUDA TF32); machine logged per run, round-robin so no optimizer is tied to one machine → impl: muonarch/runner.py:HASHED_FIELDS (precision hashed, machine logged-only) + muonarch/train.py:run → test: tests/test_runner.py::test_precision_hashed_device_not
- [ ] transformers pinned to 4.57.6; MoE experts separate nn.Linear in nn.ModuleList → impl: muonarch/models.py:assert_moe_unfused (+ uv.lock) → test: tests/test_models.py::test_moe_unfused_in_pinned_transformers
- [ ] Compute platform: laptops + optional RTX 4090, verified by the timed smoke runs (same 200-step runs, steps 21–200 timed, Manifold at cap 10 where undecided) → impl: scripts/smoke.py + scripts/timing.py:main → test: tests/test_scripts.py::test_timing_writes_hardware_csv
- [ ] MLP widths 128, 256, 512; width 1024 dropped → impl: muonarch/studies.py:STUDIES + configs/mlp{128,256,512}.yaml → test: tests/test_studies.py::test_studies_and_widths
- [ ] Study A data: CIFAR-5m one pass, train = first 1,802,240 of part0‖part1 (1× = first 901,120), val/test = the next 10k + 10k, never trained on, no augmentation → impl: muonarch/data.py:cifar_batches + scripts/prepare_cifar5m.py → test: tests/test_data.py::test_no_image_repeats_single_pass
- [ ] H5 correction: Muon vs Moonlight empirical on MLP; input layer gets 0.2·√3072 ≈ 11.1 vs 2.3–4.5 → impl: muonarch/optim.py:scale + muonarch/analysis.py:figure_5 → test: tests/test_scale.py::test_scale_values
- [ ] H3 as interaction: per-variant seed-paired (MoE gap − dense gap), real only beyond 2× its seed std → impl: muonarch/analysis.py:h3_interaction → test: tests/test_analysis.py::test_h3_interaction_pairing
- [ ] Headlines H4 (A) / H3 (B); cut order: Manifold off MoE first, then MLP-512, keep the MoE → impl: muonarch/studies.py:build_dag (whole optimizer chains removable) + DEVIATIONS.md → test: tests/test_studies.py::test_optimizer_chain_removable
- [ ] Manifold Muon oracle: unmodified TML script on real CIFAR-10, AdamW + Manifold, 3 seeds, Manifold wins train+test accuracy → impl: scripts/oracle_tml.py:main (uses third_party/manifolds unmodified) → test: tests/test_manifold_equivalence.py (gate equivalence tests) + scripts/oracle_report.py output
- [ ] Changes from the proposal told to GSI before running → impl: DEVIATIONS.md → test: tests/test_report.py::test_proposal_changes_listed
- [ ] Hyperparameter search: two phases — Phase I coordinate descent (MLP-128, dense; ≤2 passes; acceptance > 2× pilot seed std), Phase II inherits non-lr/wd values and runs the 12-trial lr×wd recipe (MLP-256, MLP-512, MoE) → impl: muonarch/studies.py:cd_next + inherited_values → test: tests/test_studies.py::test_cd_acceptance_threshold
- [ ] Phase 0 clarifications (umbrella row: the six items below, answered at Phase 0) → impl: docs/design.md:"Clarifications (approved by Paul)" → test: tests/test_report.py::test_design_doc_clarifications_present
- [ ] MLP activation and bias: ReLU, biases on (default nn.Linear/init); biases in AdamW group except under MuonAll; bias-free MLP only for the TML oracle → impl: muonarch/models.py:build_mlp + muonarch/optim.py:split_params → test: tests/test_models.py::test_mlp_relu_biases
- [ ] Budget-curve step rule: b× budget = round(b×S) steps, branch leaves at round(0.8 × that) exactly (eval-grid independent), full final eval at branch end, warmup fixed at round(0.05×S) = 122/176 absolute → impl: muonarch/schedule.py:branch_steps/checkpoint_step/warmup_steps → test: tests/test_schedule.py::test_checkpoint_step_arithmetic
- [ ] Manifold Muon Stage B grid: best Stage-A lr × 2^k, k ∈ {±0.25, ±0.5, ±0.75}, edge rule applied first → impl: muonarch/studies.py:stage_b_grid → test: tests/test_studies.py::test_stage_b_grid_rules
- [ ] MuonAll tied embedding: one 4,096×128 tensor, counted once, NS with Keller scale ≈5.66, wd applies, NS FLOPs counted → impl: muonarch/optim.py:split_params("muonall") + muonarch/flops.py:optimizer_flops_per_step → test: tests/test_param_groups.py::test_muonall_tied_embedding_once
- [ ] Half-LR 2× run keeps wd unchanged (effective decay lr·wd halves with lr); seed 1 first, promoted to all curve seeds only if it beats seed 1's branched 2× → impl: muonarch/studies.py:budget_curve_configs → test: tests/test_studies.py::test_halflr_keeps_wd_and_promotes
- [ ] Dense vs MoE controls: identical tokenizer/data/order, 20M tokens, batch 32×256, schedule, attention, tied embeddings, eval sets, seeds, clipping; only FF block, router, aux loss (0.01, norm_topk_prob) differ; MoE's larger total size raises Muon FLOPs — counted → impl: muonarch/models.py:build_dense/build_moe + configs/{dense,moe}.yaml → test: tests/test_models.py::test_dense_moe_only_ffn_differs

## Protocol — optimizer / parameter-group table

- [ ] AdamW: everything via AdamW; tuned lr, wd (+ Phase-I β1/β2/warmup/clip); eps 1e-8 → impl: muonarch/optim.py:build_optimizers → test: tests/test_param_groups.py::test_adamw_all_params
- [ ] Muon: hidden matrices (q,k,v,o; MLP/SwiGLU; each expert's gate/up/down separately; MLP input 3072→w and hidden w→w) via Muon; embeddings/tied head, classifier, norms, QK-norm gains, biases, router via AdamW; AdamW group uses AdamW's tuned lr AND wd, Muon's tuned wd on the Muon group only (Gate-0 answer 12) → impl: muonarch/optim.py:split_params + build_optimizers → test: tests/test_param_groups.py::test_map_complete_and_disjoint
- [ ] Moonlight: same split, one lr and one wd shared by both groups, scale 0.2·√max(fan_out, fan_in) → impl: muonarch/optim.py:build_optimizers (rule="moonlight") → test: tests/test_ns.py::test_muonfamily_moonlight_matches_torch_muon_match_rms
- [ ] Manifold: same muon set constrained to Stiefel, projected at step 0, lr-only tuning (no wd on constrained), cap/tol fixed, batched experts, shared Nesterov buffer, AdamW group = AdamW's tuned lr and wd → impl: muonarch/manifold.py:ManifoldMuon.project_ + step → test: tests/test_manifold_equivalence.py (Stiefel-stay test)
- [ ] MuonAll: every parameter — matrices by ns, 1-D (incl. MLP biases) by ns_vec, tied embedding once by ns at scale ≈5.66 with wd; wd grid includes 0 → impl: muonarch/optim.py:split_params("muonall") → test: tests/test_ns.py::test_ns_vec_matches_diag_ns
- [ ] Param→group map in config, printed at startup, fractions reported (69% dense, >99% MLP, separate MoE experts) → impl: muonarch/optim.py:ParamGroups.table/fractions → test: tests/test_param_groups.py::test_muon_fraction_dense_69pct

## Protocol — Newton–Schulz settings table

- [ ] Muon/Moonlight: 5 steps, Keller coefficients, fp32 (not repo bf16) → impl: muonarch/optim.py:ns (defaults) → test: tests/test_ns.py::test_muonfamily_keller_matches_torch_muon_original
- [ ] MuonAll standardised to 5 steps fp32 (official 6/bf16 only in the faithful rerun config) → impl: muonarch/studies.py (default run configs: ns_steps=5, ns_dtype=fp32) → test: tests/test_studies.py::test_default_configs_ns_settings
- [ ] Manifold: Polar Express msign for both dual ascent and retraction, ≤10 dual steps; FLOP count uses iterations actually performed → impl: muonarch/manifold.py:ManifoldMuon(msign_steps=10) + muonarch/flops.py:run_flops → test: tests/test_flops.py::test_manifold_flops_use_logged_dual_steps

## Protocol — tuning, Phase I (coordinate descent, MLP-128 and dense)

- [ ] Start every optimizer at its literature defaults, seed 0 (plain default wd: AdamW 0.1, Moonlight 0.1, Muon 0.01, MuonAll 0.01; lr·wd convention everywhere) → impl: muonarch/studies.py:cd_sweep_configs + configs/*.yaml → test: tests/test_studies.py::test_cd_starts_at_defaults
- [ ] Sweep order lr (6-pt factor-2) → wd {0, 0.3d, d, 3d} → warmup {0, 2%, 5%, 10%} → momentum ({0.9, 0.95, 0.98} Muon family; β1 {0.8, 0.9, 0.95} AdamW) → β2 {0.95, 0.99, 0.999} → clip {off, 0.5, 1.0}; Manifold skips wd → impl: muonarch/studies.py:CD_ORDER/CD_GRIDS → test: tests/test_studies.py::test_cd_sweep_order_and_grids
- [ ] Accept a change only if final val loss improves by > 2× the pilot seed std of that study → impl: muonarch/studies.py:cd_accept → test: tests/test_studies.py::test_cd_acceptance_threshold
- [ ] At most 2 passes, stop early if a pass changes nothing; edge rule on the lr sweep → impl: muonarch/studies.py:cd_next → test: tests/test_studies.py::test_cd_two_passes_early_stop
- [ ] AdamW goes first; AdamW groups inside Muon-family runs use AdamW's Phase-I values → impl: muonarch/studies.py:build_dag + muonarch/optim.py:build_optimizers → test: tests/test_studies.py::test_dag_gating_order
- [ ] Phase-I studies use their Phase-I result directly as the best config; report the number of trials each optimizer used → impl: muonarch/studies.py:phase1_values + trial_counts → test: tests/test_studies.py::test_phase1_result_is_best_config

## Protocol — tuning, Phase II (12 trials; MLP-256, MLP-512, MoE)

- [ ] Inherit each optimizer's non-lr/wd values from MLP-128 (widths) or dense (MoE), carried in every Phase-II config → impl: muonarch/studies.py:inherited_values + PHASE2_DONOR → test: tests/test_studies.py::test_phase2_inheritance
- [ ] Stage A: 6 lrs on a factor-2 log grid spanning 32×, centred on the literature default, at the plain default wd → impl: muonarch/studies.py:stage_a_grid → test: tests/test_studies.py::test_stage_a_grid_span_and_centre
- [ ] Stage B: 2 best lrs × wd {0, 0.3×default, 3×default} (default already run in Stage A — Gate-0 answer 5; PLAN.md:249 to be fixed); Manifold instead best lr × 2^k, k ∈ ±{0.25, 0.5, 0.75} after the edge rule → impl: muonarch/studies.py:stage_b_grid → test: tests/test_studies.py::test_stage_b_grid_rules
- [ ] Edge rule: ≤2 trials beyond a grid edge, learning-rate grids ONLY, applied to every optimizer, frequency reported → impl: muonarch/studies.py:edge_trials + muonarch/analysis.py:edge_rule_report → test: tests/test_studies.py::test_edge_trials_lr_only
- [ ] Select on seed-0 final val loss; rerun winner on fresh seeds 1–3 (1–5 if the pilot chose 5); report those only → impl: muonarch/studies.py:best_config + fresh_seed_configs → test: tests/test_studies.py::test_selection_and_fresh_seeds
- [ ] Same grid positions relative to default at every MLP width (H4 for free) → impl: muonarch/studies.py:stage_a_grid → test: tests/test_studies.py::test_grids_share_positions_across_widths

## Protocol — shared training settings

- [ ] WSD schedule: warmup in absolute steps of the 1× budget (Phase-I-selected fraction, default 5% = 122 LM / 176 MLP), constant, linear decay to 0 over last 20% of each run's budget; fixed at Phase-I-selected values within each study, identical across budgets and optimizer arms of a run's comparison → impl: muonarch/schedule.py:wsd_multiplier → test: tests/test_schedule.py::test_decay_last_20pct_to_zero
- [ ] Budget-curve branching: checkpoint at round(0.8 × round(b×S)) exactly, branch resumes with same data order/seed, evals on grid + at branch end → impl: muonarch/schedule.py:checkpoint_step + muonarch/train.py:save_checkpoint/load_checkpoint → test: tests/test_schedule.py::test_branch_prefix_equivalence
- [ ] Global grad-norm clipping (Phase-I-selected from {off, 0.5, 1.0}, default 1.0, fixed within study); fp32 everywhere → impl: muonarch/train.py:run → test: tests/test_train.py::test_grad_clip_applied
- [ ] Seed s fixes init and data order identically across optimizers (paired comparisons); pilot uses seeds 101–103, tuning seed 0, fresh seeds 1–3/1–5 → impl: muonarch/train.py:run (seed-before-build) + muonarch/data.py iterators → test: tests/test_train.py::test_seed_fixes_init_and_order_across_optimizers
- [ ] Transformer: batch 32×256 = 8,192 tokens, 2,441 steps, eval every 100 steps on the first 0.5M val tokens, final on the full 2M; same sets for everyone → impl: configs/dense.yaml + configs/moe.yaml + muonarch/data.py:lm_val_batches → test: tests/test_studies.py::test_lm_budget_and_eval_settings
- [ ] MLP: batch 256, 3,520 steps single pass, eval every 176 steps, test set only for final table → impl: configs/mlp*.yaml + muonarch/data.py:cifar_batches → test: tests/test_studies.py::test_mlp_budget_and_eval_settings

## Protocol — metrics

- [ ] Final validation loss, mean ± std over fresh seeds (1–3, or 1–5 if the pilot chose 5) → impl: muonarch/analysis.py:main_table → test: tests/test_analysis.py::test_main_table_mean_std
- [ ] Token speedup: invert per-seed fitted L(D)=αD^−β+γ at each optimizer's final loss; half-LR 2× on seed 1 first, promoted to every curve seed only if it wins; report range over fitted curves × optimizer seeds; <1 = slower → impl: muonarch/analysis.py:token_speedup_range → test: tests/test_analysis.py::test_speedup_range_over_seeds
- [ ] FLOP speedup: same ratio on total FLOPs incl. optimizer term k(4m²n+2m³); Manifold ×(dual steps taken + 2) → impl: muonarch/flops.py:optimizer_flops_per_step + muonarch/analysis.py:flop_speedup_range → test: tests/test_flops.py::test_ns_flops_formula
- [ ] LR sensitivity: basin width in factors of 2 within 0.02 nats (LM) / 1% relative final val loss (MLP) of best, from the lr sweep → impl: muonarch/analysis.py:lr_basin_width → test: tests/test_analysis.py::test_basin_width
- [ ] Divergence rate: share of all trials with NaN, train/val loss above theoretical initial at any logged step, or (analysis-time only) final > same-study optimizer best + 0.5 nats → impl: muonarch/analysis.py:divergence_rate + muonarch/train.py:run (detection) → test: tests/test_analysis.py::test_divergence_rate
- [ ] Diagnostics logged, not tuned on: max attention logit, final spectral norms, expert-share entropy/min/max, aux loss → impl: muonarch/train.py:diagnostics → test: tests/test_train.py::test_diagnostics_logged
- [ ] "A difference is real": all paired seed diffs same sign and mean > 2× larger seed std; labeled a heuristic → impl: muonarch/analysis.py:seed_rule → test: tests/test_analysis.py::test_seed_rule_sign_and_2std

## Implementation-stack requirements (PLAN "Implementation stack" / unit-test mandates)

- [ ] MuonFamily one-step equivalence vs torch.optim.Muon (original and match_rms_adamw) before any sweep → impl: muonarch/optim.py:MuonFamily.step → test: tests/test_ns.py::test_muonfamily_keller_matches_torch_muon_original
- [ ] ns_vec(v) == diag(ns(diag(v))) (MuonAll 1-D as elementwise polynomial, O(d)) → impl: muonarch/optim.py:ns_vec → test: tests/test_ns.py::test_ns_vec_matches_diag_ns
- [ ] Batched ns/msign over stacked same-shape experts equals per-matrix loop → impl: muonarch/optim.py:ns (batch dims) + muonarch/manifold.py:grouped_by_shape → test: tests/test_ns.py::test_ns_batched_experts_matches_loop
- [ ] FineWeb memmap: uint16, ≥45M train + 2M val tokens, byte-level BPE vocab 4096 (incl. one <|endoftext|> after each document) trained on train docs only → impl: scripts/prepare_fineweb.py + muonarch/data.py:FineWeb → test: tests/test_data.py::test_fineweb_counts_and_dtype
- [ ] CIFAR-5m splits disjoint: train = C[:1,802,240], val/test = the following 10k + 10k of part0‖part1 → impl: scripts/prepare_cifar5m.py → test: tests/test_data.py::test_cifar_splits_disjoint
- [ ] Seed-determined data order; 1× order a prefix of the 2× order (budget-curve requirement); LM order over non-overlapping 257-token windows → impl: muonarch/data.py:cifar_batches + lm_batches → test: tests/test_data.py::test_seed_sets_order_and_prefix_consistency
- [ ] Analytical FLOPs within 5% of FlopCounterMode around fwd+bwd+step, once per architecture → impl: muonarch/flops.py:verify_against_counter → test: tests/test_flops.py::test_analytic_vs_flopcountermode_within_5pct
- [ ] Runs resumable/idempotent keyed by stable config hash; done runs skipped; hash % 4 laptop assignment; logged-only fields never change the hash; Phase-I-swept hps all hashed → impl: muonarch/runner.py:config_hash + Runner.run_study → test: tests/test_runner.py::test_hash_stable_across_field_order
- [ ] Results JSON written only on completion (atomic); every eval appended to logs/<study>/<run_hash>.jsonl → impl: muonarch/train.py:run (tmp+rename, JSONL stream) → test: tests/test_train.py::test_result_written_only_on_completion
- [ ] Gating DAG: pilot gates Phase I → coordinate descent (mlp128, dense) / donor-gated Stage A–B (mlp256, mlp512, moe) → fresh seeds → budget curve; Moonlight/MuonAll parallel; Muon/Manifold wait on AdamW; then init ablation and faithful reruns → impl: muonarch/studies.py:build_dag → test: tests/test_studies.py::test_dag_gating_order
- [ ] Init ablation: Muon with Stiefel-projected init, best config, fresh seeds (confounder 10) → impl: muonarch/studies.py:init_ablation_configs + muonarch/manifold.py:ManifoldMuon.project_ → test: tests/test_studies.py::test_init_ablation_configs
- [ ] third_party/manifolds vendored UNMODIFIED with MIT LICENSE; muonarch/msign.py = vendored algorithm with dtype parameterized (fp32 default; DEVIATIONS.md #4), coefficients and op order asserted equal → impl: third_party/manifolds/ + muonarch/msign.py → test: tests/test_manifold_equivalence.py::test_abc_coefficients_match_vendored + test_msign_bf16_matches_vendored_bitwise
- [ ] Smoke runs double as timing runs: 200 steps per optimizer × study, steps 21–200 timed into hardware-results.csv, Manifold at cap 10 where undecided → impl: scripts/smoke.py + scripts/timing.py → test: tests/test_scripts.py::test_timing_writes_hardware_csv
