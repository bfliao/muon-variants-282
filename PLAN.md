# Muon Variants × Architectures — Implementation & Analysis Plan

Nov 16, 2026 · @Paul Ruiz Alliata

## TL;DR — ten decisions to lock before writing code

The design is sound, but ten choices the proposal leaves open can each move the ranking more than the optimizer itself. Lock these first; everything below expands them.

1. **One shared code path for all Muon variants.** A single Newton–Schulz (NS) function (5 steps, fp32, coefficients 3.4445, −4.775, 2.0315) used by Muon, Moonlight-Muon and MuonAll, so they differ only in their stated change. Manifold Muon keeps the exact msign its algorithm needs (Protocol). The official repos differ silently: MuonAll uses 6 NS steps, bf16 and no weight decay; Manifold Muon defaults to 100 dual-ascent steps.
2. **Muon vs Moonlight-Muon is mostly a learning-rate reparameterisation.** Once each optimizer's LR is tuned, Moonlight's 0.2·√max(A,B) factor only matters (a) between matrices of different shapes and (b) because Moonlight shares one LR with the AdamW-handled parameters. On the MLP they genuinely differ: with the 3072→w input layer in the Muon group, Moonlight gives the input layer 0.2·√3072 ≈ 11.1 against 0.2·√w = 2.3–4.5 for the hidden layers, i.e. 2.4–4.9× more relative step than Muon, a ratio that changes with width. H5 tests this.
3. **MuonAll on 1-D parameters is sign-momentum.** NS applied to diag(v) is elementwise. Checked numerically here: signs match 100%, magnitudes land in 0.68–1.13. Implement it as an elementwise polynomial (O(d)), not a d×d matrix.
4. **Measure token speedup the way Wen et al. do.** Fit AdamW's loss against data budget on 3 seeds, using decay branches at 0.125×, 0.25×, 0.5× and 1× from one constant-LR run to 2×, then invert at each optimizer's final loss and report a range (Protocol). Never read "tokens to AdamW's final loss" off another optimizer's intermediate checkpoint.
5. **Count optimizer FLOPs analytically — they dominate the MLP.** At batch 256, NS costs 1.8× (width 128) to 7.9× (width 512) the forward+backward FLOPs. On the dense transformer it is about 4%; on the MoE about 12%, because every expert is orthogonalised every step.
6. **Cap Manifold Muon's dual ascent** (default 5 steps with early stopping; 10 only if a timed run shows it fits) and report the cap as part of the method. Uncapped (100 steps) it costs roughly 200× Muon's Newton–Schulz work per step, too much for a laptop even at width 512.
7. **Put the MLP input layer (3072→w) in the Muon group.** Excluding it makes the share of parameters Muon touches swing from 8% (w=128) to 25% (w=512), which confounds the width study.
8. **Pin `transformers` to a 4.x release for the MoE.** v5 stores experts as fused 3-D tensors \[E, 2I, H\]: `torch.optim.Muon` rejects them, and naive flattening orthogonalises gate and up projections, and all experts, together.
9. **Report fresh, paired seeds.** The best config is chosen on seed 0, so its seed-0 loss is optimistically biased; rerun on seeds 1–3. Each seed fixes init and data order identically across optimizers.
10. **MLP: train on CIFAR-5m in a single pass.** Twenty epochs of CIFAR-10 without augmentation overfits, so validation loss would mix optimization speed with regularisation. Showing each of 900k CIFAR-5m images once removes overfitting, so validation loss measures optimization (see Decisions log).

## Scope and validity — what the study can and cannot claim

The study answers how optimizer rankings depend on architecture and width for small models (≈5M parameters or fewer). It does not predict which optimizer wins at production scale, and the write-up should say so.

**Two studies, two questions.** They share code and protocol but are reported separately, and no cross-study ranking claim is made.

| Study | Question | What varies | What is held fixed | Claim it supports |
| --- | --- | --- | --- | --- |
| A — MLP width study | How do optimizer rankings and optimal learning rates change with width? | Width 128, 256, 512 | Task (CIFAR-5m, one pass), depth, data, schedule | Width and shape effects; LR transfer (H4, H5) |
| B — Dense vs MoE | Does the feed-forward architecture change which optimizer is best? | Dense SwiGLU vs 16-expert top-4 MoE | Data, tokens, attention, active parameters, schedule | Architecture–optimizer interaction (H3) |

An MLP-vs-transformer difference is not evidence of an architecture effect: the data, task and regime (overfitting vs not) change at the same time.

**Suggested wording for the proposal:**

- Abstract, last sentence: replace "with the aim of guiding optimizer selection for future model training" with "to characterise, for small models, how optimizer rankings depend on width and on dense versus Mixture-of-Experts feed-forward blocks."
- Key questions, last sentence: replace "we compare optimizer rankings across architectures" with "we compare rankings across MLP widths (Study A) and between the dense and MoE transformers (Study B), never between the two studies."
- Hypothesis: replace "at most about 1.3×" with "at most about 1.4×, the value Wen et al. report at 130M", and "differs between at least two of the three architectures" with "differs between the dense and MoE transformers, or across MLP widths".

**A null result is a result.** "Indistinguishable after equal tuning" is a stated outcome for every comparison, not a failure: it is the main finding of Wen et al. for many published speedups. Report it with the measured seed spread, so the reader sees how large a difference could have been detected.

**Seed-spread pilot first.** Before the sweeps, run AdamW's default config on 3 seeds in each study (6 short runs in total for the MLP widths 128 and 512, 3 for the dense LM). If the seed std is larger than about a third of the expected gap between optimizers (Wen et al.'s noise threshold is 3e-3 at 130M), spend budget on 5 final seeds rather than more tuning trials.

**Results that follow from arithmetic.** One hypothesis is largely decided before any run. Keep it, but present it as a prediction you confirm, not a discovery:

- *Muon loses per FLOP on the MLP* follows from Newton–Schulz costing 1.8–7.9× the forward and backward passes at batch 256. Report it as "the measured token speedup would need to exceed X× to break even", with X from the FLOP ratio.

## Decisions log — what we chose and why

These are settled; change one only by editing this table, so the report and the code stay in step.

| Topic | Decision | Why |
| --- | --- | --- |
| When training stops | A fixed budget per study, identical for every optimizer: 20M tokens for the transformers, 3,520 steps (one pass over 900k CIFAR-5m images) for the MLP. The LR schedule ends exactly at the budget. The only early stop is divergence (NaN, or loss above the initial loss). | The warmup–stable–decay schedule must know its end step to place the decay, and most of the final loss drop comes from that decay. A "converged" threshold is noisy and fires at different points for different optimizers (sign-like MuonAll has noisier curves), so it would become a hidden confounder. Wall-clock time measures the laptop, not the algorithm. Fixed-budget, end-of-training comparison is what Wen et al. and Semenov et al. use. |
| Tokens to target | Read from a fitted AdamW loss-versus-tokens curve, never from another optimizer's intermediate checkpoint. Get the AdamW budgets by branching, on seeds 1–3: a constant-LR run to 2× with decay branches at 0.125×, 0.25× and 0.5×, AdamW's ordinary 1× runs, and one separate 2× run at half the LR. Fit the curve per seed and report the speedup as a range. | A checkpoint taken before the LR decays understates the loss a run of that length would reach. Branching costs about 2.2 run-lengths per seed instead of 2.9 for separate runs, and all budgets share warmup and data order. Five points per seed leave slack for a 3-parameter fit, and per-seed fits turn loss noise into a speedup interval, which matters because the curve is flat near the end of training. The half-LR 2× run guards against the reused 1× LR handicapping AdamW at 2×, which would inflate the other optimizers' speedups. |
| Equal-compute check | Optional: rerun the best configs with the same total FLOPs as AdamW (fewer steps for costly optimizers, schedule ending there), 3 seeds, reported separately. | Answers whether Manifold Muon is worth its cost at equal compute, which is a different question from equal tokens. |
| Scope | Claims are about small models (≈5M parameters or fewer). | Our models are about 75× smaller than Wen et al.'s smallest, and they show Muon's speedup falling from 1.4× to 1.1× with size. |
| Study structure | Study A (MLP widths) and Study B (dense vs MoE) are reported separately; no ranking is compared across them. | MLP vs transformer changes data, task and regime together, so a difference cannot be attributed to architecture. Dense vs MoE holds data, tokens, attention and active parameters fixed. |
| Null results | "Indistinguishable after equal tuning" is a valid outcome for every comparison, reported with the measured seed spread. A 3-seed AdamW pilot per study runs first; if the spread is large, final configs get 5 seeds instead of 3. | Differences between Muon variants at this scale may be smaller than seed noise. Reporting the spread shows how large an effect could have been detected. |
| Hypotheses decided by arithmetic | H2a (Muon loses per FLOP on the MLP) is reported as a break-even prediction we confirm. | It follows from FLOP counts before any run; presenting it as a discovery would overclaim. H5 was listed here too, wrongly (see the H5 correction row). |
| Newton–Schulz for Muon, Moonlight, MuonAll | One shared routine: 5 iterations, Keller coefficients (3.4445, −4.775, 2.0315), fp32. MuonAll runs at 5 steps, not its repo's 6. | The variants should differ only in their stated idea. Step count and precision are implementation details of the official repos, not part of any method. |
| msign for Manifold Muon | Polar Express, 10 iterations, from the Thinking Machines `msign.py`, used for both dual ascent and the retraction. | Keller's coefficients are tuned for speed and leave singular values spread over roughly 0.7–1.1. Retracting with them would leave weights off the Stiefel manifold, so it would no longer be Manifold Muon. |
| Manifold Muon settings | Dual-ascent cap 5 steps by default (10 only if a timed run shows it fits), tolerance 1e-6, dual step 0.01 decaying linearly, as in the reference code. Fed the same Nesterov momentum buffer as Muon. Steps actually taken are logged. Same-shape MoE experts are processed in one batched call. | With 10-iteration Polar Express, each step costs about (cap + 1) × 10 matrix iterations per weight: roughly 12× Muon's Newton–Schulz at cap 5, 22× at cap 10 and 200× uncapped (100 steps). An earlier version of this plan costed it at half that. Batching matters because the MoE has 300+ small matrices per step, and on a Mac one call per matrix would be limited by per-operation overhead. The reference code uses no momentum; without the shared buffer we would be comparing "constraint" and "no momentum" at once. |
| Faithful-settings check | Each variant's best config is rerun once (seed 1) with its official settings: MuonAll at 6 steps in bf16, Manifold Muon at 100 dual steps (at MLP-512 only if time remains). | If it lands within seed spread, the standardisation is harmless and the report says so in one line; if not, both numbers are reported. |
| Numerical precision | fp32 for all model computation and all matrix iterations on laptops (MPS or CPU). On a CUDA GPU, a study runs entirely in one setting: either fp32 everywhere or TF32 everywhere. Never mix devices or precisions within one study. Each run logs its device and precision. | Seed spread is the yardstick for a "real" difference, so numerical noise must stay below it. MPS support for bf16 is uneven across Mac models. The reference repos use bf16 in different places (Keller's NS, MuonAll, the Polar Express code), so keeping their settings would add precision as an uncontrolled difference between optimizers. TF32 changes matmul rounding relative to fp32; harmless if uniform within a study, a confounder if mixed. |
| `transformers` version | Pinned to 4.57.6. | Checked: in 4.57.6 the Qwen3-MoE experts are separate `nn.Linear` layers in an `nn.ModuleList`, so each expert's gate, up and down matrices are orthogonalised separately. v5 fuses them into 3-D tensors. |
| Compute platform | Laptops for development and smoke tests; optionally one RunPod RTX 4090 for the final sweeps. | A 4090 is [$0.34/hr on Community Cloud](https://www.runpod.io/gpu-models/rtx-4090) ($0.74/hr Secure). Our estimate is 2–4 GPU-hours for the FLOP-heavy MLP runs plus a few hours for the transformers, so $10 should cover the plan. This is an estimate to check with one timed run per optimizer. |
| MLP widths | 128, 256 and 512. Width 1024 is dropped. | MLP-1024 was about 60% of the plan's compute, mostly Manifold Muon on the largest matrices. Without it the plan is an estimated 11–16 hours on one Mac at the default Manifold cap, plus Manifold's untimed overhead on the transformers (Compute budget), inside the 30-hour window. Three widths spanning 4× still test learning-rate transfer (H4). |
| Study A data | CIFAR-5m, one pass: 900k training images from part 0, each seen once (part 1 supplies the extra images for the 2× AdamW budget), at the same 3,520 steps and batch 256. Validation and test images come from the same part and are never trained on. No augmentation. | With no repeated images there is no overfitting, so validation loss measures how fast each optimizer reduces the population loss, the same regime as the language models (each token seen once). This is the "Ideal World" setup of the Deep Bootstrap paper. Limitations and rejected options are in the subsection below. |
| H5 correction | H5 is empirical: Muon and Moonlight-Muon are expected to differ on the MLP, by an amount that changes with width. | An earlier version called it a fairness check because the hidden layers are square. That ignored the 3072→w input layer, which is in the Muon group: Moonlight gives it 0.2·√3072 ≈ 11.1 against 2.3–4.5 for the hidden layers, while Muon gives every layer scale 1, so one learning rate cannot make the two equivalent. Found in an external review. |
| H3 test | Test the interaction directly: for each variant, the seed-paired gap (L\_variant − L\_AdamW) on MoE minus the same gap on dense; real only if beyond twice its seed std. | "The best variant differs" flips on seed noise when gaps are small. The difference in gaps uses every variant's results and pairs seeds, so it is the more robust test of an architecture effect. |
| Headlines and cut order | Study A headline: H4 (optimal learning rate versus width). Study B headline: H3 (dense-vs-MoE interaction). If time runs short, cut Manifold Muon from the MoE first, then from MLP-512; keep the MoE itself. | H4 is the cheapest result and the most likely to come out clean. H3 is the reason Study B exists, so dropping the MoE would remove the project's architecture question; Manifold Muon is its most expensive and least essential arm. |
| Manifold Muon oracle check | Before modifying the copied code, run the unmodified Thinking Machines script on real CIFAR-10 for AdamW and Manifold Muon, 3 seeds each, and check that Manifold Muon beats AdamW in train and test accuracy, as their post reports. | Shows the copied code works before we change it, and recovers the comparison with the cited CIFAR-10 result that moving Study A to CIFAR-5m otherwise loses. |
| Changes from the proposal | Tell the GSI before running: CIFAR-5m instead of CIFAR-10, width 1024 dropped, two separate studies, small-model scope, the 1.4× bound, and the interaction test for H3. | Each change is defensible, but together they change what was proposed; better agreed early than defended at the poster. |
| Hyperparameter search | Two phases after Wen et al.: Phase I coordinate descent over lr, wd, warmup, momentum/β1, β2 and clipping in MLP-128 and dense (max 2 passes, acceptance threshold 2× pilot seed std); Phase II inherits the non-lr/wd values and runs the 12-trial lr × wd recipe in MLP-256, MLP-512 and MoE (Protocol). | The proposal's 12-trial lr × wd grid followed Wen et al.'s equal-budget principle but not their procedure; fixing warmup, momentum or β2 at defaults could favour some optimizers. Phase I is cheap in MLP-128, and Wen et al. found that only the learning rate needed re-tuning across scales for Muon, which is what Phase II re-sweeps (together with wd). Their 3e-3 threshold was calibrated at 130M, so ours comes from our own seed spread. |
| Phase 0 clarifications | MLP uses ReLU and biases (AdamW group except under MuonAll). Branch checkpoints at round(0.8 × branch steps), warmup fixed at round(0.05 × 1× steps). Manifold Stage B: best LR × 2^k, k ∈ {±0.25, ±0.5, ±0.75}. MuonAll's tied embedding gets NS with scale ≈ 5.66 and weight decay. The half-LR 2× run keeps wd. Dense and MoE differ only in the feed-forward block, router and auxiliary loss. | Questions raised by the implementation agent in Phase 0; answered so the plan leaves nothing to guess. Biases make Study A exercise MuonAll's 1-D handling; the bias-free MLP is only for the Thinking Machines oracle. |
| MLP activation and bias | ReLU; biases on (PyTorch default nn.Linear and init). Biases go in the AdamW group, except under MuonAll. The bias-free MLP is used only for the Thinking Machines oracle check. | With biases, MuonAll's 1-D handling is actually exercised in Study A; without them MuonAll would differ from Muon only on the classifier. |
| Budget-curve step rule | A budget of b× has round(b × S) steps (S = 2,441 for the transformers, 3,520 for the MLP). Its branch leaves the constant run at round(0.8 × that), decays over the rest, and gets a full final evaluation at its end. Checkpoints are saved at exact branch steps, independent of the eval grid. Warmup is fixed at round(0.05 × S): 122 steps for the transformers, 176 for the MLP. | Branch points do not fall on the 100-step eval grid (e.g. the 0.125× transformer branch leaves at step 244 of 305), so the rule must be explicit; a fixed absolute warmup keeps every branch identical to a run of that length. |
| Manifold Muon Stage B grid | Best Stage-A LR × 2^k for k ∈ {−0.75, −0.5, −0.25, +0.25, +0.5, +0.75}. The edge rule applies first. | Quarter-octave steps fill both factor-2 intervals around the winner, since Manifold Muon has no weight decay to spend Stage B on. |
| MuonAll and the tied embedding | The tied 4,096×128 embedding is one tensor, counted once, updated by Newton–Schulz with the Keller scale √max(1, 4096/128) ≈ 5.66; weight decay applies; its Newton–Schulz FLOPs are counted. | Matches MuonAll's reference code, which sends embeddings and the head through Muon with the same scale rule. |
| Half-LR 2× run | Weight decay value kept unchanged. | PyTorch applies lr·wd, so the effective decay per step halves along with the learning rate; changing wd as well would move two settings at once. |
| Reference code (Gate 3) | All results are produced by commit `ab48490` of muonarch/ (optimizers, training loop, data, schedule). Each run also logs the commit it ran on. Commit `437bb22` (MLP protocol change, DEVIATIONS #12) touches only the job DAG (studies.py) and analysis.py, not the training code; every pre-existing job hash is unchanged. | Lets the deferred MoE study be checked against the code that produced the MLP and dense results. |
| Frozen protocol (Gate 3) | Frozen for every study, including the deferred MoE: LR grid centres (AdamW 1e-3, Muon 0.02, Moonlight 1e-3, MuonAll 0.02, Manifold 0.1; DEVIATIONS #8), 6-point factor-2 grid and edge rule (lr only, ≤2 trials), fixed 5% warmup (not tuned), the trial budget (Phase I coordinate descent ≤2 passes; Phase II 12 trials), pilot seeds 101–103, selection seed 0, fresh seeds 1..N (N from the Gate-4 pilot), Manifold dual-ascent cap 5 (DEVIATIONS #11). Budgets are in steps/tokens/FLOPs, never wall-clock. | Paul, Gate 3: the MoE runs later and must stay comparable. |
| Later code changes | Any change to shared code (optimizers, training loop, data, schedule) must be proven bit-identical on dense, or the dense runs are redone. Model-side MoE changes (e.g. batched experts) only with an equivalence test against the HF 4.57.6 module. Muon's Newton–Schulz is not batched (option (c) rejected). | Paul, Gate 3. |
| Manifold Muon result (audit, after Gate 3) | Report Manifold Muon's results as measured under the frozen protocol. It is worst at every MLP width, and on dense it is within AdamW's seed noise. Pair this with: (1) the Phase 1 oracle replicated the blog's CIFAR-10 numbers to within about 1 point, so the method and code work; (2) the audit found no bug, and the gap comes from holding all singular values at 1 (unconstrained Muon ends at spectral norms 2.6–12); (3) limitation: Manifold Muon, unlike Muon, gains from a 16× larger classifier/bias lr (0.916 → 0.894; Muon 0.798 → 0.806), which the protocol pins to AdamW's tuned lr. | Full evidence in docs/manifold_audit.md; DEVIATIONS #13. The protocol stays frozen. |
| Final seed count (Gate 4) | Keep 3 fresh seeds for every study (Paul's decision; no seeds 4–5). | The headline (Muon family beats AdamW) clears the seed rule by a wide margin. Comparisons at the threshold are reported as borderline, using the seed-rule heuristic with its caveat: MuonAll vs Muon on MLP-128 (margin 1.09, i.e. mean gap / twice the larger seed std), Muon vs Moonlight on MLP-128 (0.95), Manifold vs AdamW on MLP-512 (1.03). A 5-seed run was started and stopped within minutes. Its 3 finished runs are parked in logs/diagnostics/discarded_seeds45/ and not used. |
| Headline metric: loss vs accuracy | Final validation loss is the headline metric; test accuracy is reported alongside. Where loss and accuracy rank two optimizers differently, call them tied. | A single run's accuracy on 10k test images has a standard error of about ±0.45 points. Every loss/accuracy rank flip seen so far (MLP-256 Muon vs MuonAll, MLP-512 Moonlight vs MuonAll, MLP-512 Manifold vs AdamW) is a 0.1–0.6-point accuracy gap. |
| Dense vs MoE controls | Identical tokenizer, data and data order, 20M tokens, batch 32×256, schedule, attention, tied embeddings, eval sets, seeds and clipping. Only the feed-forward block differs: 16 experts of width 96, top-4 (active width 384, as in the dense SwiGLU), the router, and the auxiliary loss (coefficient 0.01, norm\_topk\_prob = True). | Holding everything else fixed is what lets Study B attribute a difference to the architecture. The MoE's larger total size (\~3.9M vs \~1.7M parameters) raises Muon's optimizer FLOPs; that is expected and counted. |

### Study A data — limitations and rejected options

CIFAR-5m in one pass is the fairest test of optimization speed we can run, at the price of four limitations the report must state.

**Limitations we accept:**

- *Synthetic data.* CIFAR-5m was generated by a diffusion model trained on CIFAR-10; its authors call it close to CIFAR-10 for research purposes but not identical. Our MLP numbers are not comparable with CIFAR-10 results, including the Thinking Machines Manifold Muon experiment.
- *Label noise.* Labels come from a classifier that is 98.5% accurate on CIFAR-10, so roughly 1–2% of labels are likely wrong. This raises the loss floor equally for every optimizer.
- *Optimization, not generalisation.* The single-pass setting says nothing about which optimizer generalises better when a small dataset is reused for many epochs. That question is out of scope; the report should say so.
- *Download and memory.* The two parts are about 6 GB as uint8 images (our estimate from their size); they fit in a 36 GB Mac's unified memory but must be downloaded once and shared.

**Options considered and not chosen:**

| Option | What it keeps | Why not chosen |
| --- | --- | --- |
| CIFAR-10, 20 epochs, final training loss as the primary metric | Comparability with CIFAR-10 and with the Manifold Muon experiment | Generalisation becomes a separate, still-confounded question; selecting on training loss pushes weight decay towards 0. Kept as the fallback if the download fails. |
| Data augmentation (flips, crops) | CIFAR-10 | An MLP has no built-in notion of spatial shifts, so crops act like input noise; it adds a design choice that may interact with optimizers, and does not stop overfitting at 20 epochs. |
| Fewer epochs | CIFAR-10 | Under-trains, and narrow and wide models end at different stages of training. |
| Early stopping on validation loss | CIFAR-10 | Conflicts with the fixed-budget rule: the LR schedule must know its end step. |
| Dropout or label smoothing | CIFAR-10 | Each adds a hyperparameter that would need fair tuning for every optimizer, which the 12-trial budget cannot cover. |

## Literature notes — what to copy and what to watch

Each cited optimizer paper was tested in a narrow setting with its own defaults; the table records the defaults you inherit and the trap each one hides.

| Ref | What it actually does | What you take | Trap for this study |
| --- | --- | --- | --- |
| \[3\] [Muon](https://github.com/KellerJordan/Muon) (Jordan et al. 2024) | Nesterov momentum 0.95 → 5 NS steps → update scaled by √max(1, fan\_out/fan\_in). Muon for hidden ≥2-D weights; AdamW for embeddings, classifier head, gains, biases. README example: Muon lr 0.02, wd 0.01; AdamW group lr 3e-4, betas (0.9, 0.95). Says only lr and wd need tuning. | Default momentum, NS steps, param split, starting grid. | "Muon" must be defined precisely: with weight decay (the repo now has it), and with which AdamW-group LR. For a ConvNet the README excludes the first layer — for your MLP that rule would leave most parameters out (see TL;DR 7). |
| \[5\] [Moonlight](https://github.com/MoonshotAI/Moonlight) (Liu et al. 2025) | Adds weight decay and rescales each update to AdamW's RMS: lr·0.2·√max(A,B). Claims \~2× compute efficiency (≈52% of AdamW's FLOPs) on a DeepSeek-V3-style MoE. | `torch.optim.Muon(adjust_lr_fn="match_rms_adamw")`; one LR and one wd shared by all parameters. | Its selling point is reusing AdamW's hyperparameters. Under per-optimizer tuning that advantage disappears by construction, so what you actually test is the shape-dependent scaling plus LR sharing. |
| \[1\] [Modular manifolds](https://thinkingmachines.ai/blog/modular-manifolds/) / Manifold Muon (Bernstein 2025) | Keeps each matrix on the Stiefel manifold (all singular values = 1). Each step: dual ascent on Λ, update −η·msign(G + 2W(Λ+Λᵀ)), then retract W ← msign(W). [Code](https://github.com/thinking-machines-lab/manifolds) (PyTorch): 100 dual steps, α=0.01 decaying, tol 1e-6, η=0.1, wide matrices transposed to tall. Experiment: bias-free MLP 3072→128→64→10, batch 1024, lr 0.1 linearly decayed, no augmentation, every layer on the manifold, weights projected onto it before training, 3 seeds. | `manifold_muon.py` + `msign.py` copied as-is, with the step cap exposed. | Initialisation is changed along with the optimizer (projection at step 0), so gains may be partly an init effect. No weight decay on constrained matrices. Cost scales with the number of dual steps. Only compared against AdamW, with the classifier also constrained. |
| \[14\] [Yoshihara 2026](https://arxiv.org/abs/2606.13276) ([code](https://github.com/kiratoyoshihara/module-wise-manifold-muon), nanoGPT-based) | GPT-2 small on OpenWebText, shared hyperparameters across arms. Stiefel on attention + DGram on MLP gives val loss 3.354, all-Stiefel 3.368, unconstrained 3.386; all-DGram is unstable. Mechanism: singular-value growth → attention-logit inflation → softmax saturation. | A diagnostic: track max attention logit (or σmax of W\_q, W\_k) per optimizer. | Qwen3 has QK-norm, which caps exactly this failure mode, so expect smaller attention-stability differences than in GPT-2. Author flags the gaps as small and needing more seeds. |
| \[8\] [MuonAll](https://arxiv.org/abs/2511.06086) (Page et al. 2025, [code](https://github.com/Saurabh750/optimizer)) | Fine-tunes models ≤0.5B on OpenOrca; reports parity with AdamW. In `muon.py`: 1-D params go through torch.diag → NS → diagonal; embeddings and LM head also go through Muon; no weight decay; bf16; 6 NS steps; tensors with >2 dims are flattened to (dim0, rest). | The idea only; reimplement 1-D handling elementwise. | It changes three things at once (1-D params, embeddings/head, no wd). Its >2-D flattening would orthogonalise across MoE experts. The diag trick allocates d×d per vector. |
| \[11\] [Wen et al.](https://arxiv.org/abs/2509.02046) ([code](https://github.com/marin-community/marin/tree/kaiyue/optimizers)) | Coordinate-descent tuning of every hyperparameter; a change is kept only if val loss improves by >3e-3. Speedup = AdamW-equivalent data from a fitted L(D)=αD^−β+γ. Muon ≈1.4× at 130M, ≈1.3× at 300–520M, ≈1.1× at 1.2B; Soap/Kron overtake Muon at 8–16× Chinchilla. Optimal wd differs by optimizer. | The speedup estimator, the 3e-3 noise threshold idea, comparing only at end of training. | Your models (1.7M) are \~75× smaller than their smallest, and 20M tokens is \~17 tokens per non-embedding parameter (≈1× Chinchilla, where Muon looked best). Your "≤1.3×" hypothesis is below their 130M figure of 1.4×: either use 1.4× or justify 1.3×. |
| \[13\] μP / Tensor Programs V (Yang et al. 2022) | Under standard parameterisation the optimal Adam LR shifts with width; μP fixes it. | A prediction for your MLP width sweep. | Muon's √(fan\_out/fan\_in) scaling is a spectral μP-like rule, so it predicts a width-stable optimal LR for Muon but not for AdamW (H4 below). |
| \[6\] AdamW | Decoupled weight decay. | Baseline. | In PyTorch the decay applied is lr·wd, so lr and wd are coupled: optimal regions appear as diagonal ridges in an lr×wd heatmap. Keep the same convention in Muon (it uses lr·wd too). |
| \[12\] Qwen3 | GQA, SwiGLU, RMSNorm, RoPE, QK-norm, no QKV bias; MoE without shared experts, global-batch load balancing. | Architecture via Hugging Face configs. | QK-norm adds per-head 1-D gains (AdamW group, or sign-updates under MuonAll). |

**Papers worth adding** (each changes a design choice, not just the bibliography):

- [Qiu et al. 2025](https://arxiv.org/abs/2512.05620), NeurIPS 2025 — μP for Muon/Shampoo: LR transfer still drifts at finite width; independent weight decay scaled as 1/width is near-optimal; with both, Muon gives a consistent \~1.4× from 190M to 1.4B. Directly frames your MLP width study.
- [Semenov, Pagliardini, Jaggi 2025](https://arxiv.org/abs/2509.01440) ([code](https://github.com/epfml/llm-optimizer-benchmark)) — benchmark of 11 optimizers including MoE models, with ablations on warmup, LR decay and batch size. Use it to justify fixed schedule choices.
- [Essential AI 2025](https://arxiv.org/abs/2505.02222) — Muon keeps its data efficiency at batch sizes far above AdamW's critical batch size. Your small fixed batch (32×256 tokens) therefore likely understates Muon; name it as a limitation.
- Bernstein & Newhouse, "Old optimizer, new norm" (2024) and Yang, Simon & Bernstein, "A spectral condition for feature learning" (2023) — the theory behind Muon's shape scaling; cite them when explaining H4 and H5.
- Amsel et al., "The Polar Express" (2025) — the msign routine in the Manifold Muon code; cite when you state its cost.

## Implementation stack — about 450 lines in total

Use Hugging Face for the models, plain PyTorch for everything else, and write the Muon family yourselves in one \~60-line file; use the official repos as test oracles, not as dependencies.

| Piece | Use | Est. lines |
| --- | --- | --- |
| Environment | `uv` lockfile shared by all four laptops: PyTorch ≥2.9 (ships `torch.optim.Muon`), `transformers` pinned to a 4.x release, `datasets`, `tokenizers`, `torchvision`, `matplotlib`. Check that `Qwen3MoeSparseMoeBlock.experts` is an `nn.ModuleList` in the pinned version. | — |
| LM data | Stream `HuggingFaceFW/fineweb-edu`, config `sample-10BT`; train a 4,096-token byte-level BPE with `tokenizers` on the train documents only; write at least 45M training tokens (the 2× AdamW budget needs 40M) plus 2M validation tokens as `uint16` memmap (nanoGPT pattern). One person builds it, everyone uses the same file. | 60 |
| CIFAR-5m data | Download two CIFAR-5m parts once (`gs://gresearch/cifar5m/part0.npz` and part1.npz, about 1M images each). Part 0 gives 900k training images plus fixed 10k validation and 10k test sets; part 1 extends training to 1.8M images for the 2× AdamW budget, so every budget stays single-pass. Hold them as `uint8` tensors on the device and normalise per batch (no DataLoader, as in [airbench](https://github.com/KellerJordan/cifar10-airbench)). One person downloads, everyone uses the same file. | 30 |
| Models | `Qwen3ForCausalLM` / `Qwen3MoeForCausalLM` from configs (below); MLP = `nn.Sequential` of 4 `nn.Linear`. | 25 |
| Optimizers | `muon_family.py` (below) for Muon, Moonlight-Muon and MuonAll; Manifold Muon's `update()` copied from [thinking-machines-lab/manifolds](https://github.com/thinking-machines-lab/manifolds/tree/main/src) with the dual-step cap exposed. AdamW from `torch.optim`. | 60 + 40 |
| Train loop + runner | One `train.py` driven by a config dict; a job list hashed to a laptop (`hash % 4`), skipped if its result JSON exists, so runs resume after crashes. Log every eval to JSONL. | 160 |
| FLOPs | Analytical formula (Protocol section), checked once per architecture with `torch.utils.flop_counter.FlopCounterMode` around forward + backward + `optimizer.step()`. | 20 |
| Plots | One script reading all JSONL into a DataFrame. | 120 |

**Model configs** (the dense model gives 1.70M parameters with tied embeddings, matching the proposal):

```python
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
common = dict(vocab_size=4096, hidden_size=128, num_hidden_layers=6,
              num_attention_heads=2, num_key_value_heads=1, head_dim=64,
              max_position_embeddings=256, tie_word_embeddings=True)
dense = Qwen3ForCausalLM(Qwen3Config(intermediate_size=384, **common))
moe = Qwen3MoeForCausalLM(Qwen3MoeConfig(
    num_experts=16, num_experts_per_tok=4, moe_intermediate_size=96,
    norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[],
    output_router_logits=True, router_aux_loss_coef=0.01, **common))  # fix the coef for all optimizers
```

**The whole Muon family** — Muon (`rule="keller"`), Moonlight (`rule="moonlight"`) and MuonAll (pass 1-D params in too) share one NS routine, so their differences are exactly the ones you claim:

```python
import math, torch
A, B, C = 3.4445, -4.7750, 2.0315

def ns(G, steps=5):                      # (..., m, n); batched dims = experts
    X = G.float(); tall = X.size(-2) > X.size(-1)
    if tall: X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        M = X @ X.mT
        X = A * X + (B * M + C * M @ M) @ X
    return X.mT if tall else X

def ns_vec(g, steps=5):                  # == diag(ns(diag(g))), in O(d)
    x = g.float() / (g.norm() + 1e-7)
    for _ in range(steps):
        x = A * x + B * x**3 + C * x**5
    return x

def scale(shape, rule):                  # nn.Linear weight = (fan_out, fan_in)
    m, n = shape[-2], shape[-1]
    return math.sqrt(max(1, m / n)) if rule == "keller" else 0.2 * math.sqrt(max(m, n))

class MuonFamily(torch.optim.Optimizer):
    def __init__(self, params, lr, wd=0.0, momentum=0.95, rule="keller"):
        super().__init__(params, dict(lr=lr, wd=wd, momentum=momentum, rule=rule))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                buf = self.state[p].setdefault("buf", torch.zeros_like(p))
                buf.mul_(g["momentum"]).add_(p.grad)
                u = p.grad + g["momentum"] * buf                      # Nesterov
                u = ns_vec(u) if p.ndim == 1 else ns(u) * scale(p.shape, g["rule"])
                p.mul_(1 - g["lr"] * g["wd"]).add_(u.to(p.dtype), alpha=-g["lr"])
```

Two unit tests before any sweep: (1) one step of `MuonFamily(rule="keller")` and `rule="moonlight"` against `torch.optim.Muon` with `adjust_lr_fn="original"` / `"match_rms_adamw"` on the same weights; (2) `ns_vec(v)` against `torch.diag(ns(torch.diag(v)))`. Feed Manifold Muon the same Nesterov buffer `u` instead of the raw gradient: the reference code uses no momentum, and without this the comparison mixes "manifold constraint" with "no momentum".

**Reference repos** — read these, copy from the first two only:

- [thinking-machines-lab/manifolds](https://github.com/thinking-machines-lab/manifolds) — `manifold_muon.py`, `msign.py` (copy).
- [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — parameter-split conventions, NS coefficients.
- [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — a tuned Muon LM recipe to sanity-check your LR ranges.
- [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) — `examples/toy_train.py` for the RMS-matched variant.
- [Saurabh750/optimizer](https://github.com/Saurabh750/optimizer) — MuonAll reference (`muon.py`), to cite exactly what you changed.
- [kiratoyoshihara/module-wise-manifold-muon](https://github.com/kiratoyoshihara/module-wise-manifold-muon) — Manifold Muon inside a transformer.
- [marin-community/marin @ kaiyue/optimizers](https://github.com/marin-community/marin/tree/kaiyue/optimizers) and [epfml/llm-optimizer-benchmark](https://github.com/epfml/llm-optimizer-benchmark) — tuning protocols and hyperparameter ranges.

## Protocol — what is fixed, what is tuned, what is measured

Every optimizer gets 12 tuning trials on seed 0 plus 3 fresh seeds for its best config, with the same schedule, clipping, batch and data order; only the rows below differ.

**Optimizers and parameter groups**

| Optimizer | Muon-type update on | AdamW on | Tuned (12 trials) | Fixed |
| --- | --- | --- | --- | --- |
| AdamW | — | everything | lr, wd | betas (0.9, 0.95), eps 1e-8 |
| Muon | hidden matrices: q, k, v, o; MLP/SwiGLU; each expert's gate, up, down separately; MLP input layer 3072→w and hidden w→w | embeddings / tied head, CIFAR classifier, norms, QK-norm gains, biases, MoE router | Muon lr, wd (same wd on both groups) | momentum 0.95 Nesterov, 5 NS steps, scale √max(1, fan\_out/fan\_in); AdamW-group lr = AdamW's tuned lr |
| Moonlight-Muon | same | same | one lr and one wd for both groups | scale 0.2·√max(fan\_out, fan\_in) |
| Manifold Muon | same, constrained to Stiefel; weights projected at step 0 | same | lr (12 values; no wd on constrained matrices) | dual steps ≤5 by default (≤10 if a timed run shows it fits), tol 1e-6; same-shape MoE experts processed in one batched call; same Nesterov buffer as Muon; AdamW group as for Muon |
| MuonAll | every parameter: matrices by NS, 1-D by `ns_vec`, embeddings/tied head by NS | — | lr, wd (grid includes wd = 0, the paper's setting) | as Muon |

Write the parameter→group map into the config, print it at start-up, and report the fraction of parameters each optimizer updates with Muon: 69% on the dense transformer (embeddings are 31%), and over 99% on the MLP once the input layer is included.

**Newton–Schulz settings — standardise the Muon update, keep Manifold Muon's exact msign**

| Variant | Official setting | Used here | Why |
| --- | --- | --- | --- |
| Muon, Moonlight-Muon | 5 steps, Keller coefficients; bf16 in Keller's repo | 5 steps, Keller coefficients, fp32 | The reference setting; fp32 because MPS bf16 is uneven |
| MuonAll | 6 steps, bf16 (`Saurabh750/optimizer`) | 5 steps, fp32 | Isolates MuonAll's idea (all parameters through Muon) from an incidental step count |
| Manifold Muon | Polar Express msign; 100 dual steps | Polar Express msign from the reference `msign.py`, for both dual ascent and retraction; ≤10 dual steps | Keller's coefficients are tuned for speed, not exactness: they leave singular values spread over roughly 0.7–1.1. Retracting with them would leave weights off the Stiefel manifold, so it would no longer be the algorithm being tested |

One faithful-settings check per variant: rerun each variant's best config once with its official settings (MuonAll at 6 steps in bf16, Manifold Muon at 100 dual steps) on seed 1. If it lands within seed spread, the standardisation is harmless and you say so in one line; if not, report both numbers. Cost: 3 runs per study, and the Manifold one is the expensive one, so run it at MLP-512 only if time remains. The FLOP count always uses the iterations each run actually performed.

**Tuning, Phase I — coordinate descent in MLP-128 and dense (after Wen et al.)**

1. Start every optimizer at its literature defaults, seed 0.
2. Sweep one hyperparameter at a time, holding the others at their current best, in this order: learning rate (6-point factor-2 grid); weight decay {0, 0.3d, d, 3d}; warmup {0, 2%, 5%, 10%} of the 1× steps; momentum (Muon family {0.9, 0.95, 0.98}; AdamW β1 {0.8, 0.9, 0.95}); AdamW β2 {0.95, 0.99, 0.999}; gradient clipping {off, 0.5, 1.0}. Manifold Muon skips weight decay on its constrained matrices.
3. Accept a new value only if final validation loss improves by more than twice the seed std measured in that study's pilot (Wen et al. used 3e-3 at 130M).
4. At most 2 passes; stop early if a full pass changes nothing. The edge rule applies to the learning rate.
5. AdamW goes first; the AdamW groups inside Muon-family runs use AdamW's Phase I values.

Phase II: MLP-256 and MLP-512 inherit each optimizer's non-lr/wd values from MLP-128, and the MoE inherits them from dense; those studies then run only the 12-trial lr × wd recipe below. The Phase I studies use their Phase I result directly. Report the number of trials each optimizer used.

**Tuning, Phase II — the same 12 trials for every optimizer in MLP-256, MLP-512 and MoE**

1. Stage A, 6 trials: learning rates on a factor-2 log grid (spans 32×), centred on the literature default, at the default wd (AdamW 0.1; Muon family 0.01 × its own scale).
2. Stage B, 6 trials: the 2 best learning rates × 3 weight decays {0, default, 3× default}. Manifold Muon instead spends Stage B on 6 more learning rates between its best neighbours.
3. Edge rule: if the winner sits on a grid edge, add up to 2 trials beyond it. Apply the rule to every optimizer and report how often it fired.
4. Select on seed-0 final validation loss; rerun the winner on seeds 1, 2, 3. Report those three only.

Use the same grid positions relative to the default for every width of the MLP, so the optimal-LR-versus-width plot (H4) comes out of the tuning runs for free.

**Shared training settings**

- Schedule: linear warmup over 5% of steps → constant → linear decay to 0 over the last 20% (warmup-stable-decay). Identical for all optimizers; not tuned.
- Global gradient-norm clipping at 1.0 for all. fp32 everywhere (MPS bf16 support is uneven, and mixed precision would differ by device).
- Seed s fixes initialisation and data order identically across optimizers, so comparisons are paired.
- Transformer: batch 32×256 = 8,192 tokens, 2,441 steps for 20M tokens. Evaluate every 100 steps on a fixed 0.5M-token set; final evaluation on a fixed 2M-token set. Same sets for everyone.
- MLP: batch 256, 3,520 steps = one pass over 900k CIFAR-5m images, no image repeated; the seed sets the image order. Evaluate validation loss and accuracy every 176 steps (45k images, the old epoch length); test set only for the final table.

**Metrics**

- *Final validation loss* — mean ± std over seeds 1–3.
- *Token speedup* — on seeds 1–3, run tuned AdamW at constant LR (warmup fixed at 5% of the 1× budget) to 2× the budget, and branch a decay at 0.125×, 0.25× and 0.5×: each branch leaves the constant run at 80% of its budget and decays over the last 20%. The 1× point is AdamW's ordinary fresh-seed run; the 2× point is the long run's own decay. Add one separate 2× run at half the LR and keep the lower 2× loss, because the branches reuse the 1× LR. Fit the curve below per seed (5 points, 3 parameters) and invert it at each optimizer's final loss. Report the speedup as the range over the 3 fitted curves × 3 optimizer seeds: near the end of training the curve is flat, so small loss noise becomes large speedup noise, and the range shows it. Speedup = D\_AdamW / D. Values below 1 mean slower than AdamW.

```latex
\hat L_{\mathrm{AdamW}}(D) = \alpha D^{-\beta} + \gamma, \qquad \mathrm{speedup}_{\mathrm{tok}} = \hat L_{\mathrm{AdamW}}^{-1}(L_{\mathrm{opt}}) / D
```

- *FLOP speedup* — the same ratio with each side's total training FLOPs, counting the optimizer:

```latex
F = 6\,N_{\mathrm{active}}\,D + F_{\mathrm{attn}} + S \sum_{W \in \mathrm{Muon}} k\,(4m^2 n + 2m^3), \quad m = \min(\mathrm{rows},\mathrm{cols}),\ n = \max(\mathrm{rows},\mathrm{cols})
```

S = steps, k = NS iterations per msign. For Manifold Muon multiply the last term by (dual steps actually taken + 2); log the steps taken. AdamW's own update is \~10 FLOPs per parameter, negligible.

- *LR sensitivity* — width of the learning-rate range (in factors of 2) whose final loss is within 0.02 nats (LM) or 1% (MLP) of the optimizer's best, read from Stage A.
- *Divergence rate* — share of all trials that hit NaN, exceed the initial loss (ln 4096 = 8.32 for the LM, ln 10 = 2.30 for CIFAR) at any evaluation, or end more than 0.5 nats above the optimizer's best.
- *Diagnostics* (logged, not tuned on): max attention logit; spectral norm of each Muon matrix at the end; for the MoE, entropy of the per-expert token share, min/max expert share, and the auxiliary loss.
- *A difference is real* when all three paired seed differences have the same sign and the mean difference exceeds twice the larger seed std. With 3 seeds this is a heuristic, not a test; say so.

## Hypotheses — each with the result that would falsify it

H1–H3 are the proposal's three, made testable and scoped to Study A or B; H4–H7 come from runs you already plan. Every comparison allows the outcome "indistinguishable after equal tuning".

| ID | Study | Hypothesis | Type | Falsified if |
| --- | --- | --- | --- | --- |
| H1 | B | After equal tuning, Muon-family optimizers reach AdamW's loss with fewer tokens, with token speedup between 1.0× and 1.4× on the dense transformer. | Empirical (Wen et al.: 1.4× at 130M) | Best variant's speedup range lies entirely below 1.0 or above 1.4 |
| H2a | A | On the MLP at batch 256, no Muon variant beats AdamW per FLOP for width ≥ 256, because breaking even needs a token speedup above the FLOP ratio (≥4.7×). | Predicted by arithmetic; confirm | Measured token speedup exceeds the break-even ratio |
| H2b | B | On the transformers, Manifold Muon's FLOP speedup is below 1, while Muon's and Moonlight's stay within 10% of their token speedup. | Empirical | Manifold Muon ≥ 1 per FLOP |
| H3 | B | The Muon-family advantage over AdamW differs between dense and MoE. Test as an interaction: for each variant, the per-seed gap (L\_variant − L\_AdamW) on MoE minus the same gap on dense, paired by seed. | Empirical (interaction) | For every variant, the difference in gaps is within twice its seed std |
| H4 | A | AdamW's optimal LR shifts by ≥2× from width 128 to 512; Muon's (shape-scaled) optimum stays within one grid step. | Empirical (μP, spectral condition, Qiu et al.) | Muon's optimum moves as much as AdamW's |
| H5 | A | Muon and Moonlight-Muon differ on the MLP, and the gap changes with width, because Moonlight gives the 3072→w input layer more step relative to the hidden layers than Muon does: 4.9× at width 128, 3.5× at 256, 2.4× at 512. | Empirical | Indistinguishable at every width |
| H6 | A, B | MuonAll is no better than Muon from scratch and has a narrower LR basin, because 1-D gains get sign-like updates of size ≈ lr. | Empirical | MuonAll beats Muon, or its basin is at least as wide |
| H7 | B (exploratory) | Muon-family optimizers give rarely routed experts full-size updates, changing expert load balance relative to AdamW. | Exploratory | Expert-share entropy curves overlap across optimizers |

## Confounders and how to neutralise them

Twenty confounders, sorted by how much they could move the ranking; 15 cost nothing beyond a decision, and the other five add a handful of runs per study.

| # | Confounder | How it biases the result | Mitigation | Extra runs |
| --- | --- | --- | --- | --- |
| 1 | Reading speedup off intermediate checkpoints | A checkpoint taken before LR decay understates the loss a run of that length would reach; Wen et al. show rankings flip | Multi-budget AdamW curve (Protocol) | 3–5 per study |
| 2 | Unequal or edge-truncated tuning | The optimizer with the luckier grid wins | Same 12-trial recipe; edge rule; publish all grids | ≤2 per optimizer, rarely |
| 3 | Winner's curse | Best-of-12 on seed 0 is biased low | Report fresh seeds 1–3 only | 1 per optimizer |
| 4 | Implementation drift (NS steps, bf16, coefficients, momentum form) | Variants differ in more than their idea | One shared `ns()`; unit tests vs `torch.optim.Muon` | 0 |
| 5 | Parameter-group membership | Muon touches 8–25% of MLP parameters if the input layer is excluded; 69% of the dense LM | Input layer in Muon; print the map; report fractions | 0 |
| 6 | AdamW-group LR inside Muon runs | Untuned auxiliary parameters can sink a variant | Reuse AdamW's tuned lr (Muon, Manifold); shared by design (Moonlight) | 0 |
| 7 | Weight-decay semantics | Optimal wd differs by optimizer; Manifold has none | Tune wd per optimizer; one lr·wd convention | 0 |
| 8 | Optimizer FLOPs omitted | MLP: NS is 1.8–7.9× the model FLOPs | Analytical count, verified with `FlopCounterMode` | 0 |
| 9 | Manifold Muon: no momentum in reference code | Mixes "constraint" with "no momentum" | Feed it the Nesterov buffer | 0 |
| 10 | Manifold Muon: orthogonal projection at init | Gain may come from the init | Muon with the same projected init, best config, 3 seeds | 3 per study |
| 11 | Manifold Muon: dual-step cap | Fewer steps = cheaper but less exact | Fix the cap; log steps taken and final dual residual | 0 |
| 12 | MLP overfitting | Final val loss rewards regularisation, not speed | Train on CIFAR-5m in one pass, so no image repeats and overfitting cannot build up; still report the train–validation gap | 0 |
| 13 | Fixed small batch | Muon's edge grows with batch size (Essential AI) | Name as limitation; optional 4×-batch run for AdamW and Muon best configs | 6 (optional) |
| 14 | LR schedule and warmup | Optimizers prefer different warmups | One fixed WSD schedule; cite Semenov et al. for sensitivity | 0 |
| 15 | MoE fused tensors | Orthogonalising gate+up together, or across experts, is a different optimizer | `transformers` 4.x; per-expert, per-projection NS | 0 |
| 16 | MoE routing settings | Aux-loss coefficient interacts with optimizer | Fix the coefficient; router on AdamW (except MuonAll); log load entropy | 0 |
| 17 | QK-norm | Removes the attention-logit failure Muon variants can trigger | Keep it (Qwen-faithful); log max logit; discuss | 0 |
| 18 | Hardware heterogeneity | Numerics and speed differ across laptops | FLOPs, not wall-clock; fp32; round-robin jobs across laptops; log the device; one lockfile | 0 |
| 19 | Data and evaluation noise | Unpaired seeds and small val sets inflate spread | Paired seeds; fixed 2M-token val set; tokenizer trained on train split only | 0 |
| 20 | Tiny scale | Results at 1.7M may not extrapolate (Wen et al.: speedups shrink with size) | Claim rankings at this scale only; compare to their 130M figure | 0 |

## Plots — eight figures, one per question

> **Superseded (DEVIATIONS #14):** the report's figures are specified in docs/figures_spec.md (8 main + 8 appendix) and generated by `uv run python -m muonarch.figures`. The table below is the original plan, kept for the record.

Figures 1–4 go in the main text and answer the three key questions; 5–8 are mechanism and robustness figures for the appendix. Use one fixed colour per optimizer across all figures.

| # | Figure | x | y | Answers |
| --- | --- | --- | --- | --- |
| 1 | Headline speedup: dot + whisker per optimizer, one panel per architecture, separate markers for tokens and FLOPs, reference line at 1 | optimizer | speedup vs AdamW (log scale) | Q2, H1, H2 |
| 2 | Learning-rate sensitivity: final loss vs LR (Stage A), one line per optimizer, × for diverged trials, one panel per architecture | learning rate (log2) | final val loss | Q3, H6 |
| 3 | Validation-loss curves of the tuned configs, mean line with min–max band over seeds 1–3, inset zoomed on the last 20% | tokens; second row = FLOPs | val loss | Q1, H1, H2 |
| 4 | Rank chart: each optimizer's rank in MLP-128, MLP-256, MLP-512, dense, MoE; lines connect the same optimizer; hollow markers where differences are within seed spread | study | rank by final loss | H3 |
| 5 | Width transfer (MLP): optimal LR vs width, one line per optimizer, plus best loss vs width | width (log2) | optimal LR; best loss | H4, H5 |
| 6 | Optimizer cost: NS FLOPs as a multiple of forward+backward FLOPs, per architecture and width; Manifold Muon at its logged dual steps | width / architecture | FLOP ratio (log) | H2 |
| 7 | lr × wd heatmaps from Stage B, one per optimizer and architecture | lr | wd | Fairness of tuning; lr·wd coupling |
| 8 | Diagnostics: max attention logit over training (dense, MoE); expert-share entropy over training (MoE); final spectral norms per layer | tokens | diagnostic | H7, Yoshihara mechanism |

Plus one table for the report: final val loss (mean ± std), token and FLOP speedup, LR-basin width and divergence rate, per optimizer × study, with the winner per study in bold only where the difference passes the seed rule.

## Compute budget and run plan

With Manifold Muon costed correctly (10-iteration Polar Express per matrix sign), the plan is about 480–530 runs and 7.4×10¹⁶ FLOPs at the default cap of 5 dual steps (9.9×10¹⁶ at cap 10): about 21 hours at an assumed 1 TFLOP/s (28 at cap 10). On one Mac that is an estimated 11–16 hours plus Manifold Muon's untimed overhead on the transformers — inside the 30-hour window, but time the transformer Manifold runs first.

Two-phase tuning adds an estimated 6–10 hours on one Mac, mostly the dense coordinate descent, bringing the total to about 20–26 hours plus Manifold Muon's transformer overhead. That is tight for the 30-hour window, so plan on a teammate's laptop or the RunPod option for part of the sweep.

Full run count per study, including the extras an earlier version left out:

| Item | Runs per study |
| --- | --- |
| Tuning: 5 optimizers × 12 trials | 60 |
| Fresh seeds: 5 optimizers × 3 | 15 |
| Edge-rule trials (only if a winner sits on a grid edge) | 0–10 |
| AdamW budget curve: 3 constant-LR runs to 2×, 9 short decay branches, 1 half-LR 2× run | 13 (≈ 7 run-lengths) |
| Init ablation: Muon with Stiefel-projected init, 3 seeds | 3 |
| Faithful-settings reruns, 1 per variant | 3 |
| Seed-spread pilot (MLP-128, MLP-512 and dense only) | 3 |
| **Total** | **94–107** |

Across the five studies that is about 480–530 runs. Once for the whole project: the Manifold Muon oracle check (6 short runs) and the timed 200-step runs (about 25). Optional, not costed: the equal-compute check and the 4×-batch check.

| Study | Total FLOPs, cap 5 | Total FLOPs, cap 10 | Hours at 1 TFLOP/s, cap 5 | Estimated hours on one Mac, cap 5 | Share from Manifold Muon, cap 5 |
| --- | --- | --- | --- | --- | --- |
| MLP-128 | 1.2×10¹⁵ | 1.9×10¹⁵ | 0.3 | 0.1–0.2 | 70% |
| MLP-256 | 5.1×10¹⁵ | 8.2×10¹⁵ | 1.4 | 0.5–0.7 | 75% |
| MLP-512 | 2.4×10¹⁶ | 3.9×10¹⁶ | 6.7 | 2–3.5 | 77% |
| Dense Qwen | 2.0×10¹⁶ | 2.2×10¹⁶ | 5.6 | 2–4, plus Manifold overhead | 24% |
| MoE Qwen | 2.4×10¹⁶ | 2.8×10¹⁶ | 6.7 | 6–8, plus Manifold overhead | 33% |

Manifold Muon figures assume every allowed dual step is used; early stopping can only make them cheaper. The Mac column assumes 2–3 TFLOP/s sustained in fp32 on the MLP, and that the tiny transformers are limited by per-operation overhead (about 1.5–3 minutes per dense run; MoE slower because Hugging Face 4.x loops over experts in Python). Manifold Muon on the transformers runs up to (cap + 1) × 10 matrix iterations on every small matrix each step, so its time is set by per-operation overhead that FLOPs do not capture; batch same-shape experts and time one run before scheduling.

The 1 TFLOP/s figure is a placeholder: replace it with your `hardware-results.csv` measurements. At d\_model = 128 the transformer runs are probably bound by kernel-launch overhead rather than FLOPs, so time one full run per optimizer on each laptop before scheduling.

**If the budget is tight, cut in this order:**

1. Drop Manifold Muon from the MoE study first (keep the MoE with the other four optimizers: it is Study B's only comparison), then from MLP-512, and say so.
2. Optionally raise the MLP batch to 1,024, as in the Thinking Machines experiment: NS overhead per training image falls 4× (to 0.4–4× forward+backward), at the price of 880 steps instead of 3,520. Decide before any run; it changes the study.
3. Last resort: drop MLP-256. Two widths still show whether the optimal LR moves, but not the trend.
4. Run the MoE after the dense study, but do not drop it: without it Study B has no comparison.

**Order of work** (each step gates the next):

1. Shared environment, data memmap, CIFAR-5m tensor; unit tests for `ns`, `ns_vec`, Moonlight scaling and FLOP counts. Then the Manifold Muon oracle check (reproduce the Thinking Machines CIFAR-10 MLP result with the unmodified code, on real CIFAR-10) and a timed 200-step run of every optimizer in every study.
2. One smoke run per optimizer × architecture (200 steps): checks param groups, that loss falls, and logging.
3. MLP study, smallest width first; plot Figures 2 and 5 as soon as width 128 lands.
4. Dense transformer study; AdamW budget curve as soon as AdamW's tuning ends.
5. MoE study.
6. Figures, the results table, and a short limitations paragraph built from confounders 13, 17 and 20.

## Sources

Pages opened for this plan. The FLOP figures and the MuonAll-equals-sign-momentum check were computed here, not taken from a source.

- [torch.optim.Muon documentation (PyTorch 2.9)](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html)
- [KellerJordan/Muon README](https://github.com/KellerJordan/Muon)
- [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight)
- [Modular Manifolds — Thinking Machines Lab](https://thinkingmachines.ai/blog/modular-manifolds/)
- [thinking-machines-lab/manifolds: manifold\_muon.py](https://github.com/thinking-machines-lab/manifolds/blob/main/src/manifold_muon.py) and [main.py](https://github.com/thinking-machines-lab/manifolds/blob/main/src/main.py)
- [MuonAll (arXiv 2511.06086)](https://arxiv.org/abs/2511.06086) and [Saurabh750/optimizer: muon.py](https://github.com/Saurabh750/optimizer/blob/main/muon.py)
- [Fantastic Pretraining Optimizers and Where to Find Them (arXiv 2509.02046)](https://arxiv.org/abs/2509.02046)
- [Different Layers, Different Manifolds (arXiv 2606.13276)](https://arxiv.org/abs/2606.13276), [code](https://github.com/kiratoyoshihara/module-wise-manifold-muon), [summary notes](https://en.papernotes.org/ICML2026/llm_pretraining/different_layers_different_manifolds_module-wise_weight-space_geometry_in_transf/)
- [Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales (arXiv 2512.05620)](https://arxiv.org/abs/2512.05620)
- [Benchmarking Optimizers for Large Language Model Pretraining (arXiv 2509.01440)](https://arxiv.org/abs/2509.01440)
- [Practical Efficiency of Muon for Pretraining (arXiv 2505.02222)](https://arxiv.org/abs/2505.02222)
- [Transformers v5 MoE weight layout (VeOmni docs)](https://veomni.readthedocs.io/en/latest/transformers_v5/transformers_v5_moe_weight_loading.html)
- [CIFAR-5m dataset (preetum/cifar5m)](https://github.com/preetum/cifar5m)
- [The Deep Bootstrap Framework (arXiv 2010.08127)](https://arxiv.org/abs/2010.08127)
- [Berkeley EECS 182/282A Fall 2025 course policies](https://berkeley-cs182.github.io/fa25/policies)
