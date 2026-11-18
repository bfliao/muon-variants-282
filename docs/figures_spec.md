# Figure specification (v2, after the Gate-3 sweeps)

Replaces the eight-figure set in PLAN.md "Plots" (DEVIATIONS #14). Implemented in
`muonarch/figures.py`; data selection reuses `muonarch/analysis.py`. Output: `figures/*.png`
(200 dpi) and `figures/*.pdf`. Generate everything with `uv run python -m muonarch.figures`.

Design references: Wen et al. 2509.02046 (speedup, sensitivity), Qiu et al. 2512.05620 and
Yang et al. 2203.03466 (loss-vs-LR per width, optimum markers), Liu et al. 2502.16982 (fitted
curve with a labelled arrow), Semenov et al. 2509.01440 (small-multiple sweeps).

## House style (every figure)

- **Colour and marker follow the optimizer, in every figure** (validated all-pairs for
  colour-blind safety; the marker is the secondary encoding):

  | Optimizer | Label | Colour | Marker |
  |---|---|---|---|
  | adamw | AdamW | `#3a3a38` (neutral, the baseline) | circle `o` |
  | muon | Muon | `#2a78d6` | square `s` |
  | moonlight | Moonlight | `#eda100` | diamond `D` |
  | manifold | Manifold Muon | `#008300` | triangle up `^` |
  | muonall | MuonAll | `#e87ba4` | triangle down `v` |

- Optimizer order is always AdamW, Muon, Moonlight, MuonAll, Manifold Muon.
- Study labels: "MLP-128", "MLP-256", "MLP-512", "Dense transformer". Never raw ids.
- Width ramp (when width is the series): one hue, light to dark = 128, 256, 512
  (`#86b6ef`, `#2a78d6`, `#0d366b`).
- Text is always dark ink (`#0b0b0b` titles/labels, `#52514e` secondary), never the series colour.
- Lines 1.8–2 pt; markers ≥ 6 pt with a white edge (1 pt) so overlaps stay legible.
- Grid: solid hairlines, light grey (`#e6e5e0`), behind the data; no top/right spines.
- No dual axes. No suptitle that collides with panel titles: use `constrained_layout`, and check
  every rendered PNG for overlap.
- Legend: one per figure (not per panel), outside the plot area (top or right), with marker+line keys.
- Points sit exactly on their x value. Never offset (dodge) markers on a quantitative axis. If
  two series coincide, show it honestly: facet into small multiples, or annotate ("Muon = MuonAll").
- Uncertainty: show the 3 seeds (individual dots, or min–max band/whisker). Say "3 seeds" in the
  caption text placed in the figure only if short; otherwise leave for the report caption.
- Font sizes: 9 pt base, 10 pt panel titles, 8 pt ticks. Figures are sized for a two-column
  paper: full width 7.2 in, single column 3.5 in.
- Loss is "final validation loss" (lower is better); say "(lower is better)" on the axis once.
- Data: headline = `fresh_seed` runs (seeds 1–3, completed, 1× budget). Tuning data for a study =
  `analysis.headline_tuning_phases(df, study)` (cd where it exists). Never mix the inherited
  track (stage_a/stage_b/edge at MLP-256/512) into headline plots. Diverged runs are drawn as an
  `x` at the top of the panel, never dropped silently.

## Main-text figures

**F1 `f1_final_loss` — Headline results.** One row of 4 panels (the 4 studies), each a horizontal
dot plot: one row per optimizer; small dots = the 3 seeds, large marker = mean; a thin vertical
reference line at AdamW's mean; mean value printed at the right. Sorted in the fixed optimizer
order. x axis per panel (not shared; scales differ). Answers: who wins, by how much, vs seed noise.

**F2 `f2_speedup` — Speedup over AdamW.** 2 rows × 4 columns: top row token/image speedup, bottom
row FLOP speedup; columns = studies. Horizontal dot + whisker (min–max over the 3×3 seed matrix
from `token_speedup_range` / `flop_speedup_range`), log x, vertical line at 1 labelled once
"AdamW". Value label "2.3×" beside each marker. Shade x < 1 very lightly ("slower than AdamW").
Shared x range within a row. Row labels at the left ("Per token / image", "Per FLOP").

**F3 `f3_budget_curve` — How speedup is measured.** 4 panels (studies). x = training data as a
multiple of the 1× budget (log2: 0.125, 0.25, 0.5, 1, 2), y = AdamW final val loss. AdamW points
(per seed, grey dots) + the fitted curve L(D)=αD^−β+γ (dashed; use `analysis.curve_points` /
`fit_budget_curve`; fit shown for the seed-mean or one line per seed, whichever the analysis
uses — state which). For each other optimizer: its mean final loss at 1× as a marker at x=1, and a
thin horizontal leader from that marker to the AdamW curve, ending in a marker on the curve, with
the equivalent-data value labelled ("1.7×"). Manifold Muon lands left of 1. This is the Moonlight
Fig 1a / Wen Fig 3 idea.

**F4 `f4_loss_curves_tokens` — Training curves.** 4 panels; val loss vs images/tokens, mean line +
min–max band over fresh seeds; y limited to the informative range (clip the first evals); inset or
second row zoomed on the last 25%. Direct end labels where they do not collide, else legend only.
**F4b `f4b_loss_curves_flops`**: same, x = cumulative FLOPs (log x), so Manifold Muon's cost is
visible without squashing the others.

**F5 `f5_width_transfer` — Does the optimal LR move with width? (H4)** One row of 5 panels (one
per optimizer). x = learning rate (log2), y = final val loss, one curve per MLP width (width ramp,
light→dark), headline tuning data (seed 0 lr sweep: `cd` rows with `cd_sweep == "lr"` plus the
accepted centre), a filled marker with dark edge on each curve's minimum, and a short vertical
tick or arrow annotation showing the shift between the 128 and 512 optimum ("÷2" for AdamW,
"no shift" where equal). Shared y within the figure is not required (losses differ by width):
instead plot **loss minus that curve's minimum** in a second row so basin shapes are comparable
(TP5 variant), y clipped at ~0.15. **F5b `f5b_optimal_lr_vs_width`**: small-multiples summary,
one panel per optimizer (shared log2 y), x = width (ticks exactly 128/256/512), re-tuned optimum
(solid, filled) and inherited-track optimum (dashed, hollow; `analysis.inherited_best_lr`). No
dodging: the two tracks coincide where they coincide.

**F6 `f6_hparam_sensitivity_<study>` — Hyperparameter sensitivity (coordinate descent).** One
figure per study with CD data (mlp128, mlp256, mlp512, dense). Grid: rows = optimizers (5),
columns = hyperparameters (learning rate, weight decay, momentum / β1, β2, gradient clip).
Each cell: x = the hyperparameter value (log x for lr; categorical-evenly-spaced for wd incl. 0,
momentum, β2, clip incl. "none"), y = final val loss − best loss for that optimizer (≥ 0), points
joined by a line in the optimizer colour; the accepted value marked with a star; a light grey
band from 0 to the acceptance threshold (2 × pilot seed std for that study — get it from the
pilot rows / `studies` helper) labelled once "noise". Cells for hyperparameters an optimizer does
not have (e.g. β2 for Muon, wd for Manifold) are blank with a small grey "n/a". y clipped (same
limit across a row) with out-of-range points drawn as `^` at the top edge, diverged as `x`.
Second-pass CD points included. Wen et al. show this only as tables; we show the grid.

**F7 `f7_compute_cost` — Where the FLOPs go.** 4 panels (studies), horizontal stacked bars per
optimizer: model forward+backward (light neutral) + optimizer (the optimizer colour), in units of
"× the model's forward+backward cost" (so the model segment is 1 everywhere), total labelled at
the bar end ("3.0×"). Log x only if Manifold Muon makes linear unreadable; if so use a linear axis
with a broken/annotated Manifold bar or put Manifold's number as a label with a clipped bar and
`»` marker. From measured `flops_model` / `flops_optimizer` of fresh-seed runs.

**F8 `f8_rank_summary` — Ranking across studies.** Keep the bump chart idea: x = study, y = rank,
one line per optimizer with its marker, hollow marker where the gap to the next-better optimizer
fails the seed rule. Direct labels at the right end instead of a legend.

## Appendix figures

**A1 `a1_lr_sensitivity` — Loss vs LR, all optimizers.** 4 panels; x = LR / that optimizer's best
LR (log2, vertical line at 1), y = loss − best (clipped), one line per optimizer; diverged `x`.
Shows basin width directly (H6). A second row with the raw LR axis.

**A2 `a2_inherited_vs_retuned` — MLP tuning tracks (DEVIATIONS #12).** MLP-256 and MLP-512: per
optimizer, final loss of the re-tuned winner vs the inherited-settings winner (seed 0, paired
dots joined by a line), plus the LR each chose. Dumbbell plot.

**A3 `a3_robustness` — Robustness checks.** (a) Faithful-settings reruns: per study, standard
(seed 1 fresh run) vs faithful (Muon bf16 NS, MuonAll 6-step bf16, Manifold cap 100) as paired
dots with the 3-seed min–max band behind. (b) Init ablation: Muon vs Muon with Stiefel init,
3 seeds each, per study. (c) AdamW 2× with half LR vs 2× (budget_halflr) where present.

**A4 `a4_loss_vs_accuracy` — Loss and accuracy agree (MLP).** Scatter of final val loss vs test
accuracy for every completed 1× MLP run (tuning runs small and pale, fresh seeds large), coloured
by optimizer, one panel per width, with ±1 SE of accuracy (0.45 pt) shown as a reference bar.

**A5 `a5_manifold_audit` — Why Manifold Muon lags.** (a) MLP-128 ablations from
`logs/diagnostics/manifold_audit/summary.jsonl` (labels from docs/manifold_audit.md): horizontal
dot plot of val loss, reference lines for Muon and AdamW (seed 0 / fresh mean, state which),
sorted by loss. Forward-scale runs are in `summary.jsonl` under names `scale*`; skip the diverged
one with an `x`. (b) Final spectral norms: strip/box per optimizer of σ_max over all Muon-group
matrices (dense), log y, horizontal line at 1 — Manifold is exactly 1, others 1.6–27.

**A6 `a6_attention_logits` — Max attention logit over training (dense).** Mean ± min–max over
fresh seeds per optimizer. Single panel.

**A7 `a7_spectral_norms_by_layer` — σ_max by matrix type (dense).** Small multiples: one panel
per matrix type (q, k, v, o, gate, up, down), x = layer 0–5, y = σ_max (log), line per optimizer,
mean over fresh seeds.

**A8 `a8_divergence` — Stability.** Per study, per optimizer: the LR range trained without
divergence, as a horizontal span on a log2 LR axis, with the best LR marked and diverged LRs `x`.

MoE figures (expert entropy, dense-vs-MoE interaction) are not drawn until MoE fresh seeds exist;
`figures.py` must skip them cleanly and say so on stdout.

## Engineering

- `muonarch/figures.py`: a `style` section (palette, markers, labels, rcParams context manager),
  one function per figure `fig_<id>(df, logs, outdir) -> list[Path]`, small pure data helpers
  returning DataFrames (tested), and `main()` that builds all and prints one line per figure
  (written or skipped + reason). No training, no writes outside `outdir`.
- The old `analysis.figure_1..8` stay (tests depend on them) but are no longer the report figures.
- Tests (`tests/test_figures.py`): style table complete and fixed; each data helper on synthetic
  frames (reuse fixtures in tests/test_analysis.py); every figure function renders on the
  synthetic fixtures without error and writes its files; markers in F5b sit exactly on the widths;
  inherited-track rows never enter headline helpers; parked/partial runs never counted.
