# Main results

Winners are **bold** only where the difference to the runner-up passes the seed rule (3-seed sign+2*std seed rule: a heuristic, not a test).
Study A and Study B are reported separately; no ranking is compared across the two studies.

## Study A — MLP width study (CIFAR-5m)

### mlp128

| optimizer | final val loss (mean±std) | token speedup | FLOP speedup | LR basin (factors of 2) | divergence rate |
| --- | --- | --- | --- | --- | --- |
| adamw | 0.8881 ± 0.0062 | [0.92, 1.08] | [0.92, 1.08] | 1.00 | 0.0% |
| muon | 0.8073 ± 0.0068 | [1.76, 2.16]† | [0.64, 0.79]† | 0.00 | 0.0% |
| moonlight | 0.8201 ± 0.0036 | [1.64, 1.89] | [0.60, 0.69] | 1.00 | 0.0% |
| manifold | 0.9139 ± 0.0045 | [0.75, 0.85] | [0.04, 0.04] | 1.00 | 0.0% |
| **muonall** | 0.7925 ± 0.0043 | [2.06, 2.43]† | [0.75, 0.88]† | 0.00 | 4.8% |

### mlp256

| optimizer | final val loss (mean±std) | token speedup | FLOP speedup | LR basin (factors of 2) | divergence rate |
| --- | --- | --- | --- | --- | --- |
| adamw | 0.8193 ± 0.0018 | [0.99, 1.04] | [0.99, 1.04] | 0.00 | 0.0% |
| muon | 0.7227 ± 0.0021 | [2.17, 2.33]† | [0.47, 0.50]† | 1.00 | 0.0% |
| moonlight | 0.7247 ± 0.0038 | [2.09, 2.30]† | [0.45, 0.49]† | 0.00 | 0.0% |
| manifold | 0.8513 ± 0.0043 | [0.76, 0.83] | [0.02, 0.02] | 0.00 | 0.0% |
| **muonall** | 0.7120 ± 0.0025 | [2.37, 2.58]† | [0.51, 0.55]† | 0.00 | 4.8% |

### mlp512

| optimizer | final val loss (mean±std) | token speedup | FLOP speedup | LR basin (factors of 2) | divergence rate |
| --- | --- | --- | --- | --- | --- |
| adamw | 0.7831 ± 0.0041 | [0.96, 1.04] | [0.96, 1.04] | 0.00 | 0.0% |
| muon | 0.6761 ± 0.0034 | [2.28, 2.47]† | [0.26, 0.28]† | 0.00 | 0.0% |
| moonlight | 0.6667 ± 0.0049 | [2.41, 2.71]† | [0.27, 0.31]† | 0.00 | 0.0% |
| manifold | 0.7925 ± 0.0045 | [0.89, 0.97] | [0.01, 0.01] | 0.00 | 0.0% |
| muonall | 0.6583 ± 0.0047 | [2.59, 2.88]† | [0.29, 0.33]† | 0.00 | 4.2% |

## Study B — dense vs MoE transformer (FineWeb-Edu)

### dense

| optimizer | final val loss (mean±std) | token speedup | FLOP speedup | LR basin (factors of 2) | divergence rate |
| --- | --- | --- | --- | --- | --- |
| adamw | 4.0366 ± 0.0266 | [0.95, 1.16] | [0.95, 1.16] | 0.00 | 10.5% |
| muon | 3.9494 ± 0.0064 | [1.26, 1.41] | [1.22, 1.36] | 3.00 | 0.0% |
| **moonlight** | 3.8738 ± 0.0020 | [1.63, 1.80] | [1.58, 1.74] | 1.00 | 11.1% |
| manifold | 4.0673 ± 0.0147 | [0.90, 1.01] | [0.64, 0.72] | 1.00 | 17.6% |
| muonall | 3.8961 ± 0.0061 | [1.50, 1.69] | [1.43, 1.61] | 2.00 | 0.0% |

† Extrapolated: for at least one (AdamW curve seed, optimizer seed) pair the AdamW-equivalent data exceeds the largest measured AdamW budget (2×), so the speedup is read off the fitted curve beyond the data.

## Null results (indistinguishable after equal tuning)

Reported with the measured seed spread, so the reader sees how large an effect could have been detected (3-seed sign+2*std seed rule: a heuristic, not a test).

| study | optimizer | vs | mean diff | seed spread |
| --- | --- | --- | --- | --- |
| dense | manifold | adamw | +0.03068 | 0.02661 |


Edge rule fired 0 time(s); 14 extra edge trial(s) run.
  edge trials — dense/adamw: 1
  edge trials — dense/manifold: 2
  edge trials — dense/moonlight: 1
  edge trials — mlp128/manifold: 2
  edge trials — mlp256/manifold: 4
  edge trials — mlp512/manifold: 4
