# Deviations log

What / why / who approved. Clarifications approved at gates live in
docs/design.md ("Clarifications" sections) and PLAN.md's Decisions log; this file
records deviations and pending discrepancies.

| # | What | Why | Status / approved by |
| --- | --- | --- | --- |
| 1 | Stage B weight decays {0, 0.3×default, 3×default}, while PLAN.md Phase II step 2 still reads {0, default, 3×default}. | Paul's Gate-0 answer (newer, explicit: "the default was already run in Stage A"). | Following Paul's message; PLAN.md line to be fixed by Paul. Flagged at Gate 0 follow-up. |
| 2 | Faithful-settings reruns are three (Muon bf16-NS, MuonAll 6-step bf16, Manifold 100 dual steps), while PLAN.md's Decisions-log row names only MuonAll and Manifold. | Paul's Gate-0 answer adds Muon-bf16 as the third, covering Moonlight too. | Following Paul's message; PLAN.md row to be fixed by Paul. |
| 3 | Python pinned to 3.12 (uv defaulted to 3.11). | Vendored third_party/manifolds/src/main.py uses PEP-701 f-strings (nested quotes), a Python ≥3.12 feature; 3.11 failed with SyntaxError. Environment detail, not a protocol change. | Lead engineer; reported at Gate 1. |
| 4 | muonarch/msign.py is the vendored Polar Express algorithm with compute dtype parameterized (default fp32) rather than a byte-identical copy. | PLAN.md requires fp32 matrix iterations in our code (bf16 only in faithful reruns); the vendored msign hardcodes bf16. Identical coefficients and operation order, asserted by test. third_party/ copy remains untouched. | Lead engineer per PLAN.md precision row; reported at Gate 1. |
| 5 | Six PLAN.md lines contradict the Gate-0 answers or the new two-phase-tuning row (full list: docs/design.md §9 "Noted discrepancies"). | PLAN.md edited piecemeal at Gate 0; Gate-0 answers treated as newest word. | Text fixes pending Paul's next PLAN.md pass. |
| 6 | Warmup is NOT tuned for any optimizer: the warmup sweep listed in PLAN.md's Phase I coordinate-descent order is removed; warmup fixed at 5% of the 1× budget everywhere, including the budget curve. Deliberate departure from Wen et al., whose protocol tunes warmup. | Paul's Gate-1 answer: the 12-trial budget covers lr and wd only; warmup is a shared fixed setting. | Approved by Paul at Gate 1; PLAN.md:239 to be fixed by Paul. |
| 7 | Manifold Muon FLOPs count the msign calls actually made per step (including the extra ascent call when early stop fires, plus the retraction), reported by the optimizer itself, rather than deriving counts from logged dual steps. | Paul's Gate-1 answer; avoids the off-by-one found in review. | Approved by Paul at Gate 1. |
