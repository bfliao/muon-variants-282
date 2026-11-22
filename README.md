# Muon variants × architectures

Course project for Berkeley CS 282. We compare six optimizers — AdamW, Muon, Moonlight-Muon,
Manifold Muon, MuonAll and Lion-Muon — under one equal-tuning protocol, on MLPs of width 128/256/512
(CIFAR-5m) and on tiny Qwen3-style transformers, dense and MoE (FineWeb-Edu).

**Status (2026-11-21):** the MLP and dense-transformer studies are complete (3 seeds), including
Lion-Muon, added as a sixth optimizer (DEVIATIONS #15). The MoE
study has not run yet.

## Where things are

| Path | What |
|---|---|
| `PLAN.md` | The plan and its Decisions log. This is the source of truth. |
| `DEVIATIONS.md` | Every departure from the plan, with the reason. |
| `PROGRESS.md` | Running log of what was done and when. |
| `docs/team_briefing.pdf` | Short briefing: what we do, why, and what changed from the proposal. |
| `docs/manifold_audit.md` | Why Manifold Muon lags, and the checks that rule out a bug. |
| `docs/figures_spec.md` | Specification of the report's figures. |
| `paper/method.tex`, `paper/method.pdf` | Method section in NeurIPS format (two pages). Build: `cd paper && latexmk -pdf main.tex`. |
| `figures/` | The generated figures (PNG and PDF). |
| `tables/main.md`, `tables/hypotheses.csv` | Main results table and hypothesis verdicts. |
| `results/<study>/*.json` | One file per run: config, final losses, FLOPs. |
| `logs/<study>/*.jsonl` | One file per run: the training curve and diagnostics. |
| `muonarch/` | Code: optimizers, models, training loop, study definitions, analysis, figures. |
| `configs/` | One YAML per study. |
| `third_party/manifolds/`, `third_party/lion_muon/` | Reference code for Manifold Muon and Lion-Muon, vendored unmodified (test oracles). |

## Setup

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest -q          # about 4 minutes
```

## Rebuild the figures and tables (no training, no datasets needed)

The results and logs are in the repo, so this works on a fresh clone:

```
uv run python -m muonarch.figures     # writes figures/*.png and *.pdf
```

## Datasets

The datasets are not in git. Put them under `data/`. The code checks a content hash and refuses
to run on data that differs from ours.

**CIFAR-5m (MLP studies, 5.2 GB).** Downloaded from Google's public storage and split
deterministically:

```
uv run python scripts/prepare_cifar5m.py --out data/cifar5m
```

**FineWeb-Edu tokens (transformer studies, 94 MB).** Download our exact copy from Hugging Face
(public, no login needed):
<https://huggingface.co/datasets/Paulopasso/fineweb-edu-bpe4096-45m>

```
uv run hf download Paulopasso/fineweb-edu-bpe4096-45m --repo-type dataset --local-dir data/fineweb
```

It holds `train.bin` (45,045,743 tokens), `val.bin` (2,000,000 tokens), `tokenizer.json`
(4,096-token byte-level BPE) and `meta.json`; version `fineweb-edu-10bt-v1-5d037df91d97`.

`scripts/prepare_fineweb.py` can rebuild it from the raw text, but a different library version
can change the tokens, and the code would then refuse to run. Use the download.

## Running a study

Runs are resumable and keyed by a hash of their config; finished runs are never repeated.

```
uv run python -m muonarch.runner --study mlp128 --status     # what is done, what is pending
uv run python -m muonarch.runner --study mlp128 --dry-run    # list the jobs that would run
caffeinate -i uv run python -m muonarch.runner --study mlp128
```

Each MLP process loads CIFAR-5m into memory (about 10 GB), so run one MLP study at a time on a
laptop. Runs use fp32 on Apple-silicon GPUs. Talk to Paul before launching anything long: only
one person launches the long runs, so the protocol stays identical across runs.
