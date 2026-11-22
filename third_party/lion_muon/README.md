# lion-muon (vendored reference, test oracle only)

- repo: https://github.com/brain-lab-research/lion-muon
- commit: e9077a87196906deaa10dd194e8d17a3891f531c (e9077a8, 2026-10-27 23:14:31 +0500)
- license: MIT (see `LICENSE`)
- files: `LICENSE`, `src/optim/lion_muon.py` and the modules it imports at
  load time (`src/optim/muon.py`, `norm_diagnostics.py`, `schedule.py`,
  `utils.py`), vendored unmodified (byte-for-byte).

`muonarch` never imports from this directory. Only `tests/test_lionmuon.py`
does, to compare `muonarch.optim.LionMuon` against the reference `LionMuon`
(Algorithm 1: single process, non-adaptive, `nesterov=False`). The reference
hard-codes `device="cuda"` for its update buffer and runs Newton-Schulz in
bfloat16 under `torch.compile`; the test redirects the device to CPU and
turns `torch.compile` into a no-op from the outside, without editing these
files.
