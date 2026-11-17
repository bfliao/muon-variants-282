"""Data loading for the Muon-variants studies (design.md §3.5).

FineWeb-Edu LM data: nanoGPT-style little-endian uint16 memmaps produced by
``scripts/prepare_fineweb.py``. CIFAR-5m: uint8 HWC .npy splits produced by
``scripts/prepare_cifar5m.py``, held as device tensors (airbench pattern,
no DataLoader).

Seed contract: every iterator here is a pure function of (seed, steps, ...) —
the same seed gives the same order for every optimizer. All randomness comes
from a locally constructed ``numpy.random.Generator``; global RNG state is
never read or written.

Prefix contract (budget-curve requirement): each training iterator draws ONE
seed-determined permutation of a FIXED pool (the first LM_WINDOW_POOL
non-overlapping windows of train.bin / all 1,802,240 training images) and
every run consumes a prefix of it. A 1x run's data order is therefore exactly
the first half of the 2x run's order for the same seed, and this holds for
every budget multiplier. Because the pool sizes are fixed constants — never
derived from file sizes — the order at a given seed is identical on every
machine and survives a rebuild of train.bin with more tokens.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Production constants (tests shrink the world via the explicit size
# parameters on FineWeb(...) / Cifar5m(...), never by editing these).
# ---------------------------------------------------------------------------

FINEWEB_MIN_TRAIN_TOKENS = 45_000_000   # 2x AdamW budget needs 40M
FINEWEB_VAL_TOKENS = 2_000_000          # exactly; first 0.5M = periodic eval set
FINEWEB_VOCAB_SIZE = 4096               # incl. the <|endoftext|> special token
FINEWEB_DTYPE = np.dtype("<u2")         # little-endian uint16

# The LM training permutation is over EXACTLY this fixed pool of
# non-overlapping 257-token windows: all windows needed for the 2x budget,
# 4,882 steps x 32 seqs/step (design.md §3.5). Fixing the pool (instead of
# using len(train.bin) // 257) makes the data order at a given seed
# independent of the file size, so a rebuilt or over-provisioned train.bin
# never silently changes any run's order (paired cross-machine comparisons).
LM_WINDOW_POOL = 4_882 * 32             # = 156_224 windows

CIFAR_N_TRAIN = 1_802_240               # C[:1_802_240], C = part0 ‖ part1
CIFAR_N_VAL = 10_000                    # C[1_802_240:1_812_240]
CIFAR_N_TEST = 10_000                   # C[1_812_240:1_822_240]

# Standard CIFAR-10 per-channel normalization constants (train-set mean/std of
# CIFAR-10 in RGB order), applied to CIFAR-5m as fixed constants of this module.
CIFAR_MEAN: tuple[float, float, float] = (0.49139968, 0.48215827, 0.44653124)
CIFAR_STD: tuple[float, float, float] = (0.24703233, 0.24348505, 0.26158768)


# ---------------------------------------------------------------------------
# FineWeb-Edu
# ---------------------------------------------------------------------------

class FineWeb:
    """Memmapped FineWeb-Edu token data; everything validated on open.

    Expects ``root`` to hold train.bin, val.bin, tokenizer.json and meta.json
    as written by scripts/prepare_fineweb.py.
    """

    def __init__(self, root: Path | str,
                 min_train_tokens: int = FINEWEB_MIN_TRAIN_TOKENS,
                 val_tokens: int = FINEWEB_VAL_TOKENS) -> None:
        self.root = Path(root)
        for name in ("train.bin", "val.bin", "tokenizer.json", "meta.json"):
            if not (self.root / name).is_file():
                raise FileNotFoundError(self.root / name)

        self.meta: dict = json.loads((self.root / "meta.json").read_text())
        for key in ("version", "n_train", "n_val", "vocab_size",
                    "tokenizer_sha256", "source"):
            if key not in self.meta:
                raise ValueError(f"meta.json missing key {key!r}")

        self.train = self._open_bin("train.bin")
        self.val = self._open_bin("val.bin")

        if len(self.train) < min_train_tokens:
            raise ValueError(
                f"train.bin has {len(self.train)} tokens, "
                f"need at least {min_train_tokens}")
        if len(self.val) != val_tokens:
            raise ValueError(
                f"val.bin has {len(self.val)} tokens, need exactly {val_tokens}")
        if self.meta["n_train"] != len(self.train) or self.meta["n_val"] != len(self.val):
            raise ValueError("meta.json token counts do not match the .bin files")

        vocab = int(self.meta["vocab_size"])
        for name, arr in (("train.bin", self.train), ("val.bin", self.val)):
            if int(arr.max()) >= vocab:
                raise ValueError(
                    f"{name} contains token id {int(arr.max())} >= vocab size {vocab}")

        tok_sha = hashlib.sha256((self.root / "tokenizer.json").read_bytes()).hexdigest()
        if tok_sha != self.meta["tokenizer_sha256"]:
            raise ValueError("meta.json tokenizer_sha256 does not match tokenizer.json")

    def _open_bin(self, name: str) -> np.memmap:
        path = self.root / name
        size = path.stat().st_size
        if size == 0 or size % FINEWEB_DTYPE.itemsize != 0:
            raise ValueError(f"{name} size {size} is not a whole number of uint16 tokens")
        return np.memmap(path, dtype=FINEWEB_DTYPE, mode="r")

    def version(self) -> str:
        return self.meta["version"]


def lm_batches(fw: FineWeb, seed: int, steps: int, batch_seqs: int = 32,
               seq_len: int = 256, pool_windows: int = LM_WINDOW_POOL
               ) -> Iterator[tuple[Tensor, Tensor]]:
    """Seed-shuffled training batches over non-overlapping (seq_len+1)-token
    windows of train.bin: window i covers tokens [i*(seq_len+1), (i+1)*(seq_len+1));
    x = first seq_len tokens, y = the same window shifted by one (seq_len
    predictions). One permutation of the FIXED pool of the first
    ``pool_windows`` windows per seed (default LM_WINDOW_POOL = the 2x
    budget's 156,224), so the order at a given seed does not depend on the
    file size; runs take a prefix, so no window repeats within a run and the
    1x order prefixes the 2x order.
    """
    win = seq_len + 1
    n_available = len(fw.train) // win
    if n_available < pool_windows:
        raise ValueError(
            f"train.bin holds only {n_available} windows of {win} tokens; "
            f"the fixed permutation pool needs {pool_windows}")
    need = steps * batch_seqs
    if need > pool_windows:
        raise ValueError(
            f"run needs {need} windows but the pool only has {pool_windows}; "
            "a single pass admits no repeats")
    rng = np.random.default_rng(seed)        # local RNG, never global state
    perm = rng.permutation(pool_windows)
    for step in range(steps):
        idx = perm[step * batch_seqs:(step + 1) * batch_seqs]
        buf = np.empty((batch_seqs, win), dtype=np.int64)
        for row, w in enumerate(idx):
            start = int(w) * win
            buf[row] = fw.train[start:start + win]
        t = torch.from_numpy(buf)
        yield t[:, :seq_len], t[:, 1:]


def lm_val_batches(fw: FineWeb, n_tokens: int, batch_seqs: int = 32,
                   seq_len: int = 256) -> Iterator[tuple[Tensor, Tensor]]:
    """Deterministic, unshuffled eval batches over the FIRST n_tokens of
    val.bin (500_000 = periodic eval set, 2_000_000 = the whole file; the
    small set is a prefix of the large one). The final partial batch is kept.
    """
    if n_tokens > len(fw.val):
        raise ValueError(f"requested {n_tokens} val tokens, file has {len(fw.val)}")
    win = seq_len + 1
    n_windows = n_tokens // win
    for start_w in range(0, n_windows, batch_seqs):
        rows = min(batch_seqs, n_windows - start_w)
        buf = np.empty((rows, win), dtype=np.int64)
        for row in range(rows):
            start = (start_w + row) * win
            buf[row] = fw.val[start:start + win]
        t = torch.from_numpy(buf)
        yield t[:, :seq_len], t[:, 1:]


# ---------------------------------------------------------------------------
# CIFAR-5m
# ---------------------------------------------------------------------------

class Cifar5m:
    """CIFAR-5m splits as device tensors (uint8 HWC images, int64 labels).

    Expects ``root`` to hold {train,val,test}.npy, {train,val,test}_labels.npy
    and meta.json as written by scripts/prepare_cifar5m.py.
    """

    def __init__(self, root: Path | str, device: torch.device,
                 n_train: int = CIFAR_N_TRAIN, n_val: int = CIFAR_N_VAL,
                 n_test: int = CIFAR_N_TEST) -> None:
        self.root = Path(root)
        self.device = device
        self.meta: dict = json.loads((self.root / "meta.json").read_text())
        if "version" not in self.meta:
            raise ValueError("meta.json missing key 'version'")

        expected = {"train": n_train, "val": n_val, "test": n_test}
        for split, n in expected.items():
            x = np.load(self.root / f"{split}.npy")
            y = np.load(self.root / f"{split}_labels.npy")
            if x.dtype != np.uint8 or x.shape != (n,) + (32, 32, 3):
                raise ValueError(
                    f"{split}.npy: expected uint8 ({n}, 32, 32, 3), "
                    f"got {x.dtype} {x.shape}")
            if y.dtype != np.int64 or y.shape != (n,):
                raise ValueError(
                    f"{split}_labels.npy: expected int64 ({n},), got {y.dtype} {y.shape}")
            setattr(self, f"x_{split}", torch.from_numpy(x).to(device))
            setattr(self, f"y_{split}", torch.from_numpy(y).to(device))

    def version(self) -> str:
        return self.meta["version"]


def normalize(x_uint8: Tensor) -> Tensor:
    """fp32, /255, per-channel CIFAR-10 mean/std, flattened to (batch, 3072).

    Input: uint8 (batch, 32, 32, 3) HWC on any device. No augmentation.
    """
    x = x_uint8.to(torch.float32) / 255.0
    mean = torch.tensor(CIFAR_MEAN, dtype=torch.float32, device=x.device)
    std = torch.tensor(CIFAR_STD, dtype=torch.float32, device=x.device)
    x = (x - mean) / std                      # broadcasts over the channel dim
    return x.reshape(x.shape[0], -1)


def cifar_batches(ds: Cifar5m, seed: int, steps: int, batch: int = 256
                  ) -> Iterator[tuple[Tensor, Tensor]]:
    """Seed-shuffled single-pass training batches: one permutation of all
    training indices per seed, runs take a prefix (1x = 901,120 images =
    3,520 steps; the 1x order prefixes the 2x order; no index repeats).
    Yields (normalized fp32 (batch, 3072), int64 labels) on ds.device.
    """
    n = ds.x_train.shape[0]
    need = steps * batch
    if need > n:
        raise ValueError(
            f"run needs {need} images but the train split only has {n}; "
            "a single pass admits no repeats")
    rng = np.random.default_rng(seed)        # local RNG, never global state
    perm = rng.permutation(n)
    for step in range(steps):
        idx = torch.from_numpy(perm[step * batch:(step + 1) * batch].copy()
                               ).to(ds.device)
        yield normalize(ds.x_train[idx]), ds.y_train[idx]
