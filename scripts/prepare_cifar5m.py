#!/usr/bin/env python
"""Download CIFAR-5m parts 0 and 1 and write the fixed project splits.

Usage:
    uv run python scripts/prepare_cifar5m.py --out data/cifar5m

Per design.md §3.5 / Gate-0 answer 1: with C = part0 ‖ part1 (file order),
    train = C[:1_802_240]        (uint8 HWC images + int64 labels)
    val   = C[1_802_240:1_812_240]
    test  = C[1_812_240:1_822_240]
written as {train,val,test}.npy / {split}_labels.npy plus meta.json with a
content-derived version string.

Resumable: finished outputs are skipped; interrupted .npz downloads resume
with an HTTP Range request; downloaded parts are verified (keys, shapes,
dtypes) before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

PART_URLS = [
    "https://storage.googleapis.com/gresearch/cifar5m/part0.npz",
    "https://storage.googleapis.com/gresearch/cifar5m/part1.npz",
]
N_TRAIN = 1_802_240
N_VAL = 10_000
N_TEST = 10_000
SOURCE = "cifar5m/part0+part1 (storage.googleapis.com/gresearch/cifar5m)"


# ---------------------------------------------------------------- pure logic

def slice_parts(parts: list[tuple[np.ndarray, np.ndarray]],
                n_train: int = N_TRAIN, n_val: int = N_VAL, n_test: int = N_TEST
                ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Slice the concatenation C = part0 ‖ part1 into the fixed contiguous
    train/val/test splits. parts = [(X0, Y0), (X1, Y1)]; returns
    {"train": (X, Y), "val": ..., "test": ...} with uint8 images, int64 labels.
    """
    need = n_train + n_val + n_test
    x = np.concatenate([p[0] for p in parts], axis=0)
    y = np.concatenate([p[1] for p in parts], axis=0).astype(np.int64)
    if len(x) < need:
        raise ValueError(f"parts hold {len(x)} images, need {need}")
    x = np.ascontiguousarray(x[:need], dtype=np.uint8)
    y = y[:need]
    return {
        "train": (x[:n_train], y[:n_train]),
        "val": (x[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        "test": (x[n_train + n_val:need], y[n_train + n_val:need]),
    }


# ---------------------------------------------------------------- download

def download(url: str, dest: Path) -> None:
    """Download url to dest, resuming a partial .part file via Range."""
    if dest.is_file():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    offset = tmp.stat().st_size if tmp.is_file() else 0
    req = urllib.request.Request(url)
    if offset:
        req.add_header("Range", f"bytes={offset}-")
        print(f"resuming {dest.name} at {offset / 1e6:.0f} MB")
    with urllib.request.urlopen(req) as resp:
        if offset and resp.status != 206:   # server ignored Range: restart
            offset = 0
        mode = "ab" if offset else "wb"
        total = resp.headers.get("Content-Length")
        total = offset + int(total) if total else None
        done, t0 = offset, time.time()
        with open(tmp, mode) as f:
            while chunk := resp.read(1 << 22):
                f.write(chunk)
                done += len(chunk)
                if time.time() - t0 > 5:
                    pct = f" ({100 * done / total:.0f}%)" if total else ""
                    print(f"  {dest.name}: {done / 1e9:.2f} GB{pct}", flush=True)
                    t0 = time.time()
    tmp.rename(dest)
    print(f"downloaded {dest.name}")


def load_part(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and verify one CIFAR-5m part (keys X uint8 (N,32,32,3), Y (N,))."""
    with np.load(path) as z:
        if "X" not in z or "Y" not in z:
            raise ValueError(f"{path.name}: expected keys X and Y, got {z.files}")
        x, y = z["X"], z["Y"]
    if x.ndim != 4 or x.shape[1:] != (32, 32, 3) or x.dtype != np.uint8:
        raise ValueError(f"{path.name}: X has shape {x.shape} dtype {x.dtype}, "
                         "expected uint8 (N, 32, 32, 3)")
    if y.shape != (len(x),):
        raise ValueError(f"{path.name}: Y shape {y.shape} does not match X ({len(x)})")
    print(f"{path.name}: {len(x):,} images OK")
    return x, y


# ---------------------------------------------------------------- main

def outputs(out: Path) -> list[Path]:
    names = [f"{s}{k}.npy" for s in ("train", "val", "test") for k in ("", "_labels")]
    return [out / n for n in names] + [out / "meta.json"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/cifar5m"))
    args = ap.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    if all(p.is_file() for p in outputs(out)):
        print(f"all outputs already present in {out}; nothing to do")
        return

    parts = []
    for url in PART_URLS:
        dest = out / url.rsplit("/", 1)[1]
        download(url, dest)
        parts.append(load_part(dest))

    print("slicing C = part0 ‖ part1 ...")
    splits = slice_parts(parts)
    del parts

    sha = hashlib.sha256()
    for split in ("train", "val", "test"):
        x, y = splits[split]
        np.save(out / f"{split}.npy", x)
        np.save(out / f"{split}_labels.npy", y)
        sha.update(np.ascontiguousarray(x).data)
        sha.update(np.ascontiguousarray(y).data)
        print(f"wrote {split}: {len(x):,} images")

    meta = {
        "version": f"cifar5m-v1-{sha.hexdigest()[:12]}",
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "source": SOURCE,
        "slices": "C = part0||part1; train=C[:1802240], val=C[1802240:1812240], "
                  "test=C[1812240:1822240]",
        "content_sha256": sha.hexdigest(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote meta.json, version {meta['version']}")
    print("You may now delete part0.npz/part1.npz to reclaim disk, or keep "
          "them for reproducibility.")


if __name__ == "__main__":
    sys.exit(main())
