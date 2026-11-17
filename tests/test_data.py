"""Offline tests for muonarch/data.py and the pure slicing logic of
scripts/prepare_cifar5m.py.

The world is shrunk via the explicit size parameters on FineWeb(...) and
Cifar5m(...) (production defaults live in module constants and are untouched):
tiny synthetic train.bin/val.bin memmaps and .npy splits are built in tmp_path,
so no test ever downloads anything.

Trick used throughout: synthetic FineWeb tokens are ``token[j] = j % 251``
(251 prime, < vocab), so the first token of non-overlapping window w is
(w*257) % 251 = (6*w) % 251, invertible via 42 = 6^-1 mod 251. That lets us
recover *which* window each batch row came from and also check the exact x/y
shift. Synthetic CIFAR images carry their global index in the first two bytes
of pixel (0,0), recoverable after normalize().
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from muonarch.data import (
    CIFAR_MEAN,
    CIFAR_STD,
    Cifar5m,
    FineWeb,
    cifar_batches,
    lm_batches,
    lm_val_batches,
    normalize,
)

MOD = 251          # prime < any vocab we use
INV6 = 42          # 6 * 42 = 252 = 1 (mod 251)
WIN = 257          # seq_len + 1 for the default seq_len=256
POOL = 64          # small-world window pool (production: data.LM_WINDOW_POOL)


def batches(fw_, seed, steps, batch_seqs=8, pool=POOL):
    """lm_batches with the shrunk pool threaded through."""
    return lm_batches(fw_, seed, steps, batch_seqs=batch_seqs,
                      pool_windows=pool)


# ---------------------------------------------------------------- fixtures

def write_fineweb(root: Path, n_train: int, n_val: int, vocab_size: int = 4096,
                  train_tokens: np.ndarray | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if train_tokens is None:
        train_tokens = (np.arange(n_train) % MOD).astype("<u2")
    else:
        train_tokens = train_tokens.astype("<u2")
        n_train = len(train_tokens)
    val_tokens = ((np.arange(n_val) + 7) % MOD).astype("<u2")
    (root / "train.bin").write_bytes(train_tokens.tobytes())
    (root / "val.bin").write_bytes(val_tokens.tobytes())
    tok_bytes = b'{"model": {"type": "BPE"}, "note": "dummy test tokenizer"}'
    (root / "tokenizer.json").write_bytes(tok_bytes)
    meta = {
        "version": "fineweb-test-v1",
        "n_train": n_train,
        "n_val": n_val,
        "vocab_size": vocab_size,
        "tokenizer_sha256": hashlib.sha256(tok_bytes).hexdigest(),
        "source": "fineweb-edu/sample-10BT",
        "tokenizer_trained_on": "train_docs_only",
    }
    (root / "meta.json").write_text(json.dumps(meta))
    return root


@pytest.fixture
def fw(tmp_path):
    # 64 full windows of 257 tokens; 8 val windows.
    root = write_fineweb(tmp_path / "fineweb", n_train=WIN * 64, n_val=WIN * 8)
    return FineWeb(root, min_train_tokens=WIN * 64, val_tokens=WIN * 8)


def window_ids(x: torch.Tensor) -> list[int]:
    """Recover window index from the first token of each row (see header)."""
    return [(int(t) * INV6) % MOD for t in x[:, 0]]


N_TRAIN_IMG, N_EVAL_IMG = 512, 16


def write_cifar(root: Path, n_train: int = N_TRAIN_IMG, n_eval: int = N_EVAL_IMG) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    n_total = n_train + 2 * n_eval
    ids = np.arange(n_total)
    x = np.zeros((n_total, 32, 32, 3), dtype=np.uint8)
    x[:, 0, 0, 0] = ids % 256          # every image unique by construction
    x[:, 0, 0, 1] = ids // 256
    y = (ids % 10).astype(np.int64)
    splits = {"train": slice(0, n_train),
              "val": slice(n_train, n_train + n_eval),
              "test": slice(n_train + n_eval, n_total)}
    for name, sl in splits.items():
        np.save(root / f"{name}.npy", x[sl])
        np.save(root / f"{name}_labels.npy", y[sl])
    (root / "meta.json").write_text(json.dumps({"version": "cifar-test-v1"}))
    return root


@pytest.fixture
def cifar(tmp_path):
    root = write_cifar(tmp_path / "cifar5m")
    return Cifar5m(root, torch.device("cpu"), n_train=N_TRAIN_IMG,
                   n_val=N_EVAL_IMG, n_test=N_EVAL_IMG)


def image_ids(x_norm: torch.Tensor) -> list[int]:
    """Recover global image ids from a normalized, flattened batch."""
    mean = torch.tensor(CIFAR_MEAN)
    std = torch.tensor(CIFAR_STD)
    # flatten of (32, 32, 3) is channel-last: flat[0] = pixel(0,0) ch0, flat[1] = ch1
    b0 = torch.round((x_norm[:, 0] * std[0] + mean[0]) * 255.0)
    b1 = torch.round((x_norm[:, 1] * std[1] + mean[1]) * 255.0)
    return [int(a) + 256 * int(b) for a, b in zip(b0, b1)]


# ---------------------------------------------------------------- FineWeb open

def test_fineweb_counts_and_dtype(tmp_path, fw):
    assert fw.train.dtype == np.dtype("<u2")
    assert fw.val.dtype == np.dtype("<u2")
    assert fw.meta["vocab_size"] == 4096
    assert fw.version() == "fineweb-test-v1"
    assert len(fw.train) == WIN * 64 and len(fw.val) == WIN * 8

    # too-small train.bin is rejected
    with pytest.raises(ValueError, match="train"):
        FineWeb(fw.root, min_train_tokens=WIN * 64 + 1, val_tokens=WIN * 8)
    # wrong val token count is rejected
    with pytest.raises(ValueError, match="val"):
        FineWeb(fw.root, min_train_tokens=WIN * 64, val_tokens=WIN * 8 + 1)

    # a token id >= vocab_size is rejected (vocab bound check)
    bad = (np.arange(WIN * 4) % MOD).astype("<u2")
    bad[10] = 4096
    root2 = write_fineweb(tmp_path / "fw_bad", n_train=0, n_val=WIN * 2,
                          train_tokens=bad)
    with pytest.raises(ValueError, match="vocab"):
        FineWeb(root2, min_train_tokens=WIN * 4, val_tokens=WIN * 2)


def test_fineweb_meta_mismatch_rejected(tmp_path):
    root = write_fineweb(tmp_path / "fw", n_train=WIN * 8, n_val=WIN * 2)
    meta = json.loads((root / "meta.json").read_text())
    meta["n_train"] = 123
    (root / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="meta"):
        FineWeb(root, min_train_tokens=WIN * 8, val_tokens=WIN * 2)


# ---------------------------------------------------------------- lm_batches

def test_lm_window_content_and_shift(fw):
    for x, y in batches(fw, seed=3, steps=2, batch_seqs=4):
        assert x.shape == (4, 256) and y.shape == (4, 256)
        assert x.dtype == torch.int64 and y.dtype == torch.int64
        # y is x shifted by one inside the same 257-token window
        assert torch.equal(x[:, 1:], y[:, :-1])
        for row, w in enumerate(window_ids(x)):
            start = w * WIN
            expect = torch.arange(start, start + WIN, dtype=torch.int64) % MOD
            assert torch.equal(x[row], expect[:256])
            assert torch.equal(y[row], expect[1:])


def test_lm_no_window_repeats_within_run(fw):
    seen = []
    for x, _ in batches(fw, seed=0, steps=8):  # all 64 windows
        seen += window_ids(x)
    assert len(seen) == 64 and len(set(seen)) == 64


def test_lm_prefix_consistency(fw):
    # 1x run's order is exactly the first half of the 2x run's order
    half = [window_ids(x) for x, _ in batches(fw, seed=11, steps=4)]
    full = [window_ids(x) for x, _ in batches(fw, seed=11, steps=8)]
    assert full[:4] == half


def test_lm_seed_determinism(fw):
    a = [window_ids(x) for x, _ in batches(fw, seed=5, steps=6)]
    b = [window_ids(x) for x, _ in batches(fw, seed=5, steps=6)]
    c = [window_ids(x) for x, _ in batches(fw, seed=6, steps=6)]
    assert a == b
    assert a != c
    # iterators never touch global RNG state: a global reseed changes nothing
    torch.manual_seed(0)
    np.random.seed(0)
    d = [window_ids(x) for x, _ in batches(fw, seed=5, steps=6)]
    assert d == a


def test_lm_rejects_overdraw(fw):
    with pytest.raises(ValueError, match="window"):
        next(batches(fw, seed=0, steps=9))  # 72 > pool of 64


def test_lm_fixed_pool_order_independent_of_file_size(tmp_path, fw):
    # same seed, same pool, LONGER train.bin -> identical order: the
    # permutation is over the fixed pool, never over len(file) // 257
    big_root = write_fineweb(tmp_path / "fw_big", n_train=WIN * 96, n_val=WIN * 8)
    fw_big = FineWeb(big_root, min_train_tokens=WIN * 96, val_tokens=WIN * 8)
    small = [window_ids(x) for x, _ in batches(fw, seed=13, steps=8)]
    big = [window_ids(x) for x, _ in batches(fw_big, seed=13, steps=8)]
    assert small == big

    # overdraw is judged against the pool, not the (larger) file
    with pytest.raises(ValueError, match="pool"):
        next(batches(fw_big, seed=0, steps=9))  # 72 > pool of 64 (file has 96)


def test_lm_rejects_file_smaller_than_pool(fw):
    # production default pool (LM_WINDOW_POOL) >> the tiny fixture
    with pytest.raises(ValueError, match="pool"):
        next(lm_batches(fw, seed=0, steps=1))
    with pytest.raises(ValueError, match="pool"):
        next(batches(fw, seed=0, steps=1, pool=65))  # file has 64 windows


# ------------------------------------------- prepare_fineweb.py resume logic

def load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_doc_caches(mod, out: Path, n_docs: int) -> tuple[int, int]:
    """Write consistent train/val jsonl caches for docs 0..n_docs-1 under the
    script's own split rule; returns (train_bytes, val_bytes) of the text."""
    tb = vb = 0
    with open(out / "train_docs.jsonl", "w", encoding="utf-8") as ft, \
         open(out / "val_docs.jsonl", "w", encoding="utf-8") as fv:
        for i in range(n_docs):
            text = f"document number {i} body"
            line = json.dumps({"text": text}) + "\n"
            if i % mod.VAL_EVERY == 0:
                fv.write(line)
                vb += len(text.encode())
            else:
                ft.write(line)
                tb += len(text.encode())
    return tb, vb


def test_prepare_fineweb_resume_reconciliation(tmp_path):
    mod = load_script("prepare_fineweb")
    out = tmp_path
    n_docs = 2 * mod.VAL_EVERY + 3  # 43 docs -> 3 val, 40 train
    tb, vb = write_doc_caches(mod, out, n_docs)

    # stale checkpoint (lags the caches) + torn final line from a SIGKILL
    (out / "collect_state.json").write_text(json.dumps(
        {"docs_seen": 7, "train_bytes": 1, "val_bytes": 1}))
    with open(out / "train_docs.jsonl", "a", encoding="utf-8") as f:
        f.write('{"text": "torn mid-wri')  # no newline

    state = mod.reconcile_collection(out)
    # ground truth from the files, not the stale checkpoint; torn line dropped
    assert state == {"docs_seen": n_docs, "train_bytes": tb, "val_bytes": vb}
    raw = (out / "train_docs.jsonl").read_bytes()
    assert raw.endswith(b"\n") and b"torn" not in raw
    # every remaining line parses (iter_texts would not crash)
    assert sum(1 for _ in mod.iter_texts(out / "train_docs.jsonl")) == n_docs - 3

    # idempotent on clean caches, and zero state on a fresh directory
    assert mod.reconcile_collection(out) == state
    assert mod.reconcile_collection(tmp_path / "fresh") == \
        {"docs_seen": 0, "train_bytes": 0, "val_bytes": 0}


def test_prepare_fineweb_rejects_inconsistent_caches(tmp_path):
    mod = load_script("prepare_fineweb")
    write_doc_caches(mod, tmp_path, 2 * mod.VAL_EVERY + 3)
    with open(tmp_path / "val_docs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": "extra val doc"}) + "\n")  # breaks the rule
    with pytest.raises(RuntimeError, match="inconsistent"):
        mod.reconcile_collection(tmp_path)


def test_prepare_fineweb_skips_when_outputs_complete(tmp_path, capsys):
    # lead deleted the jsonl caches + state; only final outputs remain ->
    # main() must do nothing (and must not try to stream anything)
    mod = load_script("prepare_fineweb")
    for name in ("train.bin", "val.bin"):
        (tmp_path / name).write_bytes(b"\x00\x00")
    (tmp_path / "meta.json").write_text("{}")
    mod.main(["--out", str(tmp_path)])
    assert "nothing to do" in capsys.readouterr().out


# ---------------------------------------------------------------- lm_val_batches

def test_lm_val_deterministic_and_prefix_rule(fw):
    full1 = [(x.clone(), y.clone()) for x, y in lm_val_batches(fw, WIN * 8, batch_seqs=3)]
    full2 = [(x.clone(), y.clone()) for x, y in lm_val_batches(fw, WIN * 8, batch_seqs=3)]
    assert len(full1) == len(full2)
    for (x1, y1), (x2, y2) in zip(full1, full2):
        assert torch.equal(x1, x2) and torch.equal(y1, y2)

    # unshuffled: window i covers val tokens [i*257, (i+1)*257)
    val = torch.from_numpy(np.asarray(fw.val, dtype=np.int64))
    rows = torch.cat([x for x, _ in full1])
    assert rows.shape[0] == 8  # all 8 windows used (partial last batch kept)
    for i in range(8):
        assert torch.equal(rows[i], val[i * WIN: i * WIN + 256])

    # the small eval set is the FIRST n_tokens of val.bin: a prefix of the full eval
    small = torch.cat([x for x, _ in lm_val_batches(fw, WIN * 3, batch_seqs=3)])
    assert small.shape[0] == 3
    assert torch.equal(small, rows[:3])

    with pytest.raises(ValueError, match="val"):
        next(lm_val_batches(fw, WIN * 8 + 1))


# ---------------------------------------------------------------- Cifar5m open

def test_cifar_counts_validated_on_open(tmp_path, cifar):
    assert cifar.x_train.shape == (N_TRAIN_IMG, 32, 32, 3)
    assert cifar.x_train.dtype == torch.uint8
    assert cifar.y_train.dtype == torch.int64
    assert cifar.x_val.shape[0] == N_EVAL_IMG and cifar.x_test.shape[0] == N_EVAL_IMG
    assert cifar.version() == "cifar-test-v1"

    root = write_cifar(tmp_path / "c2", n_train=N_TRAIN_IMG, n_eval=N_EVAL_IMG)
    with pytest.raises(ValueError, match="train"):
        Cifar5m(root, torch.device("cpu"), n_train=N_TRAIN_IMG + 1,
                n_val=N_EVAL_IMG, n_test=N_EVAL_IMG)
    with pytest.raises(ValueError, match="val"):
        Cifar5m(root, torch.device("cpu"), n_train=N_TRAIN_IMG,
                n_val=N_EVAL_IMG + 1, n_test=N_EVAL_IMG)


def test_cifar_splits_disjoint(cifar):
    def sigs(x):
        return {int(a) + 256 * int(b)
                for a, b in zip(x[:, 0, 0, 0], x[:, 0, 0, 1])}

    tr, va, te = sigs(cifar.x_train), sigs(cifar.x_val), sigs(cifar.x_test)
    assert len(tr) == N_TRAIN_IMG and len(va) == N_EVAL_IMG and len(te) == N_EVAL_IMG
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)


def test_prepare_cifar5m_slicing_disjoint(tmp_path):
    """The actual slice logic of scripts/prepare_cifar5m.py, on synthetic parts."""
    spec = importlib.util.spec_from_file_location(
        "prepare_cifar5m",
        Path(__file__).resolve().parents[1] / "scripts" / "prepare_cifar5m.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ids = np.arange(300)
    x = np.zeros((300, 32, 32, 3), dtype=np.uint8)
    x[:, 0, 0, 0] = ids % 256
    x[:, 0, 0, 1] = ids // 256
    y = (ids % 10).astype(np.int64)
    parts = [(x[:170], y[:170]), (x[170:], y[170:])]  # part0 ‖ part1

    out = mod.slice_parts(parts, n_train=240, n_val=20, n_test=20)
    # exact contiguous slices of the concatenation C = part0 ‖ part1
    assert np.array_equal(out["train"][0], x[:240])
    assert np.array_equal(out["val"][0], x[240:260])
    assert np.array_equal(out["test"][0], x[260:280])
    assert np.array_equal(out["train"][1], y[:240])
    assert out["train"][0].dtype == np.uint8 and out["train"][1].dtype == np.int64

    def sigs(a):
        return {int(p) + 256 * int(q) for p, q in zip(a[:, 0, 0, 0], a[:, 0, 0, 1])}
    tr, va, te = sigs(out["train"][0]), sigs(out["val"][0]), sigs(out["test"][0])
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)


# ---------------------------------------------------------------- cifar_batches

def test_no_image_repeats_single_pass(cifar):
    seen = []
    for x, y in cifar_batches(cifar, seed=1, steps=16, batch=32):  # all 512 images
        assert x.shape == (32, 3072) and x.dtype == torch.float32
        assert y.shape == (32,) and y.dtype == torch.int64
        ids = image_ids(x)
        assert all(int(l) == i % 10 for i, l in zip(ids, y))  # labels follow images
        seen += ids
    assert len(seen) == N_TRAIN_IMG and len(set(seen)) == N_TRAIN_IMG

    with pytest.raises(ValueError, match="image"):
        next(cifar_batches(cifar, seed=1, steps=17, batch=32))


def test_cifar_prefix_consistency(cifar):
    half = [image_ids(x) for x, _ in cifar_batches(cifar, seed=9, steps=4, batch=32)]
    full = [image_ids(x) for x, _ in cifar_batches(cifar, seed=9, steps=8, batch=32)]
    assert full[:4] == half


def test_seed_sets_order_and_prefix_consistency(cifar, fw):
    """Checklist name: same seed -> same order, different seed -> different order,
    and the 1x prefix property, for both datasets."""
    a = [image_ids(x) for x, _ in cifar_batches(cifar, seed=2, steps=4, batch=32)]
    b = [image_ids(x) for x, _ in cifar_batches(cifar, seed=2, steps=4, batch=32)]
    c = [image_ids(x) for x, _ in cifar_batches(cifar, seed=3, steps=4, batch=32)]
    assert a == b and a != c
    torch.manual_seed(0)
    np.random.seed(0)
    assert [image_ids(x) for x, _ in cifar_batches(cifar, seed=2, steps=4, batch=32)] == a

    full = [image_ids(x) for x, _ in cifar_batches(cifar, seed=2, steps=8, batch=32)]
    assert full[:4] == a
    lm_half = [window_ids(x) for x, _ in batches(fw, seed=2, steps=4)]
    lm_full = [window_ids(x) for x, _ in batches(fw, seed=2, steps=8)]
    assert lm_full[:4] == lm_half


# ---------------------------------------------------------------- normalize

def test_normalize_correctness():
    x = torch.zeros((2, 32, 32, 3), dtype=torch.uint8)
    x[1] = 255
    out = normalize(x)
    assert out.shape == (2, 3072)
    assert out.dtype == torch.float32
    mean = torch.tensor(CIFAR_MEAN)
    std = torch.tensor(CIFAR_STD)
    # channel-last flatten: flat index k corresponds to channel k % 3
    for c in range(3):
        zero_expect = (0.0 - mean[c]) / std[c]
        one_expect = (1.0 - mean[c]) / std[c]
        assert torch.allclose(out[0, c::3], zero_expect.expand(1024), atol=1e-6)
        assert torch.allclose(out[1, c::3], one_expect.expand(1024), atol=1e-6)

    # a known mid value: 128/255
    x2 = torch.full((1, 32, 32, 3), 128, dtype=torch.uint8)
    out2 = normalize(x2)
    expect = (128.0 / 255.0 - mean) / std
    assert torch.allclose(out2[0, :3], expect, atol=1e-6)
