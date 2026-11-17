#!/usr/bin/env python
"""Build the FineWeb-Edu token dataset for the LM studies.

Usage:
    uv run python scripts/prepare_fineweb.py \
        --out data/fineweb --train-tokens 45000000 --val-tokens 2000000

Per design.md §3.5 / §3.12 and Gate-0 answer 7:
  1. Stream HuggingFaceFW/fineweb-edu, config sample-10BT, and split DOCUMENTS
     into train/val first (every 20th document, by global stream index, goes
     to val; the rest to train). Collected text is cached locally as jsonl so
     later stages never re-stream.
  2. Train a byte-level BPE tokenizer, vocab 4096 INCLUDING the <|endoftext|>
     special token, on the train documents ONLY. -> tokenizer.json
  3. Encode each document followed by one <|endoftext|>; write >= train-tokens
     training tokens (stops at the document boundary that crosses the target)
     and exactly val-tokens validation tokens as little-endian uint16 memmaps.
     -> train.bin, val.bin
  4. Write meta.json (counts, vocab_size, tokenizer sha256, source, and a
     content-derived version string).

Resumable where cheap: each stage is skipped when its output already exists
(delete a stage's output to redo it); an interrupted collection resumes the
stream right after the last COMPLETE cached document — the resume position is
reconciled from the jsonl caches themselves (torn final lines truncated), not
from the checkpoint file, so no document is written twice. When only the
final outputs (train.bin, val.bin, meta.json) exist, the script does nothing.
If the collected text turns out to
encode to too few tokens, collection automatically continues where it left
off and encoding is retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

VOCAB_SIZE = 4096
EOT = "<|endoftext|>"
SOURCE = "fineweb-edu/sample-10BT"
VAL_EVERY = 20              # doc split rule: global doc index % 20 == 0 -> val
BYTES_PER_TOKEN_TARGET = 5  # byte-level BPE @4096 gives ~3-4 bytes/token; margin
ENCODE_CHUNK = 1024


def iter_texts(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


# ---------------------------------------------------------------- stage 1

def _truncate_torn_tail(path: Path) -> None:
    """Drop a torn final line (one not ending in a newline) left by a crash
    mid-write, so the jsonl holds only complete documents."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    size = path.stat().st_size
    with open(path, "rb+") as f:
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return
        pos = size
        block = 1 << 16
        while pos > 0:
            start = max(0, pos - block)
            f.seek(start)
            nl = f.read(pos - start).rfind(b"\n")
            if nl != -1:
                f.truncate(start + nl + 1)
                return
            pos = start
        f.truncate(0)


def _scan_jsonl(path: Path) -> tuple[int, int]:
    """Return (complete line count, total utf-8 text bytes) of a doc cache."""
    n = nbytes = 0
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                nbytes += len(json.loads(line)["text"].encode("utf-8"))
                n += 1
    return n, nbytes


def reconcile_collection(out: Path) -> dict:
    """Rebuild collection state from the jsonl caches themselves.

    collect_state.json is only a progress hint — it is checkpointed less often
    than the caches are appended to, so after a crash it can lag the files
    (trusting it would re-stream and double-write documents). Ground truth is
    the caches: truncate any torn final line, count complete lines, and derive
    docs_seen = train lines + val lines. The split rule (global doc index %
    VAL_EVERY == 0 -> val) fixes how many of the first N docs are val docs,
    which cross-checks that the two caches describe one consistent prefix.
    """
    for name in ("train_docs.jsonl", "val_docs.jsonl"):
        _truncate_torn_tail(out / name)
    n_train, train_bytes = _scan_jsonl(out / "train_docs.jsonl")
    n_val, val_bytes = _scan_jsonl(out / "val_docs.jsonl")
    docs_seen = n_train + n_val
    expect_val = (docs_seen + VAL_EVERY - 1) // VAL_EVERY
    if n_val != expect_val:
        raise RuntimeError(
            f"inconsistent doc caches: {n_train} train + {n_val} val lines, "
            f"but the first {docs_seen} docs should contain {expect_val} val "
            "docs. Delete train_docs.jsonl, val_docs.jsonl and "
            "collect_state.json and re-run.")
    return {"docs_seen": docs_seen, "train_bytes": train_bytes,
            "val_bytes": val_bytes}


def collect(out: Path, train_bytes_target: int, val_bytes_target: int) -> None:
    """Stream documents into train_docs.jsonl / val_docs.jsonl until both byte
    targets are met. Resume position is reconciled from the actual caches
    (never from the checkpoint file alone), so an interrupted run continues
    exactly after the last complete line with no duplicates."""
    state_path = out / "collect_state.json"
    state = reconcile_collection(out)
    if state_path.is_file():
        stale = json.loads(state_path.read_text())
        if stale != state:
            print(f"collect_state.json lagged the caches "
                  f"(said {stale.get('docs_seen')} docs, files hold "
                  f"{state['docs_seen']}); reconciled from the files")
    if state["train_bytes"] >= train_bytes_target and \
       state["val_bytes"] >= val_bytes_target:
        print(f"collection complete ({state['train_bytes'] / 1e6:.0f} MB train, "
              f"{state['val_bytes'] / 1e6:.0f} MB val); skipping")
        return

    from datasets import load_dataset  # deferred: needs network for streaming
    print(f"streaming {SOURCE} from document {state['docs_seen']} "
          f"(targets: {train_bytes_target / 1e6:.0f} MB train, "
          f"{val_bytes_target / 1e6:.0f} MB val)")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    if state["docs_seen"]:
        ds = ds.skip(state["docs_seen"])

    with open(out / "train_docs.jsonl", "a", encoding="utf-8") as f_train, \
         open(out / "val_docs.jsonl", "a", encoding="utf-8") as f_val:
        for doc in ds:
            i = state["docs_seen"]
            text = doc["text"]
            line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            nbytes = len(text.encode("utf-8"))
            if i % VAL_EVERY == 0:
                f_val.write(line)
                state["val_bytes"] += nbytes
            else:
                f_train.write(line)
                state["train_bytes"] += nbytes
            state["docs_seen"] = i + 1
            if state["docs_seen"] % 20_000 == 0:
                f_train.flush()
                f_val.flush()
                state_path.write_text(json.dumps(state))
                print(f"  {state['docs_seen']:,} docs: "
                      f"{state['train_bytes'] / 1e6:.0f} MB train, "
                      f"{state['val_bytes'] / 1e6:.0f} MB val", flush=True)
            if state["train_bytes"] >= train_bytes_target and \
               state["val_bytes"] >= val_bytes_target:
                break
    state_path.write_text(json.dumps(state))
    print(f"collected {state['docs_seen']:,} docs "
          f"({state['train_bytes'] / 1e6:.0f} MB train, "
          f"{state['val_bytes'] / 1e6:.0f} MB val)")


# ---------------------------------------------------------------- stage 2

def train_tokenizer(out: Path):
    from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
    from tokenizers.models import BPE

    tok_path = out / "tokenizer.json"
    if tok_path.is_file():
        print("tokenizer.json exists; skipping tokenizer training")
        return Tokenizer.from_file(str(tok_path))

    print(f"training byte-level BPE, vocab {VOCAB_SIZE} incl. {EOT}, "
          "on train documents only ...")
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(iter_texts(out / "train_docs.jsonl"), trainer=trainer)
    if tok.get_vocab_size() != VOCAB_SIZE:
        raise RuntimeError(f"tokenizer vocab is {tok.get_vocab_size()}, "
                           f"expected {VOCAB_SIZE}")
    tok.save(str(tok_path))
    print("wrote tokenizer.json")
    return tok


# ---------------------------------------------------------------- stage 3

class InsufficientTokens(Exception):
    pass


def encode_split(tok, jsonl: Path, bin_path: Path, target: int, exact: bool) -> int:
    """Encode docs (each + one EOT) to a little-endian uint16 bin.

    exact=False (train): stop at the first document boundary at/after target
    (so count >= target). exact=True (val): truncate to exactly target tokens.
    Raises InsufficientTokens when the jsonl runs out first.
    """
    eot_id = tok.token_to_id(EOT)
    assert eot_id is not None and eot_id < VOCAB_SIZE
    total = 0
    with open(bin_path, "wb") as f:
        chunk: list[str] = []

        def flush(chunk):
            nonlocal total
            ids: list[int] = []
            for enc in tok.encode_batch(chunk):
                ids.extend(enc.ids)
                ids.append(eot_id)
            if exact and total + len(ids) > target:
                ids = ids[: target - total]
            f.write(np.asarray(ids, dtype="<u2").tobytes())
            total += len(ids)

        for text in iter_texts(jsonl):
            chunk.append(text)
            if len(chunk) >= ENCODE_CHUNK:
                flush(chunk)
                chunk = []
                if total >= target:
                    break
        if chunk and total < target:
            flush(chunk)
    if total < target:
        raise InsufficientTokens(
            f"{jsonl.name} encoded to {total:,} tokens < target {target:,}")
    print(f"wrote {bin_path.name}: {total:,} tokens")
    return total


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/fineweb"))
    ap.add_argument("--train-tokens", type=int, default=45_000_000)
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    args = ap.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    if (out / "meta.json").is_file() and (out / "train.bin").is_file() \
            and (out / "val.bin").is_file():
        print(f"{out} already complete; nothing to do")
        return

    train_target_bytes = args.train_tokens * BYTES_PER_TOKEN_TARGET
    val_target_bytes = args.val_tokens * BYTES_PER_TOKEN_TARGET

    for attempt in range(4):
        collect(out, train_target_bytes, val_target_bytes)
        tok = train_tokenizer(out)  # train docs ONLY; val docs held out above
        try:
            n_train = encode_split(tok, out / "train_docs.jsonl",
                                   out / "train.bin", args.train_tokens,
                                   exact=False)
            n_val = encode_split(tok, out / "val_docs.jsonl",
                                 out / "val.bin", args.val_tokens, exact=True)
            break
        except InsufficientTokens as e:
            print(f"{e}; collecting 30% more text and re-encoding")
            train_target_bytes = int(train_target_bytes * 1.3)
            val_target_bytes = int(val_target_bytes * 1.3)
    else:
        raise RuntimeError("could not reach the token targets in 4 attempts")

    tok_sha = hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest()
    content = hashlib.sha256()
    for name in ("train.bin", "val.bin"):
        with open(out / name, "rb") as f:
            while block := f.read(1 << 24):
                content.update(block)
    content.update(tok_sha.encode())

    meta = {
        "version": f"fineweb-edu-10bt-v1-{content.hexdigest()[:12]}",
        "n_train": n_train,
        "n_val": n_val,
        "vocab_size": VOCAB_SIZE,
        "tokenizer_sha256": tok_sha,
        "source": SOURCE,
        "tokenizer_trained_on": "train_docs_only",
        "doc_split_rule": f"global doc index %% {VAL_EVERY} == 0 -> val",
        "content_sha256": content.hexdigest(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote meta.json, version {meta['version']}")
    print("The cached *_docs.jsonl and collect_state.json may be deleted "
          "once you are happy with the build.")


if __name__ == "__main__":
    sys.exit(main())
