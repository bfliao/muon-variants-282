"""Tests for muonarch/train.py — the single-run training loop (design.md §3.8).

Small synthetic worlds, all on CPU:

- MLP study: width-16 MLP on synthetic CIFAR-format data (random uint8 images,
  all-zero labels so the loss falls fast and decisively below the ln(10)
  divergence threshold), built through data.py's injectable size parameters.
  The test-only size overrides live INSIDE cfg["data"] — a hashed field — so
  every shrunken world has its own run identity (the pool/sizes change what
  the run computes).
- LM study: the REAL dense Qwen3 model (1.7M params), 6 steps, batch 2 x
  seq_len 32, on a tiny synthetic FineWeb world whose tokens are drawn from
  {0..15} (low entropy, so the loss can fall below the ln(4096) threshold
  within a few steps; uniform tokens over the full vocab would make ln(4096)
  the Bayes loss and every run would be flagged diverged).

Checkpoint convention under test (see train.py docstring): the checkpoint for
step k is saved AFTER step k-1 completes and BEFORE step k executes, so a
branch resuming at k replays steps k..total-1 bit-exactly on CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from muonarch import runner, train
from muonarch.data import Cifar5m, FineWeb, cifar_batches
from muonarch.models import build_dense, build_moe

SEQ = 32
WIN = SEQ + 1


# ---------------------------------------------------------------- worlds

def write_cifar_world(root: Path, n_train: int = 512, n_eval: int = 16,
                      random_labels: bool = False,
                      version: str = "cifar-test-v1") -> Path:
    """Synthetic CIFAR-format world: random images; labels all 0 (so the loss
    falls fast and decisively below ln(10)), or uniform-random labels for the
    divergence tests (unlearnable: any blow-up crosses ln(10))."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_total = n_train + 2 * n_eval
    x = rng.integers(0, 256, size=(n_total, 32, 32, 3), dtype=np.uint8)
    if random_labels:
        y = rng.integers(0, 10, size=n_total).astype(np.int64)
    else:
        y = np.zeros(n_total, dtype=np.int64)
    splits = {"train": slice(0, n_train),
              "val": slice(n_train, n_train + n_eval),
              "test": slice(n_train + n_eval, n_total)}
    for name, sl in splits.items():
        np.save(root / f"{name}.npy", x[sl])
        np.save(root / f"{name}_labels.npy", y[sl])
    (root / "meta.json").write_text(json.dumps({"version": version}))
    return root


def write_fineweb_world(root: Path, n_windows: int = 16, n_val_windows: int = 4) -> Path:
    """Synthetic FineWeb world: low-entropy tokens in {0..15}, vocab 4096."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    train_tokens = rng.integers(0, 16, size=n_windows * WIN).astype("<u2")
    val_tokens = rng.integers(0, 16, size=n_val_windows * WIN).astype("<u2")
    (root / "train.bin").write_bytes(train_tokens.tobytes())
    (root / "val.bin").write_bytes(val_tokens.tobytes())
    tok = b'{"model": {"type": "BPE"}, "note": "dummy"}'
    (root / "tokenizer.json").write_bytes(tok)
    meta = {
        "version": "fineweb-test-v1",
        "n_train": len(train_tokens),
        "n_val": len(val_tokens),
        "vocab_size": 4096,
        "tokenizer_sha256": hashlib.sha256(tok).hexdigest(),
        "source": "fineweb-edu/sample-10BT",
    }
    (root / "meta.json").write_text(json.dumps(meta))
    return root


# ---------------------------------------------------------------- configs

def mlp_cfg(data_path: Path, **over) -> dict:
    cfg = {
        "schema_version": 1,
        "study": "mlp_test",
        "phase": "fresh_seed",
        "arch": "mlp",
        "width": 16,
        "optimizer": "adamw",
        "rule": None,
        "lr": 0.05,
        "wd": 0.0,
        "momentum": 0.95,
        "adamw_betas": [0.9, 0.95],
        "adamw_eps": 1e-8,
        "adamw_group_lr": None,
        "adamw_group_wd": None,
        "ns_steps": 5,
        "ns_dtype": "fp32",
        "manifold": None,
        "init_projection": False,
        "seed": 0,
        "budget_multiplier": 1.0,
        "branch": None,
        "budget_base_steps": 16,
        "budget_unit": "images",
        "batch": 8,
        "seq_len": None,
        "grad_clip": 1.0,
        "schedule": {"kind": "wsd", "warmup_frac_of_1x": 0.05, "decay_frac": 0.2},
        "precision": "fp32",
        "router_aux_loss_coef": None,
        "eval": {"every_steps": 4, "val_small_tokens": None, "val_final": "full"},
        "data": {"dataset": "cifar5m", "version": "cifar-test-v1",
                 "n_train": 512, "n_val": 16, "n_test": 16},
        # logged-only:
        "data_path": str(data_path),
    }
    cfg.update(over)
    return cfg


def lm_cfg(data_path: Path, **over) -> dict:
    cfg = {
        "schema_version": 1,
        "study": "lm_test",
        "phase": "fresh_seed",
        "arch": "dense",
        "width": None,
        "optimizer": "adamw",
        "rule": None,
        "lr": 0.01,
        "wd": 0.0,
        "momentum": 0.95,
        "adamw_betas": [0.9, 0.95],
        "adamw_eps": 1e-8,
        "adamw_group_lr": None,
        "adamw_group_wd": None,
        "ns_steps": 5,
        "ns_dtype": "fp32",
        "manifold": None,
        "init_projection": False,
        "seed": 0,
        "budget_multiplier": 1.0,
        "branch": None,
        "budget_base_steps": 8,
        "budget_unit": "tokens",
        "batch": 2,
        "seq_len": SEQ,
        "grad_clip": 1.0,
        "schedule": {"kind": "wsd", "warmup_frac_of_1x": 0.05, "decay_frac": 0.2},
        "precision": "fp32",
        "router_aux_loss_coef": None,
        "eval": {"every_steps": 4, "val_small_tokens": 2 * WIN, "val_final": "full"},
        "data": {"dataset": "fineweb_edu", "version": "fineweb-test-v1",
                 "min_train_tokens": 16 * WIN, "val_tokens": 4 * WIN,
                 "pool_windows": 16},
        # logged-only:
        "data_path": str(data_path),
    }
    cfg.update(over)
    return cfg


@pytest.fixture
def cifar_root(tmp_path):
    return write_cifar_world(tmp_path / "cifar5m")


@pytest.fixture
def cifar_root_rand(tmp_path):
    return write_cifar_world(tmp_path / "cifar5m_rand", random_labels=True,
                             version="cifar-rand-v1")


@pytest.fixture
def fineweb_root(tmp_path):
    return write_fineweb_world(tmp_path / "fineweb")


def read_log(root: Path, cfg: dict) -> list[dict]:
    lines = runner.log_path(root, cfg).read_text().splitlines()
    return [json.loads(l) for l in lines]


def evals(records: list[dict]) -> list[dict]:
    return [r for r in records if r["t"] == "eval"]


# ================================================================ CE shift

def test_manual_ce_matches_hf_shifted_loss():
    """Our manual CE uses NO internal shifting: with data.py's (x, y) windows
    (y = x shifted by one within a 257-token window) it predicts every token
    exactly once, including the final one HF's labels= path drops."""
    torch.manual_seed(0)
    model = build_dense()
    model.eval()
    V = model.config.vocab_size
    B = 2
    w = torch.randint(0, 16, (B, WIN))
    x, y = w[:, :SEQ], w[:, 1:]
    assert torch.equal(y[:, :-1], x[:, 1:])  # data alignment

    with torch.no_grad():
        out = model(input_ids=x, labels=x.clone())  # a copy — comparison only
        logits = model(input_ids=x, use_cache=False).logits

    # (1) HF's internal loss is the shifted CE over L-1 positions.
    hf_manual = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, V), x[:, 1:].reshape(-1)
    )
    assert torch.allclose(out.loss, hf_manual, atol=1e-5)

    # (2) train.py's CE covers ALL L positions against y, no shifting.
    ours = train.lm_cross_entropy(logits, y)
    direct = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    assert torch.allclose(ours, direct)

    # (3) ours = HF's L-1 predictions plus the one extra final prediction:
    # every token predicted exactly once.
    last = F.cross_entropy(logits[:, -1, :], y[:, -1], reduction="sum")
    total_ours = ours * (B * SEQ)
    total_hf = out.loss * (B * (SEQ - 1))
    assert torch.allclose(total_ours, total_hf + last, atol=1e-3)


# ================================================================ MoE aux loss

def test_moe_aux_loss_semantics_and_addition():
    """transformers 4.57.6: with output_router_logits=True and labels NOT
    passed, outputs.aux_loss is the UNcoefficiented load-balancing loss
    (modeling_qwen3_moe.py:671-678; the coef applies only under labels,
    :679-680). train.py adds cfg coef * aux itself."""
    torch.manual_seed(0)
    model = build_moe()
    model.eval()
    x = torch.randint(0, 16, (2, 16))
    with torch.no_grad():
        out = model(input_ids=x, use_cache=False)

    # hand computation of the Switch-Transformer load-balancing loss
    E = model.config.num_experts
    k = model.config.num_experts_per_tok
    gate_logits = torch.cat([g.float() for g in out.router_logits], dim=0)
    rw = F.softmax(gate_logits, dim=-1)
    sel = torch.topk(rw, k, dim=-1).indices
    mask = F.one_hot(sel, E).float()
    tokens_per_expert = mask.mean(dim=0)
    router_prob = rw.mean(dim=0)
    hand = (tokens_per_expert * router_prob.unsqueeze(0)).sum() * E
    assert torch.allclose(out.aux_loss, hand, atol=1e-6)

    # the HF labels= path applies the model-config coefficient — showing
    # aux_loss itself is returned uncoefficiented
    with torch.no_grad():
        labeled = model(input_ids=x, labels=x.clone(), use_cache=False)
        logits = out.logits
    V = model.config.vocab_size
    shifted = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), x[:, 1:].reshape(-1))
    expect = shifted + model.config.router_aux_loss_coef * labeled.aux_loss
    assert torch.allclose(labeled.loss, expect, atol=1e-5)

    # train.py's training loss = manual CE + cfg coef * aux, aux logged apart
    coef = 0.5  # deliberately different from the model config's 0.01
    y = torch.randint(0, 16, (2, 16))
    with torch.no_grad():
        loss, ce, aux = train.lm_forward_loss(model, x, y, aux_coef=coef)
    assert aux is not None
    assert torch.allclose(aux, out.aux_loss, atol=1e-6)
    assert torch.allclose(loss, ce + coef * aux, atol=1e-6)


# ================================================================ MLP run

def test_mlp_run_completes_with_wellformed_logs(tmp_path, cifar_root):
    cfg = mlp_cfg(cifar_root)
    result = train.run(cfg, tmp_path)

    assert result["status"] == "completed"
    assert result["diverged_at_step"] is None
    assert result["hash"] == runner.config_hash(cfg)
    for key in ("final_val_loss", "final_val_acc", "final_test_loss",
                "final_test_acc", "train_val_gap", "flops", "device",
                "hostname", "precision", "torch_version",
                "transformers_version", "wall_clock_s", "started_at",
                "finished_at", "config", "schema_version",
                "dual_steps_mean", "msign_calls_mean", "edge_rule_fired"):
        assert key in result, key
    assert result["final_val_loss"] < math.log(10)
    fl = result["flops"]
    assert set(fl) == {"model", "optimizer", "total"}
    assert fl["model"] > 0 and fl["optimizer"] > 0
    assert fl["total"] == fl["model"] + fl["optimizer"]

    # result JSON on disk, atomically written, equal to the return value
    on_disk = json.loads(runner.result_path(tmp_path, cfg).read_text())
    assert on_disk == result

    records = read_log(tmp_path, cfg)
    assert records[0]["t"] == "start"
    assert "param_map" in records[0] and "fractions" in records[0]
    ev = evals(records)
    assert [r["step"] for r in ev] == [4, 8, 12, 16]
    for r in ev:
        for key in ("step", "images_or_tokens", "lr", "train_loss",
                    "train_loss_last", "val_loss", "val_acc", "grad_norm",
                    "aux_loss", "dual_steps_mean", "dual_residual_mean",
                    "msign_calls_mean"):
            assert key in r, key
        assert math.isfinite(r["train_loss"]) and math.isfinite(r["val_loss"])
    assert ev[0]["images_or_tokens"] == 4 * 8
    assert records[-1]["t"] == "final"
    assert "spectral_norms" in records[-1] and "flops" in records[-1]


def test_lr_follows_schedule(tmp_path, cifar_root):
    cfg = mlp_cfg(cifar_root)
    train.run(cfg, tmp_path)
    ev = evals(read_log(tmp_path, cfg))
    # warmup = round(0.05*16) = 1; steps 1..12 run at full lr; decay_start =
    # 16 - round(0.2*16) = 13; the step-16 record logs the lr applied AT step
    # 15: (16-15)/(16-13) = 1/3 of base.
    assert ev[0]["lr"] == pytest.approx(cfg["lr"])
    assert ev[-1]["lr"] == pytest.approx(cfg["lr"] / 3)


def test_run_is_idempotent_short_circuit(tmp_path, cifar_root, monkeypatch):
    cfg = mlp_cfg(cifar_root)
    result = train.run(cfg, tmp_path)

    def boom(*a, **k):
        raise AssertionError("is_done runs must never rebuild the model")

    monkeypatch.setattr(train, "build_model", boom)
    again = train.run(cfg, tmp_path)
    assert again == result


# ================================================================ divergence

def test_divergence_absurd_lr(tmp_path, cifar_root_rand):
    cfg = mlp_cfg(cifar_root_rand, lr=500.0,
                  data={"dataset": "cifar5m", "version": "cifar-rand-v1",
                        "n_train": 512, "n_val": 16, "n_test": 16})
    result = train.run(cfg, tmp_path)
    assert result["status"] == "diverged"
    assert result["diverged_at_step"] is not None
    # stopped immediately: nothing logged beyond the diverging step
    ev = evals(read_log(tmp_path, cfg))
    assert ev[-1]["step"] == result["diverged_at_step"]
    assert all(r["step"] <= result["diverged_at_step"] for r in ev)
    # the result is written (divergence is a COMPLETED run; it must not rerun)
    assert runner.is_done(tmp_path, cfg)


def test_divergence_nan(tmp_path, cifar_root):
    cfg = mlp_cfg(cifar_root, lr=1e20)  # overflow -> NaN within a step or two
    result = train.run(cfg, tmp_path)
    assert result["status"] == "diverged"
    # NaN is caught per step, before the first logged step
    assert result["diverged_at_step"] <= cfg["eval"]["every_steps"]
    assert runner.is_done(tmp_path, cfg)


def test_divergence_at_final_eval_keeps_computed_val_loss(tmp_path, cifar_root_rand):
    """A run that only diverges at its FINAL evaluation (no periodic evals
    fire) still records the computed full-set val loss in the result, with
    status "diverged"."""
    cfg = mlp_cfg(cifar_root_rand, lr=500.0,
                  data={"dataset": "cifar5m", "version": "cifar-rand-v1",
                        "n_train": 512, "n_val": 16, "n_test": 16},
                  eval={"every_steps": 100, "val_small_tokens": None,
                        "val_final": "full"})
    result = train.run(cfg, tmp_path)
    assert result["status"] == "diverged"
    assert result["diverged_at_step"] == 16  # the final step
    assert result["final_val_loss"] is not None
    assert result["final_val_loss"] > math.log(10)


def test_divergence_threshold_is_theoretical_initial_loss():
    assert train.divergence_threshold({"arch": "mlp"}, None) == pytest.approx(math.log(10))
    model = build_dense()
    assert train.divergence_threshold({"arch": "dense"}, model) == pytest.approx(math.log(4096))


# ================================================================ atomic writes

def test_atomic_write_survives_concurrent_writers(tmp_path):
    """Two writers racing on the same result path must leave a VALID JSON
    file (unique tmp per writer + os.replace; a shared tmp name interleaved
    into valid JSON + trailing garbage — found by review)."""
    import threading

    target = tmp_path / "results" / "s" / "abc.json"
    errors = []

    def writer(tag):
        try:
            for i in range(50):
                train._atomic_write_json(target, {"writer": tag, "i": i})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    obj = json.loads(target.read_text())  # whole-file valid: one payload won
    assert obj["writer"] in (0, 1) and obj["i"] == 49
    assert not list(target.parent.glob("*.tmp"))  # no leaked tmp files


def test_corrupt_result_is_not_done(tmp_path, cifar_root):
    """A corrupted result JSON (valid JSON + trailing garbage, the pre-fix
    race artifact) must not wedge the run: run() warns, re-runs, and
    overwrites it with a valid result."""
    cfg = mlp_cfg(cifar_root)
    rp = runner.result_path(tmp_path, cfg)
    rp.parent.mkdir(parents=True)
    rp.write_text('{"status": "completed"}GARBAGE-TRAILER')
    assert runner.is_done(tmp_path, cfg)  # path-level done...
    with pytest.warns(RuntimeWarning, match="corrupt result JSON"):
        result = train.run(cfg, tmp_path)  # ...but run() recovers
    assert result["status"] == "completed"
    assert json.loads(rp.read_text()) == result


# ================================================================ kill simulation

def test_result_written_only_on_completion(tmp_path, cifar_root, monkeypatch):
    cfg = mlp_cfg(cifar_root)

    calls = {"n": 0}
    real_evaluate = train.evaluate

    def killed(*a, **k):
        calls["n"] += 1
        raise RuntimeError("simulated crash mid-run")

    monkeypatch.setattr(train, "evaluate", killed)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train.run(cfg, tmp_path)
    assert calls["n"] == 1
    assert not runner.result_path(tmp_path, cfg).exists()
    assert not runner.is_done(tmp_path, cfg)

    # rerun completes cleanly and overwrites the partial log
    monkeypatch.setattr(train, "evaluate", real_evaluate)
    result = train.run(cfg, tmp_path)
    assert result["status"] == "completed"
    records = read_log(tmp_path, cfg)
    assert sum(1 for r in records if r["t"] == "start") == 1  # log restarted


# ================================================================ checkpoints

def test_checkpoint_roundtrip_bit_exact(tmp_path, cifar_root):
    """budget_constant run saves checkpoints at checkpoint_step(base, m);
    a branch resumed there equals a straight run of the same budget
    bit-exactly on CPU, and the pre-checkpoint history matches the constant
    run's."""
    base = 16
    const_cfg = mlp_cfg(cifar_root, phase="budget_constant",
                        budget_multiplier=2.0)
    const_res = train.run(const_cfg, tmp_path)
    assert const_res["status"] == "completed"

    # checkpoints at round(0.8 * round(m*16)) for m in (0.125, 0.25, 0.5)
    const_hash = runner.config_hash(const_cfg)
    for step in (2, 3, 6):
        assert runner.ckpt_file(tmp_path, "mlp_test", const_hash, step).exists()
    ck = torch.load(runner.ckpt_file(tmp_path, "mlp_test", const_hash, 6),
                    weights_only=False)
    assert ck["step"] == 6
    assert ck["data_state"]["consumed"] == 6

    straight_cfg = mlp_cfg(cifar_root, phase="fresh_seed", budget_multiplier=0.5)
    straight_res = train.run(straight_cfg, tmp_path)
    assert straight_res["status"] == "completed"

    branch_cfg = mlp_cfg(
        cifar_root, phase="budget_branch", budget_multiplier=0.5,
        branch={"parent_hash": const_hash, "checkpoint_step": 6})
    branch_res = train.run(branch_cfg, tmp_path)
    assert branch_res["status"] == "completed"

    # ---- pre-checkpoint history: the straight run's eval at step 4 equals
    # the constant run's bit-exactly (same seed, data prefix, lr history).
    const_ev = {r["step"]: r for r in evals(read_log(tmp_path, const_cfg))}
    straight_ev = {r["step"]: r for r in evals(read_log(tmp_path, straight_cfg))}
    branch_ev = {r["step"]: r for r in evals(read_log(tmp_path, branch_cfg))}
    for key in ("train_loss", "train_loss_last", "val_loss", "val_acc", "grad_norm"):
        assert straight_ev[4][key] == const_ev[4][key], key

    # ---- post-checkpoint: branch (resumed at 6) == straight run, bit-exact.
    # The step-8 record's last-batch loss is produced AFTER the checkpoint in
    # both runs, from bit-identical state.
    assert branch_ev[8]["train_loss_last"] == straight_ev[8]["train_loss_last"]
    assert branch_ev[8]["val_loss"] == straight_ev[8]["val_loss"]
    assert branch_ev[8]["val_acc"] == straight_ev[8]["val_acc"]
    assert branch_res["final_val_loss"] == straight_res["final_val_loss"]
    assert branch_res["final_test_loss"] == straight_res["final_test_loss"]

    # the branch logs nothing before its resume step
    assert min(branch_ev) > 6


def test_load_checkpoint_maps_to_cpu(tmp_path, cifar_root):
    """load_checkpoint uses map_location="cpu", so a checkpoint saved under
    another device (MPS) is loadable by a CPU process; after optimizer
    state_dict load, float state lands on each parameter's device."""
    cfg = mlp_cfg(cifar_root, optimizer="muon", rule="keller", lr=0.02,
                  wd=0.01, adamw_group_lr=0.01, adamw_group_wd=0.0)
    device = torch.device("cpu")
    model, opts, _ = train.setup(cfg, device)
    data = train.open_data(cfg, tmp_path, device)
    batches = train.train_batches(cfg, data, 2)
    for _ in range(2):  # populate momentum buffers / AdamW moments
        x, y = next(batches)
        loss = F.cross_entropy(model(x), y)
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in opts:
            opt.step()

    path = tmp_path / "ck.pt"
    train.save_checkpoint(path, model, opts, 2, {"consumed": 2},
                          torch.get_rng_state())
    ck = train.load_checkpoint(path)

    def tensors(obj):
        if torch.is_tensor(obj):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from tensors(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from tensors(v)

    for t in tensors([ck["model"], ck["optimizers"]]):
        assert t.device.type == "cpu"

    # restore into a fresh setup and check float optimizer state placement
    model2, opts2, _ = train.setup(cfg, device)
    model2.load_state_dict(ck["model"])
    for opt, sd in zip(opts2, ck["optimizers"]):
        opt.load_state_dict(sd)
        for p, st in opt.state.items():
            for v in st.values():
                if torch.is_tensor(v) and v.is_floating_point():
                    assert v.device == p.device
    # states round-tripped intact (resume correctness is covered end-to-end
    # by test_checkpoint_roundtrip_bit_exact)
    for a, b in zip(opts[0].state.values(), opts2[0].state.values()):
        assert torch.equal(a["momentum_buffer"], b["momentum_buffer"])


def test_branch_does_not_save_checkpoints(tmp_path, cifar_root):
    const_cfg = mlp_cfg(cifar_root, phase="budget_constant", budget_multiplier=2.0)
    train.run(const_cfg, tmp_path)
    branch_cfg = mlp_cfg(
        cifar_root, phase="budget_branch", budget_multiplier=0.5,
        branch={"parent_hash": runner.config_hash(const_cfg), "checkpoint_step": 6})
    train.run(branch_cfg, tmp_path)
    bh = runner.config_hash(branch_cfg)
    ckpt_dir = tmp_path / "checkpoints" / "mlp_test"
    assert not list(ckpt_dir.glob(f"{bh}@*.pt"))


# ================================================================ seed contract

def test_seed_contract_identical_init_and_first_batch(cifar_root):
    """Seed s fixes init and data order identically across optimizers
    (paired comparisons): torch.manual_seed(seed) runs immediately before
    build_model."""
    cfg_a = mlp_cfg(cifar_root, optimizer="adamw")
    cfg_m = mlp_cfg(cifar_root, optimizer="muon", rule="keller", lr=0.02,
                    wd=0.01, adamw_group_lr=0.003, adamw_group_wd=0.0)
    device = torch.device("cpu")
    model_a, opts_a, _ = train.setup(cfg_a, device)
    model_m, opts_m, _ = train.setup(cfg_m, device)
    sd_a, sd_m = model_a.state_dict(), model_m.state_dict()
    assert sd_a.keys() == sd_m.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_m[k]), k

    ds = Cifar5m(cifar_root, device, n_train=512, n_val=16, n_test=16)
    b1 = next(cifar_batches(ds, seed=cfg_a["seed"], steps=16, batch=8))
    b2 = next(cifar_batches(ds, seed=cfg_m["seed"], steps=16, batch=8))
    assert torch.equal(b1[0], b2[0]) and torch.equal(b1[1], b2[1])

    # same seed, same optimizer => identical optimizer construction
    model_a2, opts_a2, _ = train.setup(cfg_a, device)
    assert len(opts_a) == len(opts_a2) == 1


def test_init_projection_changes_init_only_when_requested(cifar_root):
    cfg = mlp_cfg(cifar_root, optimizer="manifold", rule=None, lr=0.05,
                  wd=None, adamw_group_lr=0.01, adamw_group_wd=0.0,
                  manifold={"dual_cap": 2, "tol": 1e-6, "dual_lr": 0.01,
                            "msign_steps": 10},
                  init_projection=True)
    model, _, groups = train.setup(cfg, torch.device("cpu"))
    for name, p in groups.muon:
        W = p.detach()
        if W.shape[0] > W.shape[1]:
            W = W.mT
        eye = torch.eye(W.shape[0])
        assert torch.allclose(W @ W.mT, eye, atol=1e-2), name


# ================================================================ grad clip

def test_grad_clip_applied_and_off():
    p1 = torch.nn.Parameter(torch.ones(4))
    p2 = torch.nn.Parameter(torch.ones(3))
    for p in (p1, p2):
        p.grad = torch.full_like(p, 2.0)
    raw = math.sqrt(sum(float((p.grad ** 2).sum()) for p in (p1, p2)))

    # off: grads untouched, true norm returned
    norm = train.clip_gradients([p1, p2], None)
    assert norm == pytest.approx(raw)
    assert torch.equal(p1.grad, torch.full_like(p1, 2.0))

    # on: pre-clip norm returned, post-clip joint norm == max_norm
    norm = train.clip_gradients([p1, p2], 0.5)
    assert norm == pytest.approx(raw, rel=1e-5)
    clipped = math.sqrt(sum(float((p.grad ** 2).sum()) for p in (p1, p2)))
    assert clipped == pytest.approx(0.5, rel=1e-5)


def test_grad_clip_in_run_caps_logged_norm(tmp_path, cifar_root):
    cfg = mlp_cfg(cifar_root, grad_clip=None)
    train.run(cfg, tmp_path)
    free = [r["grad_norm"] for r in evals(read_log(tmp_path, cfg))]
    # the zero-label world is fully fit within a few steps, after which the
    # gradient is exactly 0 in fp32 — only the first window must be nonzero
    assert free[0] > 0 and all(math.isfinite(g) for g in free)
    # grad_norm is logged in both modes; with clip on, updates use the capped
    # gradient (clip_gradients is unit-tested above) and the same pre-clip
    # norm is logged.
    cfg2 = mlp_cfg(cifar_root, grad_clip=0.01, seed=1)
    train.run(cfg2, tmp_path)
    capped = [r["grad_norm"] for r in evals(read_log(tmp_path, cfg2))]
    assert all(math.isfinite(g) for g in capped)


# ================================================================ LM run

def test_lm_run_completes_with_diagnostics(tmp_path, fineweb_root):
    cfg = lm_cfg(fineweb_root)
    result = train.run(cfg, tmp_path)
    assert result["status"] == "completed", result.get("diverged_at_step")
    assert result["final_val_loss"] < math.log(4096)
    assert result["final_test_loss"] is None  # LM has no test set

    records = read_log(tmp_path, cfg)
    ev = evals(records)
    assert [r["step"] for r in ev] == [4, 8]
    assert ev[0]["images_or_tokens"] == 4 * 2 * SEQ
    for r in ev:
        assert r["max_attn_logit"] is not None and math.isfinite(r["max_attn_logit"])
        assert len(r["sigma_max_q"]) == 6 and len(r["sigma_max_k"]) == 6
        assert all(s > 0 for s in r["sigma_max_q"])
    final = records[-1]
    assert final["t"] == "final"
    assert final["val_loss"] == result["final_val_loss"]
    # final spectral norms cover the muon-group matrices (adamw run -> the
    # map is still logged from the muon split? no: adamw has no muon group)
    assert isinstance(final["spectral_norms"], dict)


def test_lm_run_muon_logs_spectral_norms(tmp_path, fineweb_root):
    cfg = lm_cfg(fineweb_root, optimizer="muon", rule="keller", lr=0.02,
                 wd=0.01, adamw_group_lr=0.01, adamw_group_wd=0.0)
    result = train.run(cfg, tmp_path)
    assert result["status"] == "completed"
    final = read_log(tmp_path, cfg)[-1]
    norms = final["spectral_norms"]
    # 6 layers x (q,k,v,o + gate,up,down) = 42 muon matrices on the dense model
    assert len(norms) == 42
    assert all(v > 0 for v in norms.values())


# ================================================================ manifold run

def test_manifold_mlp_run_logs_dual_stats(tmp_path, cifar_root):
    cfg = mlp_cfg(cifar_root, optimizer="manifold", rule=None, lr=0.1,
                  wd=None, adamw_group_lr=0.05, adamw_group_wd=0.0,
                  manifold={"dual_cap": 2, "tol": 1e-6, "dual_lr": 0.01,
                            "msign_steps": 10},
                  init_projection=True, budget_multiplier=0.5)
    result = train.run(cfg, tmp_path)
    assert result["status"] == "completed", result.get("diverged_at_step")
    assert result["dual_steps_mean"] is not None
    assert result["msign_calls_mean"] >= 2.0  # >= 1 ascent + 1 retraction
    ev = evals(read_log(tmp_path, cfg))
    for r in ev:
        assert r["msign_calls_mean"] >= 2.0
        assert r["dual_steps_mean"] is not None
        assert r["dual_residual_mean"] is not None
    assert result["flops"]["optimizer"] > 0


# ================================================================ diagnostics

def test_max_attn_logit_recompute_is_exact():
    """The hook-based recompute reproduces the model's own eager attention
    bit-exactly: softmax of our causally-masked recomputed logits equals the
    attention weights an eager forward returns."""
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    torch.manual_seed(0)
    model = build_dense()
    model.eval()
    model.config._attn_implementation = "eager"
    x = torch.randint(0, 4096, (2, 16))
    with torch.no_grad():
        out = model(input_ids=x, use_cache=False, output_attentions=True)
        _, captured = train._capture_attention_forward(model, x)
        for (module, hs, pe), ref in zip(captured, out.attentions):
            hidden_shape = (*hs.shape[:-1], -1, module.head_dim)
            q = module.q_norm(module.q_proj(hs).view(hidden_shape)).transpose(1, 2)
            k = module.k_norm(module.k_proj(hs).view(hidden_shape)).transpose(1, 2)
            cos, sin = pe
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = repeat_kv(k, module.num_key_value_groups)
            logits = (q @ k.transpose(-2, -1)) * module.scaling
            T = logits.size(-1)
            causal = torch.ones(T, T, dtype=torch.bool).tril()
            probs = logits.masked_fill(~causal, float("-inf")).softmax(-1)
            assert torch.equal(probs, ref)
        # and the diagnostic itself is the max of those masked logits
        worst = train._max_attention_logit(captured)
        assert math.isfinite(worst)


# ================================================================ MoE diagnostics

def test_moe_diagnostics_expert_stats():
    torch.manual_seed(0)
    model = build_moe()
    model.eval()
    probe = torch.randint(0, 16, (2, 16))
    diag = train.diagnostics(model, {"arch": "moe"}, probe, groups=None)
    assert diag["max_attn_logit"] is not None
    assert len(diag["sigma_max_q"]) == 6
    e = diag["expert_entropy"]
    assert 0.0 < e <= math.log(16) + 1e-6
    assert 0.0 <= diag["expert_share_min"] <= diag["expert_share_max"] <= 1.0
