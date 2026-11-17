"""Tests for ParamGroups / split_params / build_optimizers (design.md 3.1,
Protocol "Optimizers and parameter groups" table, Gate-1 semantics).

Group assignment is tested against the REAL models (transformers 4.57.6),
not name guesses: the MoE router is ``mlp.gate.weight`` (adamw group except
under muonall); the dense FFN's ``gate_proj`` is a muon matrix.
"""

import pytest
import torch
import torch.nn as nn

from muonarch.manifold import ManifoldMuon
from muonarch.models import build_dense, build_mlp, build_moe
from muonarch.optim import MuonFamily, ParamGroups, build_optimizers, split_params

OPTIMIZERS = ("adamw", "muon", "moonlight", "manifold", "muonall")
SPLIT_LIKE_MUON = ("muon", "moonlight", "manifold")


@pytest.fixture(scope="module")
def dense():
    torch.manual_seed(0)
    return build_dense()


@pytest.fixture(scope="module")
def moe():
    torch.manual_seed(0)
    return build_moe()


@pytest.fixture(scope="module")
def mlp128():
    torch.manual_seed(0)
    return build_mlp(128)


def models_by_arch(dense, moe, mlp128):
    return {"mlp": mlp128, "dense": dense, "moe": moe}


def group_names(pg: ParamGroups):
    return [n for n, _ in pg.muon], [n for n, _ in pg.adamw]


# ------------------------------------------------- completeness/disjointness


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("arch", ["mlp", "dense", "moe"])
def test_split_complete_and_disjoint(optimizer, arch, dense, moe, mlp128):
    model = models_by_arch(dense, moe, mlp128)[arch]
    pg = split_params(model, optimizer, arch)
    muon_names, adamw_names = group_names(pg)
    all_names = [n for n, _ in model.named_parameters()]
    assert sorted(muon_names + adamw_names) == sorted(all_names)
    assert not set(muon_names) & set(adamw_names)
    # identity, not just names: every assigned param is the model's tensor
    by_name = dict(model.named_parameters())
    for n, p in pg.muon + pg.adamw:
        assert p is by_name[n]


def test_adamw_gets_everything(dense):
    pg = split_params(dense, "adamw", "dense")
    assert pg.muon == []
    assert len(pg.adamw) == len(list(dense.named_parameters()))


def test_muonall_gets_everything_tied_once(dense):
    pg = split_params(dense, "muonall", "dense")
    assert pg.adamw == []
    muon_names, _ = group_names(pg)
    assert sorted(muon_names) == sorted(n for n, _ in dense.named_parameters())
    assert muon_names.count("model.embed_tokens.weight") == 1
    assert "lm_head.weight" not in muon_names


# ------------------------------------------------------------- fractions


@pytest.mark.parametrize("optimizer", SPLIT_LIKE_MUON)
def test_muon_fraction_dense_69pct(optimizer, dense):
    pg = split_params(dense, optimizer, "dense")
    frac = pg.fractions()["muon"]
    assert 0.68 < frac < 0.71  # Protocol: 69% (embeddings are 31%)


@pytest.mark.parametrize("width", [128, 256, 512])
def test_muon_fraction_mlp_over_99pct(width):
    model = build_mlp(width)
    pg = split_params(model, "muon", "mlp")
    assert pg.fractions()["muon"] > 0.99  # Protocol: >99% with the input layer


def test_fractions_sum_to_one(dense):
    f = split_params(dense, "muon", "dense").fractions()
    assert f["muon"] + f["adamw"] == pytest.approx(1.0)


# ----------------------------------------------------- group membership


@pytest.mark.parametrize("optimizer", SPLIT_LIKE_MUON)
def test_dense_muon_group_members(optimizer, dense):
    pg = split_params(dense, optimizer, "dense")
    muon_names, adamw_names = group_names(pg)
    for i in range(6):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert f"model.layers.{i}.self_attn.{proj}.weight" in muon_names
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert f"model.layers.{i}.mlp.{proj}.weight" in muon_names
    assert len(muon_names) == 6 * 7
    # adamw: tied embedding (once), all norms incl. QK-norm gains
    assert "model.embed_tokens.weight" in adamw_names
    for i in range(6):
        assert f"model.layers.{i}.self_attn.q_norm.weight" in adamw_names
        assert f"model.layers.{i}.self_attn.k_norm.weight" in adamw_names
        assert f"model.layers.{i}.input_layernorm.weight" in adamw_names
        assert f"model.layers.{i}.post_attention_layernorm.weight" in adamw_names
    assert "model.norm.weight" in adamw_names


@pytest.mark.parametrize("optimizer", SPLIT_LIKE_MUON)
def test_moe_router_in_adamw_group(optimizer, moe):
    pg = split_params(moe, optimizer, "moe")
    muon_names, adamw_names = group_names(pg)
    for i in range(6):
        assert f"model.layers.{i}.mlp.gate.weight" in adamw_names
        assert f"model.layers.{i}.mlp.gate.weight" not in muon_names


def test_moe_router_in_muon_group_under_muonall(moe):
    pg = split_params(moe, "muonall", "moe")
    muon_names, _ = group_names(pg)
    for i in range(6):
        assert f"model.layers.{i}.mlp.gate.weight" in muon_names


def test_moe_muon_group_exact_contents(moe):
    """Per MoE layer: 4 attention matrices + 16 experts x 3 separate weights."""
    pg = split_params(moe, "muon", "moe")
    muon_names, _ = group_names(pg)
    for i in range(6):
        layer = [n for n in muon_names if n.startswith(f"model.layers.{i}.")]
        attn = [n for n in layer if ".self_attn." in n]
        experts = [n for n in layer if ".mlp.experts." in n]
        assert len(attn) == 4
        assert len(experts) == 16 * 3
        assert len(layer) == 4 + 48
        for e in range(16):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert f"model.layers.{i}.mlp.experts.{e}.{proj}.weight" in experts
    assert len(muon_names) == 6 * 52
    # every expert weight is its own 2-D tensor
    by_name = dict(moe.named_parameters())
    for n in muon_names:
        assert by_name[n].ndim == 2


@pytest.mark.parametrize("optimizer", SPLIT_LIKE_MUON)
def test_mlp_groups(optimizer, mlp128):
    pg = split_params(mlp128, optimizer, "mlp")
    muon_names, adamw_names = group_names(pg)
    # muon: input 3072->w and the two hidden w->w matrices
    assert sorted(muon_names) == ["0.weight", "2.weight", "4.weight"]
    # adamw: classifier w->10 plus ALL biases
    assert sorted(adamw_names) == ["0.bias", "2.bias", "4.bias", "6.bias", "6.weight"]


def test_mlp_muonall_all_params(mlp128):
    pg = split_params(mlp128, "muonall", "mlp")
    muon_names, adamw_names = group_names(pg)
    assert adamw_names == []
    assert sorted(muon_names) == sorted(n for n, _ in mlp128.named_parameters())


def test_qk_norm_gains_muon_under_muonall(dense):
    pg = split_params(dense, "muonall", "dense")
    muon_names, _ = group_names(pg)
    assert "model.layers.0.self_attn.q_norm.weight" in muon_names
    assert "model.layers.0.self_attn.k_norm.weight" in muon_names


# ----------------------------------------------------------- error paths


def test_split_unknown_optimizer(dense):
    with pytest.raises(ValueError):
        split_params(dense, "sgd", "dense")


def test_split_unknown_arch(dense):
    with pytest.raises(ValueError):
        split_params(dense, "muon", "transformer")


def test_split_raises_on_unassigned_param(dense):
    class Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner.model
            self.lm_head = inner.lm_head
            self.mystery = nn.Parameter(torch.zeros(3, 3))

    with pytest.raises(RuntimeError):
        split_params(Wrapped(dense), "muon", "dense")


# ---------------------------------------------------- ParamGroups helpers


def test_paramgroups_table_and_config(dense):
    pg = split_params(dense, "muon", "dense")
    table = pg.table()
    assert isinstance(table, str)
    assert "model.embed_tokens.weight" in table
    assert "muon" in table and "adamw" in table
    cfg_map = pg.as_config()
    assert cfg_map["model.embed_tokens.weight"] == "adamw"
    assert cfg_map["model.layers.0.self_attn.q_proj.weight"] == "muon"
    assert len(cfg_map) == len(list(dense.named_parameters()))


# -------------------------------------------------------- build_optimizers


def base_cfg(optimizer, arch, **over):
    cfg = {
        "optimizer": optimizer,
        "arch": arch,
        "lr": 0.02,
        "wd": 0.01,
        "momentum": 0.95,
        "adamw_betas": [0.9, 0.95],
        "adamw_eps": 1e-8,
        "adamw_group_lr": 0.0003,
        "adamw_group_wd": 0.1,
        "ns_steps": 5,
        "ns_dtype": "fp32",
        "manifold": None,
    }
    if optimizer == "manifold":
        cfg["wd"] = None
        cfg["manifold"] = {
            "dual_cap": 5, "tol": 1e-6, "dual_lr": 0.01, "msign_steps": 10,
        }
    cfg.update(over)
    return cfg


def n_params(opt):
    return sum(p.numel() for g in opt.param_groups for p in g["params"])


def test_build_adamw(mlp128):
    opts, pg = build_optimizers(base_cfg("adamw", "mlp"), mlp128)
    assert len(opts) == 1
    (adamw,) = opts
    assert isinstance(adamw, torch.optim.AdamW)
    g = adamw.param_groups[0]
    assert g["lr"] == 0.02
    assert g["weight_decay"] == 0.01
    assert tuple(g["betas"]) == (0.9, 0.95)
    assert g["eps"] == 1e-8
    assert n_params(adamw) == sum(p.numel() for p in mlp128.parameters())
    assert pg.muon == []


def test_build_adamw_default_betas_eps(mlp128):
    cfg = base_cfg("adamw", "mlp")
    del cfg["adamw_betas"], cfg["adamw_eps"]
    (adamw,), _ = build_optimizers(cfg, mlp128)
    g = adamw.param_groups[0]
    assert tuple(g["betas"]) == (0.9, 0.95)
    assert g["eps"] == 1e-8


def test_build_muon(dense):
    opts, pg = build_optimizers(base_cfg("muon", "dense"), dense)
    assert len(opts) == 2
    muon, adamw = opts
    assert isinstance(muon, MuonFamily)
    assert isinstance(adamw, torch.optim.AdamW)
    mg = muon.param_groups[0]
    assert mg["lr"] == 0.02
    assert mg["wd"] == 0.01          # Muon's tuned wd: Muon group ONLY
    assert mg["rule"] == "keller"
    assert mg["momentum"] == 0.95
    assert mg["ns_steps"] == 5
    assert mg["ns_dtype"] is torch.float32
    ag = adamw.param_groups[0]
    assert ag["lr"] == 0.0003        # = AdamW's tuned lr (Gate-1)
    assert ag["weight_decay"] == 0.1  # = AdamW's tuned wd (Gate-1)
    assert tuple(ag["betas"]) == (0.9, 0.95)
    assert n_params(muon) == sum(p.numel() for _, p in pg.muon)
    assert n_params(adamw) == sum(p.numel() for _, p in pg.adamw)


def test_build_moonlight_shared_lr_wd(dense):
    opts, _ = build_optimizers(base_cfg("moonlight", "dense"), dense)
    muon, adamw = opts
    assert muon.param_groups[0]["rule"] == "moonlight"
    assert muon.param_groups[0]["lr"] == 0.02
    assert muon.param_groups[0]["wd"] == 0.01
    ag = adamw.param_groups[0]
    assert ag["lr"] == 0.02           # shared by design
    assert ag["weight_decay"] == 0.01


def test_build_manifold(dense):
    opts, pg = build_optimizers(base_cfg("manifold", "dense"), dense)
    assert len(opts) == 2
    mani, adamw = opts
    assert isinstance(mani, ManifoldMuon)
    g = mani.param_groups[0]
    assert g["lr"] == 0.02
    assert g["dual_cap"] == 5
    assert g["tol"] == 1e-6
    assert g["dual_lr"] == 0.01
    assert g["momentum"] == 0.95
    assert g["msign_steps"] == 10
    ag = adamw.param_groups[0]
    assert ag["lr"] == 0.0003
    assert ag["weight_decay"] == 0.1
    assert n_params(mani) == sum(p.numel() for _, p in pg.muon)


def test_build_muonall(dense):
    opts, pg = build_optimizers(base_cfg("muonall", "dense"), dense)
    assert len(opts) == 1
    (muon,) = opts
    assert isinstance(muon, MuonFamily)
    g = muon.param_groups[0]
    assert g["rule"] == "keller"
    assert g["lr"] == 0.02
    assert g["wd"] == 0.01
    assert pg.adamw == []
    assert n_params(muon) == sum(p.numel() for _, p in dense.named_parameters())


def test_build_unknown_optimizer(mlp128):
    with pytest.raises(ValueError):
        build_optimizers(base_cfg("sgd", "mlp"), mlp128)


# ------------------------------------------------------------- smoke step


def step_all(model, cfg, loss):
    opts, _ = build_optimizers(cfg, model)
    model.zero_grad(set_to_none=True)
    loss.backward()
    before = [p.detach().clone() for p in model.parameters()]
    for opt in opts:
        opt.step()
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    for p in after:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_smoke_step_mlp(optimizer):
    torch.manual_seed(0)
    model = build_mlp(128)
    x = torch.randn(4, 3072)
    y = torch.randint(0, 10, (4,))
    loss = nn.functional.cross_entropy(model(x), y)
    step_all(model, base_cfg(optimizer, "mlp"), loss)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_smoke_step_moe(optimizer):
    """One real step on the 16-expert MoE — exercises ManifoldMuon's
    batched-expert path on the real layout (6 x (4 attn + 48 expert) = 312
    muon matrices)."""
    torch.manual_seed(0)
    model = build_moe()
    ids = torch.randint(0, 4096, (2, 32))
    loss = model(input_ids=ids, labels=ids).loss

    opts, pg = build_optimizers(base_cfg(optimizer, "moe"), model)
    model.zero_grad(set_to_none=True)
    loss.backward()
    before = [p.detach().clone() for p in model.parameters()]
    stats = [opt.step() for opt in opts]
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    for p in after:
        assert torch.isfinite(p).all()

    if optimizer == "manifold":
        assert len(pg.muon) == 6 * 52  # 312 constrained matrices
        # step() stats are per matrix WITH a grad: on a tiny batch a few of
        # the 16x6 experts receive no tokens, so their 3 matrices have no
        # grad and are skipped (ManifoldMuon's documented convention).
        n_with_grad = sum(1 for _, p in pg.muon if p.grad is not None)
        assert n_with_grad > 250  # most of the 312 were routed tokens
        s = stats[0]  # ManifoldMuon.step() -> per-matrix lists
        assert len(s["dual_steps"]) == n_with_grad
        assert len(s["dual_residual"]) == n_with_grad
        assert len(s["msign_calls"]) == n_with_grad
        assert all(c >= 1 for c in s["msign_calls"])


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_smoke_step_dense(optimizer):
    torch.manual_seed(0)
    model = build_dense()
    ids = torch.randint(0, 4096, (2, 16))
    out = model(input_ids=ids, labels=ids)
    step_all(model, base_cfg(optimizer, "dense"), out.loss)
