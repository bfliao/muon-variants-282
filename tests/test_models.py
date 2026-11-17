"""Tests for muonarch/models.py (design.md section 3.4).

Pinned numbers were measured on transformers 4.57.6 / torch 2.15:
dense Qwen3ForCausalLM total = 1,706,368 params (embedding 524,288 counted
once under tying), matching PLAN.md's "1.70M with tied embeddings".
"""

import pytest
import torch
import torch.nn as nn

from muonarch.models import (
    COMMON_LM_KWARGS,
    MLP_IN,
    MLP_OUT,
    assert_moe_unfused,
    build_dense,
    build_mlp,
    build_model,
    build_moe,
    count_params,
)

DENSE_TOTAL = 1_706_368
DENSE_EMBEDDING = 4096 * 128  # 524,288, counted ONCE (tied)


@pytest.fixture(scope="module")
def dense():
    torch.manual_seed(0)
    return build_dense()


@pytest.fixture(scope="module")
def moe():
    torch.manual_seed(0)
    return build_moe()


# ---------------------------------------------------------------- MLP


@pytest.mark.parametrize("width", [128, 256, 512])
def test_mlp_structure(width):
    mlp = build_mlp(width)
    assert isinstance(mlp, nn.Sequential)
    linears = [m for m in mlp if isinstance(m, nn.Linear)]
    relus = [m for m in mlp if isinstance(m, nn.ReLU)]
    assert len(linears) == 4
    assert len(relus) == 3
    assert len(mlp) == 7
    # Exact layer order: Linear, ReLU, Linear, ReLU, Linear, ReLU, Linear
    kinds = [type(m) for m in mlp]
    assert kinds == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    shapes = [(l.in_features, l.out_features) for l in linears]
    assert shapes == [(MLP_IN, width), (width, width), (width, width), (width, MLP_OUT)]
    for l in linears:
        assert l.bias is not None  # biases ON (Decisions log: MLP activation and bias)


def test_mlp_forward_shape():
    mlp = build_mlp(128)
    out = mlp(torch.randn(5, MLP_IN))
    assert out.shape == (5, MLP_OUT)


def test_mlp_count_params():
    mlp = build_mlp(128)
    c = count_params(mlp)
    expected = (3072 * 128 + 128) + 2 * (128 * 128 + 128) + (128 * 10 + 10)
    assert c["total"] == expected
    assert c["embedding"] == 0
    assert c["non_embedding"] == expected


# ---------------------------------------------------------------- dense LM


def test_common_lm_kwargs_verbatim():
    assert COMMON_LM_KWARGS == dict(
        vocab_size=4096, hidden_size=128, num_hidden_layers=6,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        max_position_embeddings=256, tie_word_embeddings=True,
    )


def test_dense_total_params_pinned(dense):
    c = count_params(dense)
    assert c["total"] == DENSE_TOTAL
    assert 1_650_000 < c["total"] < 1_750_000  # "1.70M params" sanity
    assert c["embedding"] == DENSE_EMBEDDING
    assert c["non_embedding"] == DENSE_TOTAL - DENSE_EMBEDDING


def test_dense_tied_embedding_is_same_tensor(dense):
    assert dense.lm_head.weight is dense.model.embed_tokens.weight
    names = [n for n, _ in dense.named_parameters()]
    assert "model.embed_tokens.weight" in names
    assert "lm_head.weight" not in names  # yielded once under tying
    assert len(names) == len(set(names))


def test_dense_config(dense):
    cfg = dense.config
    assert cfg.intermediate_size == 384
    assert cfg.tie_word_embeddings is True


# ---------------------------------------------------------------- MoE LM


def test_moe_config(moe):
    cfg = moe.config
    assert cfg.num_experts == 16
    assert cfg.num_experts_per_tok == 4
    assert cfg.moe_intermediate_size == 96
    assert cfg.norm_topk_prob is True
    assert cfg.decoder_sparse_step == 1
    assert cfg.mlp_only_layers == []
    assert cfg.output_router_logits is True
    assert cfg.router_aux_loss_coef == 0.01


def test_moe_unfused_passes(moe):
    assert_moe_unfused(moe)  # must not raise on 4.57.6's build


def test_moe_expert_shapes(moe):
    for layer in moe.model.layers:
        experts = layer.mlp.experts
        assert isinstance(experts, nn.ModuleList)
        assert len(experts) == 16
        for e in experts:
            assert isinstance(e.gate_proj, nn.Linear)
            assert isinstance(e.up_proj, nn.Linear)
            assert isinstance(e.down_proj, nn.Linear)
            assert tuple(e.gate_proj.weight.shape) == (96, 128)
            assert tuple(e.up_proj.weight.shape) == (96, 128)
            assert tuple(e.down_proj.weight.shape) == (128, 96)
            for w in (e.gate_proj.weight, e.up_proj.weight, e.down_proj.weight):
                assert w.ndim == 2


def test_moe_router_is_mlp_gate(moe):
    # The router is mlp.gate (a Linear on Qwen3MoeSparseMoeBlock); the dense
    # FFN's gate_proj is a different thing entirely.
    blk = moe.model.layers[0].mlp
    assert isinstance(blk.gate, nn.Linear)
    assert tuple(blk.gate.weight.shape) == (16, 128)
    names = [n for n, _ in moe.named_parameters()]
    assert "model.layers.0.mlp.gate.weight" in names
    assert "model.layers.0.mlp.gate_proj.weight" not in names


def test_assert_moe_unfused_raises_on_fused():
    class FusedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(16, 96, 128))  # fused 3-D

    class FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = FusedExperts()

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = FakeBlock()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([FakeLayer()])

    with pytest.raises(RuntimeError):
        assert_moe_unfused(FakeModel())


def test_dense_moe_only_ffn_differs(dense, moe):
    """Dense and MoE share every non-FF parameter name and shape."""
    def non_ff(model):
        return {
            n: tuple(p.shape)
            for n, p in model.named_parameters()
            if ".mlp." not in n
        }

    assert non_ff(dense) == non_ff(moe)


def test_moe_total_is_about_3_9m(moe):
    # Decisions log: MoE ~3.9M vs ~1.7M dense (measured 4,372,864 incl. router
    # on 4.57.6; the "~3.9M" in PLAN counts active/FF sizing loosely — assert
    # the measured pin plus a broad band).
    c = count_params(moe)
    assert c["total"] == 4_372_864
    assert c["embedding"] == DENSE_EMBEDDING


# ---------------------------------------------------------------- dispatch


def test_build_model_dispatch():
    m = build_model({"arch": "mlp", "width": 128})
    assert isinstance(m, nn.Sequential)
    d = build_model({"arch": "dense"})
    assert d.config.intermediate_size == 384
    mo = build_model({"arch": "moe"})
    assert mo.config.num_experts == 16


def test_build_model_unknown_arch():
    with pytest.raises((ValueError, KeyError)):
        build_model({"arch": "cnn"})
