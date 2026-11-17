"""Models for the three studies (design.md section 3.4).

MLP = plain ``nn.Sequential`` of 4 Linears with ReLU and biases ON (PyTorch
defaults — Decisions log "MLP activation and bias"; the bias-free variant
lives only in third_party/ for the oracle check and is never built here).

Dense / MoE transformers come from Hugging Face configs — EXACTLY PLAN.md's
"Model configs" snippet. Dense and MoE differ ONLY in the feed-forward
block (16 experts of width 96, top-4 = active width 384 like the dense
SwiGLU), the router (``mlp.gate`` — NOT the dense FFN's ``gate_proj``) and
the auxiliary loss (coef 0.01 fixed for all optimizers, norm_topk_prob=True)
— Decisions log "Dense vs MoE controls".

The tied 4096x128 embedding is ONE tensor (``lm_head.weight is
embed_tokens.weight``); ``named_parameters()`` yields it once and
``count_params`` counts it once.
"""

import torch.nn as nn
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

__all__ = [
    "MLP_IN",
    "MLP_OUT",
    "COMMON_LM_KWARGS",
    "build_mlp",
    "build_dense",
    "build_moe",
    "build_model",
    "assert_moe_unfused",
    "count_params",
]

MLP_IN, MLP_OUT = 3072, 10

# Exactly the PLAN.md "Model configs" snippet.
COMMON_LM_KWARGS: dict = dict(
    vocab_size=4096, hidden_size=128, num_hidden_layers=6,
    num_attention_heads=2, num_key_value_heads=1, head_dim=64,
    max_position_embeddings=256, tie_word_embeddings=True,
)


def build_mlp(width: int) -> nn.Sequential:
    """4-Linear ReLU MLP: 3072 -> w -> w -> w -> 10, biases ON, default init."""
    return nn.Sequential(
        nn.Linear(MLP_IN, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, MLP_OUT),
    )


def build_dense() -> Qwen3ForCausalLM:
    """Dense Qwen3: 1,706,368 params with the tied embedding counted once."""
    return Qwen3ForCausalLM(Qwen3Config(intermediate_size=384, **COMMON_LM_KWARGS))


def build_moe() -> Qwen3MoeForCausalLM:
    """MoE Qwen3: 16 experts, top-4, expert width 96; aux coef fixed at 0.01
    for all optimizers (PLAN.md snippet comment)."""
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        num_experts=16, num_experts_per_tok=4, moe_intermediate_size=96,
        norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[],
        output_router_logits=True, router_aux_loss_coef=0.01,
        **COMMON_LM_KWARGS))
    assert_moe_unfused(model)
    return model


def build_model(cfg: dict) -> nn.Module:
    """Dispatch on cfg["arch"]: mlp (needs cfg["width"]) | dense | moe.

    Seeding contract: the caller (train.py) runs ``torch.manual_seed`` right
    before this, so seed s gives identical init across optimizers.
    """
    arch = cfg["arch"]
    if arch == "mlp":
        return build_mlp(cfg["width"])
    if arch == "dense":
        return build_dense()
    if arch == "moe":
        return build_moe()
    raise ValueError(f"Unknown arch: {arch!r}")


def assert_moe_unfused(model: nn.Module) -> None:
    """Fail loudly unless every MoE layer keeps SEPARATE 2-D expert weights.

    The Protocol requires each expert's gate/up/down updated as its own
    matrix; a transformers version that fuses experts into 3-D tensors would
    silently change what "per-expert Muon" means. Requires ``experts`` to be
    an ``nn.ModuleList`` whose members have separate ``gate_proj`` /
    ``up_proj`` / ``down_proj`` ``nn.Linear`` modules with 2-D weights.
    """
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise RuntimeError("assert_moe_unfused: model has no decoder layers")
    checked = 0
    for i, layer in enumerate(layers):
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:
            continue  # a dense layer (mlp_only_layers) has no experts
        if not isinstance(experts, nn.ModuleList):
            raise RuntimeError(
                f"MoE layer {i}: experts is {type(experts).__name__}, not "
                "nn.ModuleList — experts appear fused; this transformers "
                "version cannot give per-expert Muon updates"
            )
        for e, expert in enumerate(experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(expert, proj, None)
                if not isinstance(lin, nn.Linear):
                    raise RuntimeError(
                        f"MoE layer {i} expert {e}: {proj} is "
                        f"{type(lin).__name__}, not nn.Linear — fused or "
                        "restructured experts"
                    )
                if lin.weight.ndim != 2:
                    raise RuntimeError(
                        f"MoE layer {i} expert {e}: {proj}.weight has ndim "
                        f"{lin.weight.ndim}, expected a separate 2-D matrix"
                    )
        checked += 1
    if checked == 0:
        raise RuntimeError("assert_moe_unfused: no MoE layer found")


def count_params(model: nn.Module) -> dict:
    """{total, embedding, non_embedding}; the tied embedding counts ONCE
    (``named_parameters()`` deduplicates the shared tensor)."""
    total = sum(p.numel() for _, p in model.named_parameters())
    embedding = 0
    get_emb = getattr(model, "get_input_embeddings", None)
    if callable(get_emb):
        emb = get_emb()
        if emb is not None:
            embedding = emb.weight.numel()
    return {
        "total": total,
        "embedding": embedding,
        "non_embedding": total - embedding,
    }
